//! Device-agnostic speculative-decoding schedule for Qwen3.5 MTP.
//!
//! This module owns only transactional token/position accounting. The CUDA
//! runtime produces two draft tokens, verifies three target rows, applies the
//! described cache/state commit, and calls [`Qwen35MtpSchedule::commit`] only
//! after both target and draft device preparations succeed.

use std::fmt;

pub const QWEN35_MTP_DRAFT_STEPS: usize = 2;
pub const QWEN35_MTP_VERIFY_ROWS: usize = QWEN35_MTP_DRAFT_STEPS + 1;

pub type TokenId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommittedPositions {
    pub base: usize,
    pub mtp: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Qwen35MtpStats {
    pub rounds: usize,
    pub zero_accept_rounds: usize,
    pub partial_accept_rounds: usize,
    pub full_accept_rounds: usize,
    pub proposed_tokens: usize,
    /// Matching draft tokens before output-budget or EOS truncation.
    pub verified_accepted_proposals: usize,
    /// Matching draft tokens actually exposed to the caller.
    pub committed_accepted_proposals: usize,
    pub correction_tokens: usize,
    pub bonus_tokens: usize,
    pub emitted_tokens: usize,
    pub proposed_by_depth: [usize; QWEN35_MTP_DRAFT_STEPS],
    pub matched_by_depth: [usize; QWEN35_MTP_DRAFT_STEPS],
    pub accepted_prefix_by_depth: [usize; QWEN35_MTP_DRAFT_STEPS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailTokenKind {
    /// The target token at the first draft mismatch.
    Correction,
    /// The target token after all drafts matched.
    Bonus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnShadowCommit {
    /// No verified row becomes visible. Used only when a device round fails.
    KeepPrimary,
    /// Replay exactly `rows` stashed recurrent inputs into primary state.
    ReplayPrefix { rows: usize },
    /// All verified rows are committed, so the complete shadow is valid.
    AdoptFullShadow,
}

/// Half-open positions affected by a speculative target pass.
///
/// `[start, committed_end)` becomes visible. The remaining verified shadow
/// `[committed_end, verified_end)` must be ignored and may be overwritten by
/// the next round. This same logical range applies to target GDN and KV state;
/// a backend may use replay/copy for GDN and absolute-slot visibility for KV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowCommitRange {
    pub start: usize,
    pub committed_end: usize,
    pub verified_end: usize,
}

impl ShadowCommitRange {
    pub fn committed_rows(self) -> usize {
        self.committed_end - self.start
    }

    pub fn discarded_rows(self) -> usize {
        self.verified_end - self.committed_end
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetCommit {
    pub gdn: GdnShadowCommit,
    pub gdn_range: ShadowCommitRange,
    pub kv_range: ShadowCommitRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpCommit {
    /// MTP KV rows already staged by the draft chain and made visible now.
    pub cache_range: ShadowCommitRange,
    /// Full acceptance plus a bonus consumes the final proposal as an input.
    /// The draft calls have not materialized that final cache row, so a
    /// runtime must run the MTP layer once more without projecting logits.
    pub materialize_final_proposal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitContract {
    pub target: TargetCommit,
    pub mtp: MtpCommit,
    pub next_positions: CommittedPositions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundTicket {
    epoch: u64,
    positions: CommittedPositions,
    remaining_budget: usize,
    proposals: [TokenId; QWEN35_MTP_DRAFT_STEPS],
    verification_inputs: [TokenId; QWEN35_MTP_VERIFY_ROWS],
}

impl RoundTicket {
    /// Target verification consumes `pending` followed by both drafts.
    pub fn verification_inputs(&self) -> &[TokenId; QWEN35_MTP_VERIFY_ROWS] {
        &self.verification_inputs
    }

    pub fn proposals(&self) -> &[TokenId; QWEN35_MTP_DRAFT_STEPS] {
        &self.proposals
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundPlan {
    epoch: u64,
    start_positions: CommittedPositions,
    accepted_proposals: usize,
    emitted: Vec<TokenId>,
    tail_kind: TailTokenKind,
    ends_with_eos: bool,
    reaches_budget: bool,
    proposal_matches: [bool; QWEN35_MTP_DRAFT_STEPS],
    contract: CommitContract,
}

impl RoundPlan {
    pub fn accepted_proposals(&self) -> usize {
        self.accepted_proposals
    }

    pub fn emitted(&self) -> &[TokenId] {
        &self.emitted
    }

    pub fn tail_kind(&self) -> TailTokenKind {
        self.tail_kind
    }

    pub fn commit_contract(&self) -> CommitContract {
        self.contract
    }

    pub fn finishes_generation(&self) -> bool {
        self.ends_with_eos || self.reaches_budget
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundOutcome {
    pub emitted: Vec<TokenId>,
    pub accepted_proposals: usize,
    pub tail_kind: TailTokenKind,
    pub contract: CommitContract,
    pub finished: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MtpSchedule {
    positions: CommittedPositions,
    output_budget: usize,
    emitted_tokens: usize,
    eos_token_ids: Vec<TokenId>,
    finished: bool,
    epoch: u64,
    stats: Qwen35MtpStats,
}

impl Qwen35MtpSchedule {
    pub fn new(
        base_position: usize,
        mtp_position: usize,
        output_budget: usize,
        already_emitted: usize,
        eos_token_ids: Vec<TokenId>,
    ) -> Result<Self, MtpScheduleError> {
        if base_position != mtp_position {
            return Err(MtpScheduleError::PositionDivergence {
                base: base_position,
                mtp: mtp_position,
            });
        }
        if already_emitted > output_budget {
            return Err(MtpScheduleError::BudgetExceeded {
                emitted: already_emitted,
                budget: output_budget,
            });
        }
        Ok(Self {
            positions: CommittedPositions {
                base: base_position,
                mtp: mtp_position,
            },
            output_budget,
            emitted_tokens: already_emitted,
            eos_token_ids,
            finished: already_emitted == output_budget,
            epoch: 0,
            stats: Qwen35MtpStats::default(),
        })
    }

    pub fn positions(&self) -> CommittedPositions {
        self.positions
    }

    pub fn remaining_budget(&self) -> usize {
        self.output_budget - self.emitted_tokens
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn stats(&self) -> &Qwen35MtpStats {
        &self.stats
    }

    /// Snapshot a speculative round without changing committed state.
    ///
    /// Draft or target failure is represented by dropping the returned ticket;
    /// positions, statistics, and visible cache ranges remain unchanged.
    pub fn begin_round(
        &self,
        pending: TokenId,
        proposals: &[TokenId],
    ) -> Result<RoundTicket, MtpScheduleError> {
        if self.finished {
            return Err(MtpScheduleError::Finished);
        }
        if self.eos_token_ids.contains(&pending) {
            return Err(MtpScheduleError::PendingIsEos(pending));
        }
        let proposals: [TokenId; QWEN35_MTP_DRAFT_STEPS] =
            proposals
                .try_into()
                .map_err(|_| MtpScheduleError::WrongProposalCount {
                    expected: QWEN35_MTP_DRAFT_STEPS,
                    got: proposals.len(),
                })?;
        let mut verification_inputs = [pending; QWEN35_MTP_VERIFY_ROWS];
        verification_inputs[1..].copy_from_slice(&proposals);
        Ok(RoundTicket {
            epoch: self.epoch,
            positions: self.positions,
            remaining_budget: self.remaining_budget(),
            proposals,
            verification_inputs,
        })
    }

    /// Resolve draft acceptance and produce an immutable device commit plan.
    ///
    /// `target_predictions[i]` is the target argmax after verification input
    /// row `i`. Hence rows 0..3 check the drafts, and row 3 becomes the bonus
    /// token only when every draft matched.
    pub fn finish_round(
        &self,
        ticket: RoundTicket,
        target_predictions: &[TokenId],
    ) -> Result<RoundPlan, MtpScheduleError> {
        self.validate_ticket(&ticket)?;
        let target_predictions: [TokenId; QWEN35_MTP_VERIFY_ROWS] =
            target_predictions
                .try_into()
                .map_err(|_| MtpScheduleError::WrongPredictionCount {
                    expected: QWEN35_MTP_VERIFY_ROWS,
                    got: target_predictions.len(),
                })?;

        let mut proposal_matches = [false; QWEN35_MTP_DRAFT_STEPS];
        for depth in 0..QWEN35_MTP_DRAFT_STEPS {
            proposal_matches[depth] = ticket.proposals[depth] == target_predictions[depth];
        }
        let accepted_proposals = proposal_matches
            .iter()
            .take_while(|matched| **matched)
            .count();
        let tail_kind = if accepted_proposals == QWEN35_MTP_DRAFT_STEPS {
            TailTokenKind::Bonus
        } else {
            TailTokenKind::Correction
        };

        let mut emitted = ticket.proposals[..accepted_proposals].to_vec();
        emitted.push(target_predictions[accepted_proposals]);
        if let Some(eos_index) = emitted
            .iter()
            .position(|token| self.eos_token_ids.contains(token))
        {
            emitted.truncate(eos_index + 1);
        }
        emitted.truncate(ticket.remaining_budget);
        debug_assert!(!emitted.is_empty());

        let committed_rows = emitted.len();
        let start = ticket.positions.base;
        let committed_end = checked_add(start, committed_rows)?;
        let verified_end = checked_add(start, QWEN35_MTP_VERIFY_ROWS)?;
        let target_range = ShadowCommitRange {
            start,
            committed_end,
            verified_end,
        };
        let gdn = if committed_rows == QWEN35_MTP_VERIFY_ROWS {
            GdnShadowCommit::AdoptFullShadow
        } else {
            GdnShadowCommit::ReplayPrefix {
                rows: committed_rows,
            }
        };

        // Three draft calls stage inputs pending, d0 and d1. Consuming the
        // bonus also consumes d2, which must be materialized before commit.
        let materialize_final_proposal = committed_rows == QWEN35_MTP_VERIFY_ROWS;
        let mtp_staged_rows = if materialize_final_proposal {
            QWEN35_MTP_VERIFY_ROWS
        } else {
            QWEN35_MTP_DRAFT_STEPS
        };
        let mtp_range = ShadowCommitRange {
            start: ticket.positions.mtp,
            committed_end: checked_add(ticket.positions.mtp, committed_rows)?,
            verified_end: checked_add(ticket.positions.mtp, mtp_staged_rows)?,
        };
        let next_positions = CommittedPositions {
            base: committed_end,
            mtp: mtp_range.committed_end,
        };
        let ends_with_eos = emitted
            .last()
            .is_some_and(|token| self.eos_token_ids.contains(token));
        let reaches_budget = emitted.len() == ticket.remaining_budget;

        Ok(RoundPlan {
            epoch: ticket.epoch,
            start_positions: ticket.positions,
            accepted_proposals,
            emitted,
            tail_kind,
            ends_with_eos,
            reaches_budget,
            proposal_matches,
            contract: CommitContract {
                target: TargetCommit {
                    gdn,
                    gdn_range: target_range,
                    kv_range: target_range,
                },
                mtp: MtpCommit {
                    cache_range: mtp_range,
                    materialize_final_proposal,
                },
                next_positions,
            },
        })
    }

    /// Publish a successfully executed plan atomically in host state.
    ///
    /// The caller must first complete all fallible target/MTP cache operations
    /// described by `plan.commit_contract()`. Dropping a plan on failure leaves
    /// this schedule unchanged. Stale or duplicate plans are rejected before
    /// any mutation.
    pub fn commit(&mut self, plan: RoundPlan) -> Result<RoundOutcome, MtpScheduleError> {
        if plan.epoch != self.epoch || plan.start_positions != self.positions {
            return Err(MtpScheduleError::StaleRound);
        }
        let new_emitted_tokens = self
            .emitted_tokens
            .checked_add(plan.emitted.len())
            .ok_or(MtpScheduleError::PositionOverflow)?;
        if new_emitted_tokens > self.output_budget {
            return Err(MtpScheduleError::BudgetExceeded {
                emitted: new_emitted_tokens,
                budget: self.output_budget,
            });
        }
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or(MtpScheduleError::PositionOverflow)?;

        let mut stats = self.stats.clone();
        stats.rounds += 1;
        stats.proposed_tokens += QWEN35_MTP_DRAFT_STEPS;
        stats.verified_accepted_proposals += plan.accepted_proposals;
        stats.committed_accepted_proposals += plan.accepted_proposals.min(plan.emitted.len());
        stats.emitted_tokens += plan.emitted.len();
        match plan.accepted_proposals {
            0 => stats.zero_accept_rounds += 1,
            QWEN35_MTP_DRAFT_STEPS => stats.full_accept_rounds += 1,
            _ => stats.partial_accept_rounds += 1,
        }
        let tail_was_emitted = plan.emitted.len() > plan.accepted_proposals;
        if tail_was_emitted {
            match plan.tail_kind {
                TailTokenKind::Correction => stats.correction_tokens += 1,
                TailTokenKind::Bonus => stats.bonus_tokens += 1,
            }
        }
        let mut prefix = true;
        for depth in 0..QWEN35_MTP_DRAFT_STEPS {
            stats.proposed_by_depth[depth] += 1;
            if plan.proposal_matches[depth] {
                stats.matched_by_depth[depth] += 1;
            } else {
                prefix = false;
            }
            if prefix {
                stats.accepted_prefix_by_depth[depth] += 1;
            }
        }

        let finished = plan.finishes_generation();
        self.positions = plan.contract.next_positions;
        self.emitted_tokens = new_emitted_tokens;
        self.finished = finished;
        self.epoch = next_epoch;
        self.stats = stats;

        Ok(RoundOutcome {
            emitted: plan.emitted,
            accepted_proposals: plan.accepted_proposals,
            tail_kind: plan.tail_kind,
            contract: plan.contract,
            finished,
        })
    }

    fn validate_ticket(&self, ticket: &RoundTicket) -> Result<(), MtpScheduleError> {
        if self.finished {
            return Err(MtpScheduleError::Finished);
        }
        if ticket.epoch != self.epoch
            || ticket.positions != self.positions
            || ticket.remaining_budget != self.remaining_budget()
        {
            return Err(MtpScheduleError::StaleRound);
        }
        Ok(())
    }
}

fn checked_add(left: usize, right: usize) -> Result<usize, MtpScheduleError> {
    left.checked_add(right)
        .ok_or(MtpScheduleError::PositionOverflow)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MtpScheduleError {
    PositionDivergence { base: usize, mtp: usize },
    BudgetExceeded { emitted: usize, budget: usize },
    WrongProposalCount { expected: usize, got: usize },
    WrongPredictionCount { expected: usize, got: usize },
    PendingIsEos(TokenId),
    Finished,
    StaleRound,
    PositionOverflow,
}

impl fmt::Display for MtpScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionDivergence { base, mtp } => {
                write!(f, "base position {base} differs from MTP position {mtp}")
            }
            Self::BudgetExceeded { emitted, budget } => {
                write!(f, "emitted token count {emitted} exceeds budget {budget}")
            }
            Self::WrongProposalCount { expected, got } => {
                write!(f, "expected {expected} MTP proposals, got {got}")
            }
            Self::WrongPredictionCount { expected, got } => {
                write!(f, "expected {expected} target predictions, got {got}")
            }
            Self::PendingIsEos(token) => write!(f, "pending token {token} is EOS"),
            Self::Finished => write!(f, "generation is already finished"),
            Self::StaleRound => write!(f, "speculative round no longer matches committed state"),
            Self::PositionOverflow => write!(f, "speculative position overflow"),
        }
    }
}

impl std::error::Error for MtpScheduleError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(budget: usize) -> Qwen35MtpSchedule {
        // One token has already been sampled by target prefill.
        Qwen35MtpSchedule::new(7, 7, budget, 1, vec![99]).unwrap()
    }

    fn verify_incrementing(
        inputs: &[TokenId; QWEN35_MTP_VERIFY_ROWS],
    ) -> [TokenId; QWEN35_MTP_VERIFY_ROWS] {
        inputs.map(|token| token + 1)
    }

    fn round(
        schedule: &mut Qwen35MtpSchedule,
        pending: TokenId,
        proposals: [TokenId; QWEN35_MTP_DRAFT_STEPS],
    ) -> RoundOutcome {
        let ticket = schedule.begin_round(pending, &proposals).unwrap();
        let predictions = verify_incrementing(ticket.verification_inputs());
        let plan = schedule.finish_round(ticket, &predictions).unwrap();
        schedule.commit(plan).unwrap()
    }

    #[test]
    fn two_step_zero_partial_and_full_accept_are_scalar_equivalent() {
        let mut schedule = schedule(13);
        let mut pending = 10;
        let mut speculative = vec![pending];

        let zero = round(&mut schedule, pending, [90, 91]);
        assert_eq!(zero.accepted_proposals, 0);
        assert_eq!(zero.tail_kind, TailTokenKind::Correction);
        assert_eq!(zero.emitted, [11]);
        pending = *zero.emitted.last().unwrap();
        speculative.extend(zero.emitted);

        let partial = round(&mut schedule, pending, [12, 80]);
        assert_eq!(partial.accepted_proposals, 1);
        assert_eq!(partial.emitted, [12, 13]);
        pending = *partial.emitted.last().unwrap();
        speculative.extend(partial.emitted);

        let full = round(&mut schedule, pending, [14, 15]);
        assert_eq!(full.accepted_proposals, 2);
        assert_eq!(full.tail_kind, TailTokenKind::Bonus);
        assert_eq!(full.emitted, [14, 15, 16]);
        speculative.extend(full.emitted);

        let scalar = (10..=16).collect::<Vec<_>>();
        assert_eq!(speculative, scalar);
        assert_eq!(
            schedule.positions(),
            CommittedPositions { base: 13, mtp: 13 }
        );
        assert_eq!(schedule.stats().zero_accept_rounds, 1);
        assert_eq!(schedule.stats().partial_accept_rounds, 1);
        assert_eq!(schedule.stats().full_accept_rounds, 1);
        assert_eq!(schedule.stats().verified_accepted_proposals, 3);
        assert_eq!(schedule.stats().correction_tokens, 2);
        assert_eq!(schedule.stats().bonus_tokens, 1);
    }

    #[test]
    fn full_accept_commits_bonus_and_complete_gdn_kv_shadow() {
        let mut schedule = schedule(8);
        let outcome = round(&mut schedule, 20, [21, 22]);
        assert_eq!(outcome.emitted, [21, 22, 23]);
        assert_eq!(
            outcome.contract.target.gdn,
            GdnShadowCommit::AdoptFullShadow
        );
        assert_eq!(
            outcome.contract.target.gdn_range,
            ShadowCommitRange {
                start: 7,
                committed_end: 10,
                verified_end: 10,
            }
        );
        assert_eq!(
            outcome.contract.target.kv_range,
            outcome.contract.target.gdn_range
        );
        assert!(outcome.contract.mtp.materialize_final_proposal);
        assert_eq!(
            outcome.contract.next_positions,
            CommittedPositions { base: 10, mtp: 10 }
        );
    }

    #[test]
    fn partial_accept_commits_only_prefix_and_discards_shadow_suffix() {
        let mut schedule = schedule(8);
        let outcome = round(&mut schedule, 20, [21, 70]);
        assert_eq!(outcome.emitted, [21, 22]);
        assert_eq!(
            outcome.contract.target.gdn,
            GdnShadowCommit::ReplayPrefix { rows: 2 }
        );
        assert_eq!(outcome.contract.target.gdn_range.committed_rows(), 2);
        assert_eq!(outcome.contract.target.gdn_range.discarded_rows(), 1);
        assert_eq!(outcome.contract.target.kv_range.discarded_rows(), 1);
        assert_eq!(outcome.contract.mtp.cache_range.committed_rows(), 2);
        assert_eq!(outcome.contract.mtp.cache_range.discarded_rows(), 0);
        assert!(!outcome.contract.mtp.materialize_final_proposal);
    }

    #[test]
    fn eos_and_output_budget_truncate_before_bonus_without_overcommit() {
        let mut eos_schedule = schedule(8);
        let ticket = eos_schedule.begin_round(97, &[98, 70]).unwrap();
        let predictions = verify_incrementing(ticket.verification_inputs());
        let plan = eos_schedule.finish_round(ticket, &predictions).unwrap();
        assert_eq!(plan.emitted(), [98, 99]);
        let eos = eos_schedule.commit(plan).unwrap();
        assert!(eos.finished);
        assert_eq!(
            eos.contract.next_positions,
            CommittedPositions { base: 9, mtp: 9 }
        );
        assert!(!eos.contract.mtp.materialize_final_proposal);

        // Only two slots remain, so a full match publishes both drafts and no
        // bonus even though all target rows were verified.
        let mut budget_schedule = schedule(3);
        let budget = round(&mut budget_schedule, 30, [31, 32]);
        assert_eq!(budget.emitted, [31, 32]);
        assert!(budget.finished);
        assert_eq!(budget_schedule.stats().bonus_tokens, 0);
        assert_eq!(budget_schedule.stats().committed_accepted_proposals, 2);
        assert_eq!(
            budget.contract.next_positions,
            CommittedPositions { base: 9, mtp: 9 }
        );
    }

    #[test]
    fn failed_or_stale_round_never_commits_positions_or_statistics() {
        let mut schedule = schedule(8);
        let before = schedule.clone();
        let ticket = schedule.begin_round(20, &[21, 22]).unwrap();
        assert!(matches!(
            schedule.finish_round(ticket.clone(), &[21, 22]),
            Err(MtpScheduleError::WrongPredictionCount { .. })
        ));
        assert_eq!(schedule, before);

        let predictions = verify_incrementing(ticket.verification_inputs());
        let plan = schedule.finish_round(ticket, &predictions).unwrap();
        let stale = plan.clone();
        schedule.commit(plan).unwrap();
        let committed = schedule.clone();
        assert_eq!(schedule.commit(stale), Err(MtpScheduleError::StaleRound));
        assert_eq!(schedule, committed);
    }

    #[test]
    fn constructor_and_round_boundaries_reject_invalid_state() {
        assert!(matches!(
            Qwen35MtpSchedule::new(4, 3, 8, 1, vec![]),
            Err(MtpScheduleError::PositionDivergence { .. })
        ));
        assert!(matches!(
            Qwen35MtpSchedule::new(4, 4, 1, 2, vec![]),
            Err(MtpScheduleError::BudgetExceeded { .. })
        ));
        let schedule = schedule(8);
        assert!(matches!(
            schedule.begin_round(20, &[21]),
            Err(MtpScheduleError::WrongProposalCount { .. })
        ));
        assert!(matches!(
            schedule.begin_round(99, &[100, 101]),
            Err(MtpScheduleError::PendingIsEos(99))
        ));
    }
}
