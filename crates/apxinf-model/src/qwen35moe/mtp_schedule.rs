//! Device-agnostic speculative-decoding schedule for Qwen3.5 MTP.
//!
//! This module owns only transactional token/position accounting. The CUDA
//! runtime produces one to three draft tokens, verifies two to four target rows, applies the
//! described cache/state commit, and calls [`Qwen35MtpSchedule::commit`] only
//! after both target and draft device preparations succeed.

use std::fmt;

pub const QWEN35_MTP_DRAFT_STEPS: usize = 3;
pub const QWEN35_MTP_VERIFY_ROWS: usize = QWEN35_MTP_DRAFT_STEPS + 1;
pub const QWEN35_MTP_MIN_DRAFT_STEPS: usize = 1;

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
    pub target_verified_rows: usize,
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
    proposals: Vec<TokenId>,
    verification_inputs: Vec<TokenId>,
}

impl RoundTicket {
    /// Target verification consumes `pending` followed by every draft.
    pub fn verification_inputs(&self) -> &[TokenId] {
        &self.verification_inputs
    }

    pub fn proposals(&self) -> &[TokenId] {
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
    proposal_matches: Vec<bool>,
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
        if !(QWEN35_MTP_MIN_DRAFT_STEPS..=QWEN35_MTP_DRAFT_STEPS)
            .contains(&proposals.len())
        {
            return Err(MtpScheduleError::DraftDepthOutOfRange {
                min: QWEN35_MTP_MIN_DRAFT_STEPS,
                max: QWEN35_MTP_DRAFT_STEPS,
                got: proposals.len(),
            });
        }
        let proposals = proposals.to_vec();
        let mut verification_inputs = Vec::with_capacity(proposals.len() + 1);
        verification_inputs.push(pending);
        verification_inputs.extend_from_slice(&proposals);
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
    /// row `i`. The first `ticket.proposals.len()` rows check drafts and the
    /// final row becomes the bonus token only when every draft matched.
    pub fn finish_round(
        &self,
        ticket: RoundTicket,
        target_predictions: &[TokenId],
    ) -> Result<RoundPlan, MtpScheduleError> {
        self.validate_ticket(&ticket)?;
        let draft_steps = ticket.proposals.len();
        let verify_rows = draft_steps + 1;
        if target_predictions.len() != verify_rows {
            return Err(MtpScheduleError::WrongPredictionCount {
                expected: verify_rows,
                got: target_predictions.len(),
            });
        }

        let mut proposal_matches = vec![false; draft_steps];
        for depth in 0..draft_steps {
            proposal_matches[depth] = ticket.proposals[depth] == target_predictions[depth];
        }
        let accepted_proposals = proposal_matches
            .iter()
            .take_while(|matched| **matched)
            .count();
        let tail_kind = if accepted_proposals == draft_steps {
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
        let verified_end = checked_add(start, verify_rows)?;
        let target_range = ShadowCommitRange {
            start,
            committed_end,
            verified_end,
        };
        let gdn = if committed_rows == verify_rows {
            GdnShadowCommit::AdoptFullShadow
        } else {
            GdnShadowCommit::ReplayPrefix {
                rows: committed_rows,
            }
        };

        // The draft chain stages `draft_steps` inputs. Consuming the
        // bonus also consumes the final proposal, which must be materialized
        // before commit.
        let materialize_final_proposal = committed_rows == verify_rows;
        let mtp_staged_rows = if materialize_final_proposal {
            verify_rows
        } else {
            draft_steps
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
        let draft_steps = plan.proposal_matches.len();
        stats.proposed_tokens += draft_steps;
        stats.verified_accepted_proposals += plan.accepted_proposals;
        stats.committed_accepted_proposals += plan.accepted_proposals.min(plan.emitted.len());
        stats.emitted_tokens += plan.emitted.len();
        stats.target_verified_rows += draft_steps + 1;
        match plan.accepted_proposals {
            0 => stats.zero_accept_rounds += 1,
            accepted if accepted == draft_steps => stats.full_accept_rounds += 1,
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
        for depth in 0..draft_steps {
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
    DraftDepthOutOfRange { min: usize, max: usize, got: usize },
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
            Self::DraftDepthOutOfRange { min, max, got } => {
                write!(f, "MTP draft depth must be in {min}..={max}, got {got}")
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

    fn verify_incrementing(inputs: &[TokenId]) -> Vec<TokenId> {
        inputs.iter().map(|token| token + 1).collect()
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
    fn three_step_zero_partial_and_full_accept_are_scalar_equivalent() {
        let mut schedule = schedule(15);
        let mut pending = 10;
        let mut speculative = vec![pending];

        let zero = round(&mut schedule, pending, [90, 91, 92]);
        assert_eq!(zero.accepted_proposals, 0);
        assert_eq!(zero.tail_kind, TailTokenKind::Correction);
        assert_eq!(zero.emitted, [11]);
        pending = *zero.emitted.last().unwrap();
        speculative.extend(zero.emitted);

        let partial = round(&mut schedule, pending, [12, 80, 81]);
        assert_eq!(partial.accepted_proposals, 1);
        assert_eq!(partial.emitted, [12, 13]);
        pending = *partial.emitted.last().unwrap();
        speculative.extend(partial.emitted);

        let full = round(&mut schedule, pending, [14, 15, 16]);
        assert_eq!(full.accepted_proposals, 3);
        assert_eq!(full.tail_kind, TailTokenKind::Bonus);
        assert_eq!(full.emitted, [14, 15, 16, 17]);
        speculative.extend(full.emitted);

        let scalar = (10..=17).collect::<Vec<_>>();
        assert_eq!(speculative, scalar);
        assert_eq!(
            schedule.positions(),
            CommittedPositions { base: 14, mtp: 14 }
        );
        assert_eq!(schedule.stats().zero_accept_rounds, 1);
        assert_eq!(schedule.stats().partial_accept_rounds, 1);
        assert_eq!(schedule.stats().full_accept_rounds, 1);
        assert_eq!(schedule.stats().verified_accepted_proposals, 4);
        assert_eq!(schedule.stats().correction_tokens, 2);
        assert_eq!(schedule.stats().bonus_tokens, 1);
    }

    #[test]
    fn full_accept_commits_bonus_and_complete_gdn_kv_shadow() {
        let mut schedule = schedule(8);
        let outcome = round(&mut schedule, 20, [21, 22, 23]);
        assert_eq!(outcome.emitted, [21, 22, 23, 24]);
        assert_eq!(
            outcome.contract.target.gdn,
            GdnShadowCommit::AdoptFullShadow
        );
        assert_eq!(
            outcome.contract.target.gdn_range,
            ShadowCommitRange {
                start: 7,
                committed_end: 11,
                verified_end: 11,
            }
        );
        assert_eq!(
            outcome.contract.target.kv_range,
            outcome.contract.target.gdn_range
        );
        assert!(outcome.contract.mtp.materialize_final_proposal);
        assert_eq!(
            outcome.contract.next_positions,
            CommittedPositions { base: 11, mtp: 11 }
        );
    }

    #[test]
    fn partial_accept_commits_only_prefix_and_discards_shadow_suffix() {
        let mut schedule = schedule(8);
        let outcome = round(&mut schedule, 20, [21, 70, 71]);
        assert_eq!(outcome.emitted, [21, 22]);
        assert_eq!(
            outcome.contract.target.gdn,
            GdnShadowCommit::ReplayPrefix { rows: 2 }
        );
        assert_eq!(outcome.contract.target.gdn_range.committed_rows(), 2);
        assert_eq!(outcome.contract.target.gdn_range.discarded_rows(), 2);
        assert_eq!(outcome.contract.target.kv_range.discarded_rows(), 2);
        assert_eq!(outcome.contract.mtp.cache_range.committed_rows(), 2);
        assert_eq!(outcome.contract.mtp.cache_range.discarded_rows(), 1);
        assert!(!outcome.contract.mtp.materialize_final_proposal);
    }

    #[test]
    fn eos_and_output_budget_truncate_before_bonus_without_overcommit() {
        let mut eos_schedule = schedule(8);
        let ticket = eos_schedule.begin_round(97, &[98, 70, 71]).unwrap();
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

        // Only two slots remain, so a full match publishes only two drafts and
        // no bonus even though all target rows were verified.
        let mut budget_schedule = schedule(3);
        let budget = round(&mut budget_schedule, 30, [31, 32, 33]);
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
        let ticket = schedule.begin_round(20, &[21, 22, 23]).unwrap();
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
            Ok(_)
        ));
        assert!(matches!(
            schedule.begin_round(20, &[]),
            Err(MtpScheduleError::DraftDepthOutOfRange { .. })
        ));
        assert!(matches!(
            schedule.begin_round(20, &[21, 22, 23, 24]),
            Err(MtpScheduleError::DraftDepthOutOfRange { .. })
        ));
        assert!(matches!(
            schedule.begin_round(99, &[100, 101, 102]),
            Err(MtpScheduleError::PendingIsEos(99))
        ));
    }

    // A scalar model independent of the speculative schedule: after consuming
    // token x its only next token is x+1. Expected output is generated by
    // stepping this model, never by inspecting a RoundPlan or its contract.
    fn counter_next(input: TokenId) -> TokenId {
        input.checked_add(1).unwrap()
    }

    fn counter_scalar_run(
        mut pending: TokenId,
        steps: usize,
        eos: Option<TokenId>,
    ) -> (Vec<TokenId>, Vec<TokenId>) {
        let mut consumed = Vec::new();
        let mut emitted = Vec::new();
        for _ in 0..steps {
            consumed.push(pending);
            pending = counter_next(pending);
            emitted.push(pending);
            if Some(pending) == eos {
                break;
            }
        }
        (consumed, emitted)
    }

    fn counter_fixture(
        pending: TokenId,
        depth: usize,
        accepted: usize,
    ) -> (Vec<TokenId>, Vec<TokenId>, Vec<TokenId>) {
        assert!((1..=3).contains(&depth) && accepted <= depth);
        let mut proposals = Vec::new();
        let mut correct = pending;
        for slot in 0..depth {
            correct = counter_next(correct);
            // After the declared accepted prefix every draft is wrong, and
            // wrong drafts differ by17, so no suffix accidentally matches.
            proposals.push(if slot < accepted {
                correct
            } else {
                900_000 + depth as u32 * 31 + slot as u32 * 17
            });
        }
        let inputs = std::iter::once(pending)
            .chain(proposals.iter().copied())
            .collect::<Vec<_>>();
        let predictions = inputs.iter().copied().map(counter_next).collect();
        (proposals, inputs, predictions)
    }

    fn assert_unchanged(label: &str, actual: &Qwen35MtpSchedule, before: &Qwen35MtpSchedule) {
        assert_eq!(actual.positions(), before.positions(), "{label}: positions");
        assert_eq!(actual.stats(), before.stats(), "{label}: statistics");
        assert_eq!(actual, before, "{label}: complete host state");
    }

    fn expected_counter_stats(
        depth: usize,
        accepted: usize,
        proposals: &[TokenId],
        emitted: &[TokenId],
    ) -> Qwen35MtpStats {
        let mut proposed_by_depth = [0; QWEN35_MTP_DRAFT_STEPS];
        let mut matched_by_depth = [0; QWEN35_MTP_DRAFT_STEPS];
        let mut accepted_prefix_by_depth = [0; QWEN35_MTP_DRAFT_STEPS];
        for slot in 0..depth {
            proposed_by_depth[slot] = 1;
            matched_by_depth[slot] = usize::from(slot < accepted);
            accepted_prefix_by_depth[slot] = usize::from(slot < accepted);
        }
        let committed_drafts = emitted.iter().zip(proposals)
            .take_while(|(output, draft)| output == draft).count();
        let target_tail_emitted = emitted.len() > accepted;
        Qwen35MtpStats {
            rounds: 1,
            zero_accept_rounds: usize::from(accepted == 0),
            partial_accept_rounds: usize::from(accepted > 0 && accepted < depth),
            full_accept_rounds: usize::from(accepted == depth),
            proposed_tokens: depth,
            verified_accepted_proposals: accepted,
            committed_accepted_proposals: committed_drafts,
            correction_tokens: usize::from(target_tail_emitted && accepted < depth),
            bonus_tokens: usize::from(target_tail_emitted && accepted == depth),
            emitted_tokens: emitted.len(),
            target_verified_rows: depth + 1,
            proposed_by_depth,
            matched_by_depth,
            accepted_prefix_by_depth,
        }
    }

    fn add_expected_stats(total: &mut Qwen35MtpStats, round: &Qwen35MtpStats) {
        total.rounds += round.rounds;
        total.zero_accept_rounds += round.zero_accept_rounds;
        total.partial_accept_rounds += round.partial_accept_rounds;
        total.full_accept_rounds += round.full_accept_rounds;
        total.proposed_tokens += round.proposed_tokens;
        total.verified_accepted_proposals += round.verified_accepted_proposals;
        total.committed_accepted_proposals += round.committed_accepted_proposals;
        total.correction_tokens += round.correction_tokens;
        total.bonus_tokens += round.bonus_tokens;
        total.emitted_tokens += round.emitted_tokens;
        total.target_verified_rows += round.target_verified_rows;
        for slot in 0..QWEN35_MTP_DRAFT_STEPS {
            total.proposed_by_depth[slot] += round.proposed_by_depth[slot];
            total.matched_by_depth[slot] += round.matched_by_depth[slot];
            total.accepted_prefix_by_depth[slot] += round.accepted_prefix_by_depth[slot];
        }
    }

    // Only the expected device ranges are derived here. Token identities and
    // consumed rows come from the independent scalar counter above. In
    // particular the final draft needs materialization iff the scalar model
    // actually consumed that input, including EOS/budget truncation.
    fn assert_counter_contract(
        contract: CommitContract,
        start: usize,
        inputs: &[TokenId],
        proposals: &[TokenId],
        consumed: &[TokenId],
    ) {
        let materialize = consumed.contains(proposals.last().unwrap());
        let target = ShadowCommitRange {
            start,
            committed_end: start + consumed.len(),
            verified_end: start + inputs.len(),
        };
        let draft = ShadowCommitRange {
            start,
            committed_end: start + consumed.len(),
            verified_end: start + proposals.len() + usize::from(materialize),
        };
        assert_eq!(contract.target.kv_range, target);
        assert_eq!(contract.target.gdn_range, target);
        assert_eq!(contract.target.gdn, if consumed == inputs {
            GdnShadowCommit::AdoptFullShadow
        } else {
            GdnShadowCommit::ReplayPrefix { rows: consumed.len() }
        });
        assert_eq!(contract.mtp.cache_range, draft);
        assert_eq!(contract.mtp.materialize_final_proposal, materialize);
        assert_eq!(contract.next_positions, CommittedPositions {
            base: start + consumed.len(), mtp: start + consumed.len(),
        });
    }

    #[test]
    fn scalar_oracle_matrix_covers_depth_acceptance_budget_and_every_eos_row() {
        let mut cases = 0;
        let mut no_round_cases = 0;
        let mut committed_rounds = 0;
        let mut reached_eos = 0;
        let mut budget_masked_eos = 0;
        for depth in [1, 2, 3] {
            for accepted in 0..=depth {
                for budget in [1, 2, 3, 4, 1024] {
                    // The last potentially emitted row is a correction when
                    // a draft mismatches, or the bonus when all drafts match.
                    let eos_rows = std::iter::once(None)
                        .chain((0..=accepted).map(Some));
                    for eos_row in eos_rows {
                        cases += 1;
                        let pending = 100;
                        let eos = eos_row.map(|row| 101 + row as TokenId);
                        let (proposals, inputs, predictions) = counter_fixture(pending, depth, accepted);
                        let mut schedule = Qwen35MtpSchedule::new(
                            64, 64, budget, 1, eos.into_iter().collect(),
                        ).unwrap();
                        let before = schedule.clone();
                        if budget == 1 {
                            // Prefill already emitted the one allowed token.
                            assert!(schedule.is_finished());
                            assert_eq!(schedule.remaining_budget(), 0);
                            assert_eq!(schedule.stats(), &Qwen35MtpStats::default());
                            assert_eq!(schedule.begin_round(pending, &proposals), Err(MtpScheduleError::Finished));
                            assert_unchanged("prefill-only budget", &schedule, &before);
                            no_round_cases += 1;
                            continue;
                        }
                        let remaining = budget - 1;
                        let (consumed, expected) = counter_scalar_run(
                            pending, remaining.min(accepted + 1), eos,
                        );
                        let ticket = schedule.begin_round(pending, &proposals).unwrap();
                        assert_eq!(ticket.verification_inputs(), inputs);
                        assert_unchanged("begin is read-only", &schedule, &before);
                        let plan = schedule.finish_round(ticket, &predictions).unwrap();
                        assert_unchanged("finish is read-only", &schedule, &before);
                        assert_eq!(plan.emitted(), expected);
                        assert_eq!(plan.accepted_proposals(), accepted);
                        assert_eq!(plan.tail_kind(), if accepted == depth {
                            TailTokenKind::Bonus
                        } else { TailTokenKind::Correction });
                        assert_counter_contract(plan.commit_contract(), 64, &inputs, &proposals, &consumed);
                        let eos_was_emitted = eos.is_some() && expected.last().copied() == eos;
                        let finished = expected.len() == remaining || eos_was_emitted;
                        assert_eq!(plan.finishes_generation(), finished);
                        let outcome = schedule.commit(plan).unwrap();
                        assert_eq!(outcome.emitted, expected);
                        assert_eq!(outcome.accepted_proposals, accepted);
                        assert_eq!(outcome.finished, finished);
                        assert_counter_contract(outcome.contract, 64, &inputs, &proposals, &consumed);
                        assert_eq!(schedule.positions(), outcome.contract.next_positions);
                        assert_eq!(schedule.remaining_budget(), remaining - expected.len());
                        assert_eq!(schedule.is_finished(), finished);
                        assert_eq!(schedule.emitted_tokens, 1 + expected.len());
                        assert_eq!(schedule.epoch, before.epoch + 1);
                        assert_eq!(schedule.stats(), &expected_counter_stats(depth, accepted, &proposals, &expected));
                        committed_rounds += 1;
                        reached_eos += usize::from(eos_was_emitted);
                        budget_masked_eos += usize::from(eos.is_some() && !eos_was_emitted);
                    }
                }
            }
        }
        assert_eq!((cases, no_round_cases, committed_rounds, reached_eos, budget_masked_eos),
                   (140, 28, 112, 61, 15));
        eprintln!("mtp-schedule scalar-oracle cases={cases} budget1_no_round={no_round_cases} committed_rounds={committed_rounds} reached_eos={reached_eos} budget_masked_eos={budget_masked_eos}");
    }

    #[test]
    fn scalar_oracle_repeated_rounds_reach_1024_for_all_acceptance_prefixes() {
        let mut cases = 0;
        let mut checked_rounds = 0;
        for depth in [1, 2, 3] {
            for accepted in 0..=depth {
                let mut schedule = Qwen35MtpSchedule::new(64, 64, 1024, 1, vec![]).unwrap();
                let mut tokens = vec![100];
                let mut stats = Qwen35MtpStats::default();
                while !schedule.is_finished() {
                    let pending = *tokens.last().unwrap();
                    let start = 64 + tokens.len() - 1;
                    let (proposals, inputs, predictions) = counter_fixture(pending, depth, accepted);
                    let (consumed, expected) = counter_scalar_run(
                        pending, (1024 - tokens.len()).min(accepted + 1), None,
                    );
                    let before = schedule.clone();
                    let ticket = schedule.begin_round(pending, &proposals).unwrap();
                    let plan = schedule.finish_round(ticket, &predictions).unwrap();
                    assert_unchanged("resolve before publish", &schedule, &before);
                    assert_eq!(plan.emitted(), expected);
                    assert_counter_contract(plan.commit_contract(), start, &inputs, &proposals, &consumed);
                    let outcome = schedule.commit(plan).unwrap();
                    assert_eq!(outcome.emitted, expected);
                    tokens.extend_from_slice(&expected);
                    add_expected_stats(&mut stats, &expected_counter_stats(depth, accepted, &proposals, &expected));
                    assert_eq!(schedule.stats(), &stats);
                    assert_eq!(schedule.positions(), CommittedPositions {
                        base: 64 + tokens.len() - 1, mtp: 64 + tokens.len() - 1,
                    });
                    assert_eq!(schedule.remaining_budget(), 1024 - tokens.len());
                    checked_rounds += 1;
                }
                // Literal scalar counter sequence, independent of proposals,
                // verification tickets, RoundPlans, and commit implementation.
                let scalar = (100u32..1124).collect::<Vec<_>>();
                assert_eq!(tokens, scalar, "depth={depth} accepted={accepted}");
                assert_eq!(schedule.stats().emitted_tokens, 1023);
                assert_eq!(schedule.stats().rounds, 1023usize.div_ceil(accepted + 1));
                assert_eq!(schedule.begin_round(*tokens.last().unwrap(), &vec![0; depth]), Err(MtpScheduleError::Finished));
                cases += 1;
            }
        }
        assert_eq!((cases, checked_rounds), (9, 5543));
        eprintln!("mtp-schedule scalar-oracle complete_output_cases={cases} outputs_per_case=1024 checked_rounds={checked_rounds}");
    }

    #[test]
    fn dropped_failed_and_stale_rounds_preserve_exact_state_at_every_depth() {
        for depth in [1, 2, 3] {
            let mut schedule = Qwen35MtpSchedule::new(64, 64, 1024, 1, vec![]).unwrap();
            let (proposals, _, predictions) = counter_fixture(100, depth, depth);
            let before = schedule.clone();
            let ticket = schedule.begin_round(100, &proposals).unwrap();
            drop(ticket.clone());
            assert_unchanged("dropped draft/verify ticket", &schedule, &before);
            assert_eq!(schedule.finish_round(ticket.clone(), &predictions[..depth]),
                       Err(MtpScheduleError::WrongPredictionCount { expected: depth + 1, got: depth }));
            assert_unchanged("failed target prediction validation", &schedule, &before);
            let plan = schedule.finish_round(ticket.clone(), &predictions).unwrap();
            drop(plan.clone());
            assert_unchanged("dropped prepared plan", &schedule, &before);
            let stale_plan = plan.clone();
            schedule.commit(plan).unwrap();
            let committed = schedule.clone();
            assert_eq!(schedule.finish_round(ticket, &predictions), Err(MtpScheduleError::StaleRound));
            assert_unchanged("stale ticket", &schedule, &committed);
            assert_eq!(schedule.commit(stale_plan), Err(MtpScheduleError::StaleRound));
            assert_unchanged("stale plan", &schedule, &committed);
        }
    }

    #[test]
    fn position_and_epoch_overflow_preserve_exact_host_state_at_every_depth() {
        for depth in [1, 2, 3] {
            let (proposals, _, predictions) = counter_fixture(100, depth, depth);
            let schedule = Qwen35MtpSchedule::new(usize::MAX - 1, usize::MAX - 1, 1024, 1, vec![]).unwrap();
            let before = schedule.clone();
            let ticket = schedule.begin_round(100, &proposals).unwrap();
            assert_eq!(schedule.finish_round(ticket, &predictions), Err(MtpScheduleError::PositionOverflow));
            assert_unchanged("position overflow", &schedule, &before);

            let mut schedule = Qwen35MtpSchedule::new(64, 64, 1024, 1, vec![]).unwrap();
            schedule.epoch = u64::MAX;
            let before = schedule.clone();
            let ticket = schedule.begin_round(100, &proposals).unwrap();
            let plan = schedule.finish_round(ticket, &predictions).unwrap();
            assert_eq!(schedule.commit(plan), Err(MtpScheduleError::PositionOverflow));
            assert_unchanged("epoch overflow", &schedule, &before);
        }
    }

    #[test]
    fn later_literal_matches_do_not_extend_the_accepted_prefix() {
        let mut schedule = Qwen35MtpSchedule::new(64, 64, 1024, 1, vec![901]).unwrap();
        // Literal one-step counter outputs for [100,900,901,902]. Rows1/2
        // match their local input but follow an earlier reject. EOS901 only
        // appears in that discarded suffix, so generation must not finish.
        let ticket = schedule.begin_round(100, &[900, 901, 902]).unwrap();
        let plan = schedule.finish_round(ticket, &[101, 901, 902, 903]).unwrap();
        assert_eq!(plan.emitted(), &[101]);
        assert_eq!(plan.accepted_proposals(), 0);
        assert!(!plan.finishes_generation());
        assert_eq!(schedule.commit(plan).unwrap().emitted, [101]);
        assert!(!schedule.is_finished());
        assert_eq!(schedule.stats().proposed_by_depth, [1, 1, 1]);
        assert_eq!(schedule.stats().matched_by_depth, [0, 1, 1]);
        assert_eq!(schedule.stats().accepted_prefix_by_depth, [0, 0, 0]);
        assert_eq!(schedule.positions(), CommittedPositions { base: 65, mtp: 65 });
    }
}
