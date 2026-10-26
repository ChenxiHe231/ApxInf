use apxinf_core::{Error, Result};

/// Model-neutral stages shared by the CPU reference and fixed CUDA candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoutedMoeSemantic {
    RouteTopK,
    AssignExperts,
    Combine,
}

/// Routing geometry shared by reference and future device implementations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RoutedMoeConfig {
    pub experts: usize,
    pub top_k: usize,
    /// Renormalize the selected probabilities to sum to one per token.
    pub renormalize: bool,
}

impl RoutedMoeConfig {
    pub const fn new(experts: usize, top_k: usize) -> Self {
        Self {
            experts,
            top_k,
            renormalize: true,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.experts == 0 {
            return Err(invalid("routed MoE requires at least one expert"));
        }
        if self.top_k == 0 || self.top_k > self.experts {
            return Err(invalid("routed MoE top_k must be in the range 1..=experts"));
        }
        if u32::try_from(self.experts).is_err() {
            return Err(invalid("routed MoE expert count exceeds the ID range"));
        }
        if u16::try_from(self.top_k).is_err() {
            return Err(invalid("routed MoE top_k exceeds the route-slot range"));
        }
        Ok(())
    }
}

/// One row in the deterministic expert-major dispatch list.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertAssignment {
    pub expert_id: u32,
    pub token_id: u32,
    pub route_slot: u16,
    pub weight: f32,
}

/// Validated output of top-k routing.
///
/// `expert_ids` and `expert_weights` are token-major `[tokens, top_k]`.
/// `assignments` is sorted by `(expert_id, token_id, route_slot)`, and
/// `expert_offsets` has `experts + 1` entries into that assignment list.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutePlan {
    config: RoutedMoeConfig,
    tokens: usize,
    expert_ids: Vec<u32>,
    expert_weights: Vec<f32>,
    assignments: Vec<ExpertAssignment>,
    expert_offsets: Vec<usize>,
}

impl RoutePlan {
    pub fn from_token_major(
        config: RoutedMoeConfig,
        tokens: usize,
        expert_ids: Vec<u32>,
        expert_weights: Vec<f32>,
    ) -> Result<Self> {
        config.validate()?;
        if tokens == 0 {
            return Err(invalid("routed MoE requires at least one token"));
        }
        if u32::try_from(tokens).is_err() {
            return Err(invalid("routed MoE token count exceeds the ID range"));
        }
        let routes = tokens
            .checked_mul(config.top_k)
            .ok_or_else(|| invalid("routed MoE route count overflow"))?;
        if expert_ids.len() != routes || expert_weights.len() != routes {
            return Err(invalid(
                "routed MoE IDs and weights must have shape [tokens, top_k]",
            ));
        }

        let mut assignments = Vec::with_capacity(routes);
        for token in 0..tokens {
            let start = token * config.top_k;
            let end = start + config.top_k;
            let ids = &expert_ids[start..end];
            let weights = &expert_weights[start..end];
            for (slot, (&expert_id, &weight)) in ids.iter().zip(weights).enumerate() {
                if expert_id as usize >= config.experts {
                    return Err(invalid("routed MoE expert ID is out of range"));
                }
                if ids[..slot].contains(&expert_id) {
                    return Err(invalid(
                        "routed MoE cannot select the same expert twice for one token",
                    ));
                }
                if !weight.is_finite() || weight < 0.0 {
                    return Err(invalid(
                        "routed MoE weights must be finite and non-negative",
                    ));
                }
                assignments.push(ExpertAssignment {
                    expert_id,
                    token_id: token as u32,
                    route_slot: slot as u16,
                    weight,
                });
            }
            let sum: f32 = weights.iter().sum();
            if config.renormalize {
                if (sum - 1.0).abs() > 1e-4 {
                    return Err(invalid(
                        "renormalized routed MoE weights must sum to one per token",
                    ));
                }
            } else if sum > 1.0 + 1e-4 {
                return Err(invalid(
                    "non-renormalized routed MoE weights cannot sum above one",
                ));
            }
        }

        assignments.sort_by_key(|assignment| {
            (
                assignment.expert_id,
                assignment.token_id,
                assignment.route_slot,
            )
        });
        let mut expert_offsets = vec![0usize; config.experts + 1];
        for assignment in &assignments {
            expert_offsets[assignment.expert_id as usize + 1] += 1;
        }
        for expert in 0..config.experts {
            expert_offsets[expert + 1] += expert_offsets[expert];
        }

        Ok(Self {
            config,
            tokens,
            expert_ids,
            expert_weights,
            assignments,
            expert_offsets,
        })
    }

    pub const fn config(&self) -> RoutedMoeConfig {
        self.config
    }

    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn expert_ids(&self) -> &[u32] {
        &self.expert_ids
    }

    pub fn expert_weights(&self) -> &[f32] {
        &self.expert_weights
    }

    pub fn assignments(&self) -> &[ExpertAssignment] {
        &self.assignments
    }

    pub fn expert_offsets(&self) -> &[usize] {
        &self.expert_offsets
    }

    pub fn assignments_for_expert(&self, expert: usize) -> Result<&[ExpertAssignment]> {
        if expert >= self.config.experts {
            return Err(invalid("routed MoE expert ID is out of range"));
        }
        Ok(&self.assignments[self.expert_offsets[expert]..self.expert_offsets[expert + 1]])
    }
}

/// Shared-expert contribution consumed by [`super::combine_reference`].
pub struct SharedExpertOutput<'a> {
    /// Token-major `[tokens, hidden]` expert output.
    pub output: &'a [f32],
    /// One pre-sigmoid gate logit per token.
    pub gate_logits: &'a [f32],
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}
