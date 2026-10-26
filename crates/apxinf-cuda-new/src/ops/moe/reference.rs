use std::cmp::Ordering;

use apxinf_core::Result;

use super::contracts::{invalid, RoutePlan, RoutedMoeConfig, SharedExpertOutput};

/// CPU reference for softmax -> deterministic top-k routing.
///
/// Ties are resolved by the lower expert ID. When `renormalize` is true, the
/// selected probabilities are divided by their selected sum.
pub fn route_topk_reference(
    config: RoutedMoeConfig,
    logits: &[f32],
    tokens: usize,
) -> Result<RoutePlan> {
    config.validate()?;
    if tokens == 0 {
        return Err(invalid("routed MoE requires at least one token"));
    }
    let expected = tokens
        .checked_mul(config.experts)
        .ok_or_else(|| invalid("routed MoE logit count overflow"))?;
    if logits.len() != expected {
        return Err(invalid(
            "routed MoE logits must have shape [tokens, experts]",
        ));
    }

    let routes = tokens
        .checked_mul(config.top_k)
        .ok_or_else(|| invalid("routed MoE route count overflow"))?;
    let mut expert_ids = Vec::with_capacity(routes);
    let mut expert_weights = Vec::with_capacity(routes);
    let mut ranking = Vec::with_capacity(config.experts);

    for row in logits.chunks_exact(config.experts) {
        if row.iter().any(|value| !value.is_finite()) {
            return Err(invalid("routed MoE logits must be finite"));
        }
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f64 = row
            .iter()
            .map(|logit| f64::from(*logit - maximum).exp())
            .sum();

        ranking.clear();
        ranking.extend(row.iter().copied().enumerate());
        ranking.sort_by(|(left_id, left), (right_id, right)| {
            right
                .partial_cmp(left)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left_id.cmp(right_id))
        });

        let selected = &ranking[..config.top_k];
        let mut weights = selected
            .iter()
            .map(|(_, logit)| (f64::from(*logit - maximum).exp() / denominator) as f32)
            .collect::<Vec<_>>();
        if config.renormalize {
            let selected_sum: f32 = weights.iter().sum();
            for weight in &mut weights {
                *weight /= selected_sum;
            }
        }
        for ((expert, _), weight) in selected.iter().zip(weights) {
            expert_ids.push(*expert as u32);
            expert_weights.push(weight);
        }
    }

    RoutePlan::from_token_major(config, tokens, expert_ids, expert_weights)
}

/// CPU reference for assignment-major expert output combination.
///
/// `assignment_output` is `[plan.assignments().len(), hidden]` in the exact
/// expert-major order returned by the plan. The optional shared expert is
/// added as `sigmoid(gate_logit[token]) * shared_output[token]`.
pub fn combine_reference(
    plan: &RoutePlan,
    assignment_output: &[f32],
    hidden: usize,
    shared: Option<SharedExpertOutput<'_>>,
) -> Result<Vec<f32>> {
    if hidden == 0 {
        return Err(invalid(
            "routed MoE combine requires a non-zero hidden width",
        ));
    }
    let assignment_elements = plan
        .assignments()
        .len()
        .checked_mul(hidden)
        .ok_or_else(|| invalid("routed MoE assignment output size overflow"))?;
    if assignment_output.len() != assignment_elements {
        return Err(invalid(
            "routed MoE assignment output must have shape [assignments, hidden]",
        ));
    }
    if assignment_output.iter().any(|value| !value.is_finite()) {
        return Err(invalid("routed MoE assignment output must be finite"));
    }

    let output_elements = plan
        .tokens()
        .checked_mul(hidden)
        .ok_or_else(|| invalid("routed MoE combined output size overflow"))?;
    let mut combined = vec![0.0f32; output_elements];
    for (assignment_index, assignment) in plan.assignments().iter().enumerate() {
        let source = &assignment_output[assignment_index * hidden..(assignment_index + 1) * hidden];
        let token = assignment.token_id as usize;
        let destination = &mut combined[token * hidden..(token + 1) * hidden];
        for (out, value) in destination.iter_mut().zip(source) {
            *out += assignment.weight * value;
        }
    }

    if let Some(shared) = shared {
        if shared.output.len() != output_elements || shared.gate_logits.len() != plan.tokens() {
            return Err(invalid(
                "shared expert output and gates must have shapes [tokens, hidden] and [tokens]",
            ));
        }
        if shared
            .output
            .iter()
            .chain(shared.gate_logits)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("shared expert output and gates must be finite"));
        }
        for token in 0..plan.tokens() {
            let gate = sigmoid(shared.gate_logits[token]);
            let start = token * hidden;
            for dimension in 0..hidden {
                combined[start + dimension] += gate * shared.output[start + dimension];
            }
        }
    }

    Ok(combined)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f32, right: f32) {
        assert!((left - right).abs() <= 1e-6, "{left} != {right}");
    }

    #[test]
    fn qwen35_geometry_selects_top8_of_256_with_stable_ties() {
        let config = RoutedMoeConfig::new(256, 8);
        let mut logits = vec![-100.0; 256];
        for (expert, score) in [
            (91, 4.0),
            (7, 7.0),
            (2, 7.0),
            (88, 3.0),
            (12, 6.0),
            (255, 2.0),
            (0, 5.0),
            (31, 1.0),
        ] {
            logits[expert] = score;
        }
        let plan = route_topk_reference(config, &logits, 1).unwrap();
        assert_eq!(plan.expert_ids(), &[2, 7, 12, 0, 91, 88, 255, 31]);
        assert_close(plan.expert_weights().iter().sum(), 1.0);
        assert_eq!(plan.assignments().len(), 8);
        assert_eq!(plan.expert_offsets().len(), 257);
        assert_eq!(plan.assignments_for_expert(2).unwrap()[0].route_slot, 0);
        assert_eq!(plan.assignments_for_expert(7).unwrap()[0].route_slot, 1);
    }

    #[test]
    fn renormalization_is_an_explicit_contract() {
        let logits = [0.0; 4];
        let normalized = route_topk_reference(RoutedMoeConfig::new(4, 2), &logits, 1).unwrap();
        assert_eq!(normalized.expert_ids(), &[0, 1]);
        assert_eq!(normalized.expert_weights(), &[0.5, 0.5]);

        let unnormalized = route_topk_reference(
            RoutedMoeConfig {
                experts: 4,
                top_k: 2,
                renormalize: false,
            },
            &logits,
            1,
        )
        .unwrap();
        assert_eq!(unnormalized.expert_ids(), &[0, 1]);
        assert_eq!(unnormalized.expert_weights(), &[0.25, 0.25]);
    }

    #[test]
    fn assignments_are_expert_major_but_preserve_token_and_slot_ids() {
        let config = RoutedMoeConfig::new(4, 2);
        let logits = [4.0, 3.0, 0.0, 0.0, 0.0, 4.0, 3.0, 0.0];
        let plan = route_topk_reference(config, &logits, 2).unwrap();
        assert_eq!(plan.expert_ids(), &[0, 1, 1, 2]);
        let identities = plan
            .assignments()
            .iter()
            .map(|assignment| {
                (
                    assignment.expert_id,
                    assignment.token_id,
                    assignment.route_slot,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(identities, [(0, 0, 0), (1, 0, 1), (1, 1, 0), (2, 1, 1)]);
        assert_eq!(plan.expert_offsets(), &[0, 1, 3, 4, 4]);
    }

    #[test]
    fn combine_adds_weighted_routes_and_sigmoid_gated_shared_expert() {
        let plan = route_topk_reference(
            RoutedMoeConfig::new(3, 2),
            &[2.0, 1.0, 0.0, 0.0, 2.0, 1.0],
            2,
        )
        .unwrap();
        // Assignment order: (expert 0, token 0), (expert 1, token 0),
        // (expert 1, token 1), (expert 2, token 1).
        let assignment_output = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let shared_output = [10.0, 100.0, 20.0, 200.0];
        let output = combine_reference(
            &plan,
            &assignment_output,
            2,
            Some(SharedExpertOutput {
                output: &shared_output,
                gate_logits: &[0.0, 3.0f32.ln()],
            }),
        )
        .unwrap();

        let high = 1.0 / (1.0 + (-1.0f32).exp());
        let low = 1.0 - high;
        let expected = [
            high * 1.0 + low * 2.0 + 0.5 * 10.0,
            high * 10.0 + low * 20.0 + 0.5 * 100.0,
            high * 3.0 + low * 4.0 + 0.75 * 20.0,
            high * 30.0 + low * 40.0 + 0.75 * 200.0,
        ];
        for (actual, expected) in output.into_iter().zip(expected) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn invalid_ids_weights_shapes_and_nonfinite_values_are_rejected() {
        let config = RoutedMoeConfig::new(3, 2);
        assert!(route_topk_reference(config, &[0.0; 5], 2).is_err());
        assert!(route_topk_reference(config, &[0.0, f32::NAN, 0.0], 1).is_err());
        assert!(RoutePlan::from_token_major(config, 1, vec![0, 0], vec![0.5, 0.5]).is_err());
        assert!(RoutePlan::from_token_major(config, 1, vec![0, 3], vec![0.5, 0.5]).is_err());
        let plan = route_topk_reference(config, &[2.0, 1.0, 0.0], 1).unwrap();
        assert!(combine_reference(&plan, &[0.0; 3], 2, None).is_err());
    }
}
