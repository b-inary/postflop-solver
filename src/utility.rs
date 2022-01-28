use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use rayon::prelude::*;

/// Normalizes the cumulative strategy.
#[inline]
pub fn normalize_strategy<T: Game>(game: &T) {
    normalize_strategy_recursive::<T>(&mut game.root());
}

/// Computes the expected value of `player`'s payoff.
#[inline]
pub fn compute_ev<T: Game>(game: &T, player: usize) -> f32 {
    let reach = [game.initial_reach(0), game.initial_reach(1)];
    compute_ev_recursive(game, &game.root(), player, reach[player], reach[player ^ 1])
}

/// Computes the exploitability of the strategy.
#[inline]
pub fn compute_exploitability<T: Game>(game: &T, bias: f32, is_normalized: bool) -> f32 {
    let mut cfv = [
        vec![0.0; game.num_private_hands(0)],
        vec![0.0; game.num_private_hands(1)],
    ];
    let reach = [game.initial_reach(0), game.initial_reach(1)];
    for player in 0..2 {
        compute_best_cfv_recursive(
            &mut cfv[player],
            game,
            &game.root(),
            player,
            reach[player ^ 1],
            is_normalized,
        );
    }
    let cfv_sum0 = cfv[0].iter().fold(0.0, |acc, v| acc + *v as f64);
    let cfv_sum1 = cfv[1].iter().fold(0.0, |acc, v| acc + *v as f64);
    (cfv_sum0 + cfv_sum1) as f32 / 2.0 - bias
}

/// The recursive helper function for normalizing the strategy.
fn normalize_strategy_recursive<T: Game>(node: &mut T::Node) {
    if node.is_terminal() {
        return;
    }

    if !node.is_chance() {
        let num_actions = node.num_actions();
        let strategy = node.strategy_mut();
        let row_size = strategy.len() / num_actions;

        let mut denom = vec![0.0; row_size];
        strategy.chunks(row_size).for_each(|row| {
            add_slice(&mut denom, row);
        });

        let default = 1.0 / num_actions as f32;
        strategy.chunks_mut(row_size).for_each(|row| {
            div_slice(row, &denom, default);
        });
    }

    node.actions().into_par_iter().for_each(|action| {
        normalize_strategy_recursive::<T>(&mut node.play(action));
    })
}

/// The recursive helper function for computing the expected value.
fn compute_ev_recursive<T: Game>(
    game: &T,
    node: &T::Node,
    player: usize,
    reach: &[f32],
    cfreach: &[f32],
) -> f32 {
    // terminal node
    if node.is_terminal() {
        let mut cfv = vec![0.0; game.num_private_hands(player)];
        game.evaluate(&mut cfv, node, player, cfreach);
        return cfv
            .iter()
            .zip(reach)
            .fold(0.0, |acc, (v, r)| acc + *v as f64 * *r as f64) as f32;
    }
    // chance node
    else if node.is_chance() {
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());
        let mut weights = vec![1.0; node.num_actions()];
        for iso_chance in node.isomorphic_chances() {
            weights[iso_chance.index] += 1.0;
        }
        node.actions()
            .into_par_iter()
            .map(|action| {
                compute_ev_recursive(game, &node.play(action), player, reach, &cfreach)
                    * weights[action]
            })
            .sum()
    }
    // player node
    else if node.player() == player {
        let strategy = node.strategy();
        let row_size = strategy.len() / node.num_actions();
        let mut reach_actions = strategy.to_vec();
        reach_actions.chunks_mut(row_size).for_each(|mut row| {
            mul_slice(&mut row, reach);
        });
        node.actions()
            .into_par_iter()
            .map(|action| {
                compute_ev_recursive(
                    game,
                    &node.play(action),
                    player,
                    row(&reach_actions, action, row_size),
                    cfreach,
                )
            })
            .sum()
    }
    // opponent node
    else {
        let strategy = node.strategy();
        let row_size = strategy.len() / node.num_actions();
        let mut cfreach_actions = strategy.to_vec();
        cfreach_actions.chunks_mut(row_size).for_each(|mut row| {
            mul_slice(&mut row, cfreach);
        });
        node.actions()
            .into_par_iter()
            .map(|action| {
                compute_ev_recursive(
                    game,
                    &node.play(action),
                    player,
                    reach,
                    row(&cfreach_actions, action, row_size),
                )
            })
            .sum()
    }
}

/// The recursive helper function for computing the counterfactual values of best response.
fn compute_best_cfv_recursive<T: Game>(
    result: &mut [f32],
    game: &T,
    node: &T::Node,
    player: usize,
    cfreach: &[f32],
    is_normalized: bool,
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    // allocates memory for storing the counterfactual values
    let num_actions = node.num_actions();
    let num_private_hands = game.num_private_hands(player);
    let cfv_actions = MutexLike::new(vec![0.0; num_actions * num_private_hands]);

    // chance node
    if node.is_chance() {
        // updates the reach probabilities
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());

        // computes the counterfactual values of each action
        node.actions().into_par_iter().for_each(|action| {
            compute_best_cfv_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &node.play(action),
                player,
                &cfreach,
                is_normalized,
            )
        });

        // sums up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        cfv_actions.chunks(num_private_hands).for_each(|row| {
            add_slice(result, row);
        });

        // get information about isomorphic chances
        let iso_chances = node.isomorphic_chances();

        // processes isomorphic chances
        for iso_chance in iso_chances {
            let tmp = row_mut(&mut cfv_actions, iso_chance.index, num_private_hands);
            for &(i, j) in &iso_chance.swap_list[player] {
                tmp.swap(i, j);
            }
            add_slice(result, tmp);
            for &(i, j) in &iso_chance.swap_list[player] {
                tmp.swap(i, j);
            }
        }
    }
    // player node
    else if node.player() == player {
        // computes the counterfactual values of each action
        node.actions().into_par_iter().for_each(|action| {
            compute_best_cfv_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &node.play(action),
                player,
                cfreach,
                is_normalized,
            )
        });

        // computes element-wise maximum (takes the best response)
        result.fill(f32::NEG_INFINITY);
        let cfv_actions = cfv_actions.lock();
        cfv_actions.chunks(num_private_hands).for_each(|row| {
            result.iter_mut().zip(row).for_each(|(l, r)| {
                *l = l.max(*r);
            });
        });
    }
    // opponent node
    else {
        // updates the reach probabilities
        let strategy = node.strategy();
        let row_size = strategy.len() / node.num_actions();
        let mut cfreach_actions = strategy.to_vec();

        // if the strategy is not normalized, we need to normalize it
        if !is_normalized {
            let mut denom = vec![0.0; row_size];
            cfreach_actions.chunks(row_size).for_each(|row| {
                add_slice(&mut denom, row);
            });

            let default = 1.0 / node.num_actions() as f32;
            cfreach_actions.chunks_mut(row_size).for_each(|row| {
                div_slice(row, &denom, default);
            });
        }

        cfreach_actions.chunks_mut(row_size).for_each(|mut row| {
            mul_slice(&mut row, cfreach);
        });

        // computes the counterfactual values of each action
        node.actions().into_par_iter().for_each(|action| {
            let cfreach = row(&cfreach_actions, action, row_size);
            if cfreach.iter().any(|&x| x > 0.0) {
                compute_best_cfv_recursive(
                    row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                    game,
                    &node.play(action),
                    player,
                    cfreach,
                    is_normalized,
                );
            }
        });

        // sums up the counterfactual values
        let cfv_actions = cfv_actions.lock();
        cfv_actions.chunks(num_private_hands).for_each(|row| {
            add_slice(result, row);
        });
    }
}
