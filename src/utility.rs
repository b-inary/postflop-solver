use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use rayon::prelude::*;

#[cfg(feature = "custom_alloc")]
use crate::alloc::*;
#[cfg(feature = "custom_alloc")]
use std::vec::from_elem_in;

/// Executes `op` for each child potentially in parallel.
#[inline]
pub fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    if node.enable_parallelization() {
        node.actions().into_par_iter().for_each(op);
    } else {
        node.actions().into_iter().for_each(op);
    }
}

/// Normalizes the cumulative strategy.
#[inline]
pub fn normalize_strategy<T: Game>(game: &T) {
    normalize_strategy_recursive::<T::Node>(&mut game.root());
}

/// Computes the expected value of `player`'s payoff.
#[inline]
pub fn compute_ev<T: Game>(game: &T, player: usize) -> f32 {
    let reach = [game.initial_reach(0), game.initial_reach(1)];
    compute_ev_recursive(game, &game.root(), player, reach[player], reach[player ^ 1])
}

/// Computes the exploitability of the strategy.
#[inline]
pub fn compute_exploitability<T: Game>(game: &T, is_normalized: bool) -> f32 {
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
    (cfv_sum0 + cfv_sum1) as f32 / 2.0
}

/// The recursive helper function for normalizing the strategy.
fn normalize_strategy_recursive<T: GameNode>(node: &mut T) {
    if node.is_terminal() {
        return;
    }

    if !node.is_chance() {
        let num_actions = node.num_actions();
        let strategy = node.strategy_mut();
        let row_size = strategy.len() / num_actions;

        #[cfg(feature = "custom_alloc")]
        let mut denom = from_elem_in(0.0, row_size, StackAlloc);
        #[cfg(not(feature = "custom_alloc"))]
        let mut denom = vec![0.0; row_size];
        strategy.chunks(row_size).for_each(|row| {
            add_slice(&mut denom, row);
        });

        let default = 1.0 / num_actions as f32;
        strategy.chunks_mut(row_size).for_each(|row| {
            div_slice(row, &denom, default);
        });
    }

    for_each_child(node, |action| {
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
        #[cfg(feature = "custom_alloc")]
        let mut cfv = from_elem_in(0.0, game.num_private_hands(player), StackAlloc);
        #[cfg(not(feature = "custom_alloc"))]
        let mut cfv = vec![0.0; game.num_private_hands(player)];
        game.evaluate(&mut cfv, node, player, cfreach);
        return cfv
            .iter()
            .zip(reach)
            .fold(0.0, |acc, (v, r)| acc + *v as f64 * *r as f64) as f32;
    }

    #[cfg(feature = "custom_alloc")]
    let ev = MutexLike::new(from_elem_in(0.0, node.num_actions(), StackAlloc));
    #[cfg(not(feature = "custom_alloc"))]
    let ev = MutexLike::new(vec![0.0; node.num_actions()]);

    // chance node
    if node.is_chance() {
        #[cfg(feature = "custom_alloc")]
        let mut cfreach = cfreach.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom_alloc"))]
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());

        #[cfg(feature = "custom_alloc")]
        let mut weights = from_elem_in(1.0, node.num_actions(), StackAlloc);
        #[cfg(not(feature = "custom_alloc"))]
        let mut weights = vec![1.0; node.num_actions()];
        for iso_chance in node.isomorphic_chances() {
            weights[iso_chance.index] += 1.0;
        }

        for_each_child(node, |action| {
            ev.lock()[action] = weights[action]
                * compute_ev_recursive(game, &node.play(action), player, reach, &cfreach);
        });
    }
    // player node
    else if node.player() == player {
        let strategy = node.strategy();
        let row_size = strategy.len() / node.num_actions();
        #[cfg(feature = "custom_alloc")]
        let mut reach_actions = strategy.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom_alloc"))]
        let mut reach_actions = strategy.to_vec();
        reach_actions.chunks_mut(row_size).for_each(|row| {
            mul_slice(row, reach);
        });

        for_each_child(node, |action| {
            ev.lock()[action] = compute_ev_recursive(
                game,
                &node.play(action),
                player,
                row(&reach_actions, action, row_size),
                cfreach,
            );
        });
    }
    // opponent node
    else {
        let strategy = node.strategy();
        let row_size = strategy.len() / node.num_actions();

        #[cfg(feature = "custom_alloc")]
        let mut cfreach_actions = strategy.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom_alloc"))]
        let mut cfreach_actions = strategy.to_vec();
        cfreach_actions.chunks_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        for_each_child(node, |action| {
            ev.lock()[action] = compute_ev_recursive(
                game,
                &node.play(action),
                player,
                reach,
                row(&cfreach_actions, action, row_size),
            );
        });
    }

    ev.lock().iter().sum::<f32>()
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
    #[cfg(feature = "custom_alloc")]
    let cfv_actions = MutexLike::new(from_elem_in(
        0.0,
        num_actions * num_private_hands,
        StackAlloc,
    ));
    #[cfg(not(feature = "custom_alloc"))]
    let cfv_actions = MutexLike::new(vec![0.0; num_actions * num_private_hands]);

    // chance node
    if node.is_chance() {
        // updates the reach probabilities
        #[cfg(feature = "custom_alloc")]
        let mut cfreach = cfreach.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom_alloc"))]
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());

        // computes the counterfactual values of each action
        for_each_child(node, |action| {
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
        for_each_child(node, |action| {
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
        #[cfg(feature = "custom_alloc")]
        let mut cfreach_actions = strategy.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom_alloc"))]
        let mut cfreach_actions = strategy.to_vec();

        // if the strategy is not normalized, we need to normalize it
        if !is_normalized {
            #[cfg(feature = "custom_alloc")]
            let mut denom = from_elem_in(0.0, row_size, StackAlloc);
            #[cfg(not(feature = "custom_alloc"))]
            let mut denom = vec![0.0; row_size];
            cfreach_actions.chunks(row_size).for_each(|row| {
                add_slice(&mut denom, row);
            });

            let default = 1.0 / node.num_actions() as f32;
            cfreach_actions.chunks_mut(row_size).for_each(|row| {
                div_slice(row, &denom, default);
            });
        }

        cfreach_actions.chunks_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // computes the counterfactual values of each action
        for_each_child(node, |action| {
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
