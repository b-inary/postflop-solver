use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;
#[cfg(feature = "custom-alloc")]
use std::vec;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Executes `op` for each child potentially in parallel.
#[cfg(feature = "rayon")]
#[inline]
pub(crate) fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    if node.enable_parallelization() {
        node.actions().into_par_iter().for_each(op);
    } else {
        node.actions().into_iter().for_each(op);
    }
}

/// Executes `op` for each child.
#[cfg(not(feature = "rayon"))]
#[inline]
pub(crate) fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    node.actions().into_iter().for_each(op);
}

/// Encodes the `f32` slice to the `i16` slice, and returns the scale.
#[inline]
pub(crate) fn encode_signed_slice(dst: &mut [i16], slice: &[f32]) -> f32 {
    let scale = slice.iter().fold(0.0f32, |m, v| v.abs().max(m));
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = i16::MAX as f32 / scale_nonzero;
    dst.iter_mut()
        .zip(slice)
        .for_each(|(d, s)| *d = (s * encoder).round() as i16);
    scale
}

/// Encodes the `f32` slice to the `u16` slice, and returns the scale.
#[inline]
pub(crate) fn encode_unsigned_slice(dst: &mut [u16], slice: &[f32]) -> f32 {
    let scale = slice.iter().fold(0.0f32, |m, v| v.max(m));
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u16::MAX as f32 / scale_nonzero;
    dst.iter_mut()
        .zip(slice)
        .for_each(|(d, s)| *d = (s * encoder).round() as u16);
    scale
}

/// Finalizes the solving process.
#[inline]
pub fn finalize<T: Game>(game: &mut T) {
    if !game.is_ready() {
        panic!("the game is not ready");
    }

    if game.is_solved() {
        panic!("the game is already solved");
    }

    let reach = [game.initial_weight(0), game.initial_weight(1)];

    // computes the expected values and saves them
    for player in 0..2 {
        let mut cfv = vec![0.0; game.num_private_hands(player)];
        compute_ev_recursive(
            &mut cfv,
            game,
            &mut game.root(),
            player,
            reach[player],
            reach[player ^ 1],
        );
    }

    // sets the game is solved
    game.set_solved();
}

/// Computes the exploitability of the current strategy.
#[inline]
pub fn compute_exploitability<T: Game>(game: &T) -> f32 {
    if !game.is_ready() {
        panic!("the game is not ready");
    }

    let mut cfv = [
        vec![0.0; game.num_private_hands(0)],
        vec![0.0; game.num_private_hands(1)],
    ];

    let reach = [game.initial_weight(0), game.initial_weight(1)];

    for player in 0..2 {
        compute_best_cfv_recursive(
            &mut cfv[player],
            game,
            &game.root(),
            player,
            reach[player ^ 1],
        );
    }

    let get_sum = |player: usize| {
        cfv[player]
            .iter()
            .zip(reach[player])
            .fold(0.0, |sum, (&cfv, &reach)| sum + cfv as f64 * reach as f64)
    };

    (get_sum(0) + get_sum(1)) as f32 / 2.0
}

/// The recursive helper function for computing the counterfactual values of the given strategy.
fn compute_ev_recursive<T: Game>(
    result: &mut [f32],
    game: &T,
    node: &mut T::Node,
    player: usize,
    reach: &[f32],
    cfreach: &[f32],
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_private_hands = game.num_private_hands(player);

    // allocates memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(vec::from_elem_in(
        0.0,
        num_actions * num_private_hands,
        StackAlloc,
    ));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(vec![0.0; num_actions * num_private_hands]);

    // chance node
    if node.is_chance() {
        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = vec::from_elem_in(0.0, num_private_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = vec![0.0; num_private_hands];

        // updates the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach = cfreach.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());

        // computes the counterfactual values of each action
        for_each_child(node, |action| {
            compute_ev_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &mut node.play(action),
                player,
                reach,
                &cfreach,
            );
        });

        // sums up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        cfv_actions.chunks(num_private_hands).for_each(|row| {
            result_f64.iter_mut().zip(row).for_each(|(r, &v)| {
                *r += v as f64;
            });
        });

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

        // processes isomorphic chances
        for i in 0..isomorphic_chances.len() {
            let tmp = row_mut(&mut cfv_actions, isomorphic_chances[i], num_private_hands);
            for &(i, j) in &game.isomorphic_swap(node, i)[player] {
                tmp.swap(i, j);
            }
            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });
            for &(i, j) in &game.isomorphic_swap(node, i)[player] {
                tmp.swap(i, j);
            }
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            *r = v as f32;
        });
    }
    // player node
    else if node.player() == player {
        // obtains the strategy
        let mut strategy = if game.is_compression_enabled() {
            let strategy = node.strategy_compressed();
            #[cfg(feature = "custom-alloc")]
            {
                let mut vec = Vec::with_capacity_in(strategy.len(), StackAlloc);
                vec.extend(strategy.iter().map(|&x| x as f32));
                vec
            }
            #[cfg(not(feature = "custom-alloc"))]
            {
                strategy.iter().map(|&x| x as f32).collect()
            }
        } else {
            #[cfg(feature = "custom-alloc")]
            {
                node.strategy().to_vec_in(StackAlloc)
            }
            #[cfg(not(feature = "custom-alloc"))]
            {
                node.strategy().to_vec()
            }
        };

        // normalizes the strategy
        normalize_strategy(&mut strategy, node.num_actions());

        // updates the reach probabilities
        let mut reach_actions = strategy.clone();
        reach_actions.chunks_mut(num_private_hands).for_each(|row| {
            mul_slice(row, reach);
        });

        // computes the counterfactual values of each action
        for_each_child(node, |action| {
            compute_ev_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &mut node.play(action),
                player,
                row(&reach_actions, action, num_private_hands),
                cfreach,
            );
        });

        // sums up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        mul_slice(&mut strategy, &cfv_actions);
        strategy.chunks(num_private_hands).for_each(|row| {
            add_slice(result, row);
        });

        // save the expected values
        cfv_actions
            .chunks_mut(num_private_hands)
            .for_each(|row| mul_slice(row, reach));
        if game.is_compression_enabled() {
            let ev_scale = encode_signed_slice(node.expected_values_compressed_mut(), &cfv_actions);
            node.set_expected_value_scale(ev_scale);
        } else {
            node.expected_values_mut().copy_from_slice(&cfv_actions);
        }
    }
    // opponent node
    else if num_actions == 1 {
        // simply recurses when the number of actions is 1
        compute_ev_recursive(result, game, &mut node.play(0), player, reach, cfreach);
    } else {
        // obtains the strategy
        let mut cfreach_actions = if game.is_compression_enabled() {
            let strategy = node.strategy_compressed();
            #[cfg(feature = "custom-alloc")]
            {
                let mut vec = Vec::with_capacity_in(strategy.len(), StackAlloc);
                vec.extend(strategy.iter().map(|&x| x as f32));
                vec
            }
            #[cfg(not(feature = "custom-alloc"))]
            {
                strategy.iter().map(|&x| x as f32).collect()
            }
        } else {
            #[cfg(feature = "custom-alloc")]
            {
                node.strategy().to_vec_in(StackAlloc)
            }
            #[cfg(not(feature = "custom-alloc"))]
            {
                node.strategy().to_vec()
            }
        };

        // normalizes the strategy
        normalize_strategy(&mut cfreach_actions, node.num_actions());

        // updates the reach probabilities
        let row_size = cfreach_actions.len() / node.num_actions();
        cfreach_actions.chunks_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // computes the counterfactual values of each action
        for_each_child(node, |action| {
            compute_ev_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &mut node.play(action),
                player,
                reach,
                row(&cfreach_actions, action, row_size),
            );
        });

        // sums up the counterfactual values
        let cfv_actions = cfv_actions.lock();
        cfv_actions.chunks(num_private_hands).for_each(|row| {
            add_slice(result, row);
        });
    }
}

/// The recursive helper function for computing the counterfactual values of best response.
fn compute_best_cfv_recursive<T: Game>(
    result: &mut [f32],
    game: &T,
    node: &T::Node,
    player: usize,
    cfreach: &[f32],
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_private_hands = game.num_private_hands(player);

    // simply recurses when the number of actions is 1
    if num_actions == 1 && !node.is_chance() {
        let child = &node.play(0);
        compute_best_cfv_recursive(result, game, child, player, cfreach);
        return;
    }

    // allocates memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(vec::from_elem_in(
        0.0,
        num_actions * num_private_hands,
        StackAlloc,
    ));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(vec![0.0; num_actions * num_private_hands]);

    // chance node
    if node.is_chance() {
        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = vec::from_elem_in(0.0, num_private_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = vec![0.0; num_private_hands];

        // updates the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach = cfreach.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
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
            )
        });

        // sums up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        cfv_actions.chunks(num_private_hands).for_each(|row| {
            result_f64.iter_mut().zip(row).for_each(|(r, v)| {
                *r += *v as f64;
            });
        });

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

        // processes isomorphic chances
        for i in 0..isomorphic_chances.len() {
            let tmp = row_mut(&mut cfv_actions, isomorphic_chances[i], num_private_hands);
            for &(i, j) in &game.isomorphic_swap(node, i)[player] {
                tmp.swap(i, j);
            }
            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });
            for &(i, j) in &game.isomorphic_swap(node, i)[player] {
                tmp.swap(i, j);
            }
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            *r = v as f32;
        });
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
            )
        });

        // computes element-wise maximum (takes the best response)
        result.fill(f32::MIN);
        let cfv_actions = cfv_actions.lock();
        cfv_actions.chunks(num_private_hands).for_each(|row| {
            result.iter_mut().zip(row).for_each(|(l, r)| {
                *l = l.max(*r);
            });
        });
    }
    // opponent node
    else {
        // obtains the strategy
        let mut cfreach_actions = if game.is_compression_enabled() {
            let strategy = node.strategy_compressed();
            #[cfg(feature = "custom-alloc")]
            {
                let mut vec = Vec::with_capacity_in(strategy.len(), StackAlloc);
                vec.extend(strategy.iter().map(|&x| x as f32));
                vec
            }
            #[cfg(not(feature = "custom-alloc"))]
            {
                strategy.iter().map(|&x| x as f32).collect()
            }
        } else {
            #[cfg(feature = "custom-alloc")]
            {
                node.strategy().to_vec_in(StackAlloc)
            }
            #[cfg(not(feature = "custom-alloc"))]
            {
                node.strategy().to_vec()
            }
        };

        // normalizes the strategy
        normalize_strategy(&mut cfreach_actions, node.num_actions());

        // updates the reach probabilities
        let row_size = cfreach_actions.len() / node.num_actions();
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

#[inline]
pub(crate) fn normalize_strategy(slice: &mut [f32], num_actions: usize) {
    let row_size = slice.len() / num_actions;

    #[cfg(feature = "custom-alloc")]
    let mut denom = vec::from_elem_in(0.0, row_size, StackAlloc);
    #[cfg(not(feature = "custom-alloc"))]
    let mut denom = vec![0.0; row_size];

    slice.chunks(row_size).for_each(|row| {
        add_slice(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    slice.chunks_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });
}
