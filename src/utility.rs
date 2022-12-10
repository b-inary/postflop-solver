use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use std::mem;
use std::ptr;

#[cfg(feature = "custom-alloc")]
use {crate::alloc::*, std::vec};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Executes `op` for each child potentially in parallel.
#[cfg(feature = "rayon")]
#[inline]
pub(crate) fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    if node.enable_parallelization() {
        node.actions().into_par_iter().for_each(op);
    } else {
        node.actions().for_each(op);
    }
}

/// Executes `op` for each child.
#[cfg(not(feature = "rayon"))]
#[inline]
pub(crate) fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    node.actions().for_each(op);
}

#[inline]
pub(crate) fn vec_memory_usage<T>(vec: &Vec<T>) -> u64 {
    vec.capacity() as u64 * mem::size_of::<T>() as u64
}

#[inline]
fn weighted_sum(values: &[f32], weights: &[f32]) -> f32 {
    let f = |sum: f64, (&v, &w): (&f32, &f32)| sum + v as f64 * w as f64;
    values.iter().zip(weights).fold(0.0, f) as f32
}

#[inline]
fn max(x: f32, y: f32) -> f32 {
    if x > y {
        x
    } else {
        y
    }
}

/// Obtains the maximum absolute value of the given slice.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn slice_absolute_max(slice: &[f32]) -> f32 {
    if slice.len() < 32 {
        slice.iter().fold(0.0, |a, x| max(a, x.abs()))
    } else {
        use std::arch::wasm32::*;

        unsafe {
            let slice_ptr = slice.as_ptr() as *const v128;
            let mut tmp: [v128; 4] = [
                f32x4_abs(v128_load(slice_ptr)),
                f32x4_abs(v128_load(slice_ptr.add(1))),
                f32x4_abs(v128_load(slice_ptr.add(2))),
                f32x4_abs(v128_load(slice_ptr.add(3))),
            ];

            let mut iter = slice[16..].chunks_exact(16);
            for chunk in iter.by_ref() {
                let chunk_ptr = chunk.as_ptr() as *const v128;
                tmp[0] = f32x4_max(tmp[0], f32x4_abs(v128_load(chunk_ptr)));
                tmp[1] = f32x4_max(tmp[1], f32x4_abs(v128_load(chunk_ptr.add(1))));
                tmp[2] = f32x4_max(tmp[2], f32x4_abs(v128_load(chunk_ptr.add(2))));
                tmp[3] = f32x4_max(tmp[3], f32x4_abs(v128_load(chunk_ptr.add(3))));
            }

            tmp[0] = f32x4_max(tmp[0], tmp[1]);
            tmp[2] = f32x4_max(tmp[2], tmp[3]);
            tmp[0] = f32x4_max(tmp[0], tmp[2]);
            let tmpmax = max(
                max(
                    f32x4_extract_lane::<0>(tmp[0]),
                    f32x4_extract_lane::<1>(tmp[0]),
                ),
                max(
                    f32x4_extract_lane::<2>(tmp[0]),
                    f32x4_extract_lane::<3>(tmp[0]),
                ),
            );

            iter.remainder().iter().fold(tmpmax, |a, x| max(a, x.abs()))
        }
    }
}

/// Obtains the maximum absolute value of the given slice.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn slice_absolute_max(slice: &[f32]) -> f32 {
    if slice.len() < 32 {
        slice.iter().fold(0.0, |a, x| max(a, x.abs()))
    } else {
        let mut tmp: [f32; 16] = slice[..16].try_into().unwrap();
        tmp.iter_mut().for_each(|x| *x = x.abs());
        let mut iter = slice[16..].chunks_exact(16);
        for chunk in iter.by_ref() {
            for i in 0..16 {
                tmp[i] = max(tmp[i], chunk[i].abs());
            }
        }
        let tmpmax = tmp.iter().fold(0.0f32, |a, &x| max(a, x));
        iter.remainder().iter().fold(tmpmax, |a, x| max(a, x.abs()))
    }
}

/// Obtains the maximum value of the given non-negative slice.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn slice_nonnegative_max(slice: &[f32]) -> f32 {
    if slice.len() < 32 {
        slice.iter().fold(0.0, |a, &x| max(a, x))
    } else {
        use std::arch::wasm32::*;

        unsafe {
            let slice_ptr = slice.as_ptr() as *const v128;
            let mut tmp: [v128; 4] = [
                v128_load(slice_ptr),
                v128_load(slice_ptr.add(1)),
                v128_load(slice_ptr.add(2)),
                v128_load(slice_ptr.add(3)),
            ];

            let mut iter = slice[16..].chunks_exact(16);
            for chunk in iter.by_ref() {
                let chunk_ptr = chunk.as_ptr() as *const v128;
                tmp[0] = f32x4_max(tmp[0], v128_load(chunk_ptr));
                tmp[1] = f32x4_max(tmp[1], v128_load(chunk_ptr.add(1)));
                tmp[2] = f32x4_max(tmp[2], v128_load(chunk_ptr.add(2)));
                tmp[3] = f32x4_max(tmp[3], v128_load(chunk_ptr.add(3)));
            }

            tmp[0] = f32x4_max(tmp[0], tmp[1]);
            tmp[2] = f32x4_max(tmp[2], tmp[3]);
            tmp[0] = f32x4_max(tmp[0], tmp[2]);
            let tmpmax = max(
                max(
                    f32x4_extract_lane::<0>(tmp[0]),
                    f32x4_extract_lane::<1>(tmp[0]),
                ),
                max(
                    f32x4_extract_lane::<2>(tmp[0]),
                    f32x4_extract_lane::<3>(tmp[0]),
                ),
            );

            iter.remainder().iter().fold(tmpmax, |a, &x| max(a, x))
        }
    }
}

/// Obtains the maximum value of the given non-negative slice.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn slice_nonnegative_max(slice: &[f32]) -> f32 {
    if slice.len() < 32 {
        slice.iter().fold(0.0, |a, &x| max(a, x))
    } else {
        let mut tmp: [f32; 16] = slice[..16].try_into().unwrap();
        let mut iter = slice[16..].chunks_exact(16);
        for chunk in iter.by_ref() {
            for i in 0..16 {
                tmp[i] = max(tmp[i], chunk[i]);
            }
        }
        let tmpmax = tmp.iter().fold(0.0f32, |a, &x| max(a, x));
        iter.remainder().iter().fold(tmpmax, |a, &x| max(a, x))
    }
}

/// Encodes the `f32` slice to the `i16` slice, and returns the scale.
#[inline]
pub(crate) fn encode_signed_slice(dst: &mut [i16], slice: &[f32]) -> f32 {
    let scale = slice_absolute_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = i16::MAX as f32 / scale_nonzero;
    dst.iter_mut()
        .zip(slice)
        .for_each(|(d, s)| *d = unsafe { (s * encoder).round().to_int_unchecked::<i32>() as i16 });
    scale
}

/// Encodes the `f32` slice to the `u16` slice, and returns the scale.
#[inline]
pub(crate) fn encode_unsigned_slice(dst: &mut [u16], slice: &[f32]) -> f32 {
    let scale = slice_nonnegative_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u16::MAX as f32 / scale_nonzero;
    // note: 0.49999997 + 0.49999997 = 0.99999994 < 1.0 | 0.5 + 0.49999997 = 1.0
    dst.iter_mut().zip(slice).for_each(|(d, s)| {
        *d = unsafe { (s * encoder + 0.49999997).to_int_unchecked::<i32>() as u16 }
    });
    scale
}

/// Finalizes the solving process.
#[inline]
pub fn finalize<T: Game>(game: &mut T) {
    if game.is_solved() {
        panic!("the game is already solved");
    }

    if !game.is_ready() {
        panic!("the game is not ready");
    }

    let mut cfvalues = [
        vec![0.0; game.num_private_hands(0)],
        vec![0.0; game.num_private_hands(1)],
    ];

    // compute the expected values and save them
    for player in 0..2 {
        let cfreach = game.initial_weights(player ^ 1);
        compute_cfvalue_recursive(
            &mut cfvalues[player],
            game,
            &mut game.root(),
            player,
            cfreach,
            true,
        );
    }

    // set the game solved
    game.set_solved(&cfvalues[1]);

    // free buffer
    #[cfg(all(feature = "custom-alloc", feature = "rayon"))]
    rayon::broadcast(|_| free_custom_alloc_buffer());
    #[cfg(all(feature = "custom-alloc", not(feature = "rayon")))]
    free_custom_alloc_buffer();
}

/// Computes the average of the expected values of the current strategy.
#[inline]
pub fn compute_current_ev_average<T: Game>(game: &T) -> f32 {
    if !game.is_ready() && !game.is_solved() {
        panic!("the game is not ready");
    }

    let mut cfvalues = [
        vec![0.0; game.num_private_hands(0)],
        vec![0.0; game.num_private_hands(1)],
    ];

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    for player in 0..2 {
        compute_cfvalue_recursive(
            &mut cfvalues[player],
            game,
            &mut game.root(),
            player,
            reach[player ^ 1],
            false,
        );
    }

    let get_sum = |player: usize| weighted_sum(&cfvalues[player], reach[player]);
    0.5 * (get_sum(0) + get_sum(1))
}

/// Computes the average of the expected values of the MES (Maximally Exploitative Strategy).
///
/// Corresponds to the exploitability value when not raked.
#[inline]
pub fn compute_mes_ev_average<T: Game>(game: &T) -> f32 {
    if !game.is_ready() && !game.is_solved() {
        panic!("the game is not ready");
    }

    let mut cfvalues = [
        vec![0.0; game.num_private_hands(0)],
        vec![0.0; game.num_private_hands(1)],
    ];

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    for player in 0..2 {
        compute_best_cfv_recursive(
            &mut cfvalues[player],
            game,
            &game.root(),
            player,
            reach[player ^ 1],
        );
    }

    let get_sum = |player: usize| weighted_sum(&cfvalues[player], reach[player]);
    0.5 * (get_sum(0) + get_sum(1))
}

/// The recursive helper function for computing the counterfactual values of the given strategy.
fn compute_cfvalue_recursive<T: Game>(
    result: &mut [f32],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    save_cfvalues: bool,
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = game.num_private_hands(player);

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(vec::from_elem_in(0.0, num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(vec![0.0; num_actions * num_hands]);

    // chance node
    if node.is_chance() {
        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = vec::from_elem_in(0.0, num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = vec![0.0; num_hands];

        // update the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach = cfreach.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach,
                save_cfvalues,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        cfv_actions.chunks_exact(num_hands).for_each(|row| {
            result_f64.iter_mut().zip(row).for_each(|(r, &v)| {
                *r += v as f64;
            });
        });

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

        // process isomorphic chances
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);

            for &(i, j) in swap_list {
                unsafe {
                    ptr::swap(
                        tmp.get_unchecked_mut(i as usize),
                        tmp.get_unchecked_mut(j as usize),
                    );
                }
            }

            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });

            for &(i, j) in swap_list {
                unsafe {
                    ptr::swap(
                        tmp.get_unchecked_mut(i as usize),
                        tmp.get_unchecked_mut(j as usize),
                    );
                }
            }
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            *r = v as f32;
        });

        // save the counterfactual values
        if save_cfvalues {
            let slice: &[f32] = match node.cfvalue_storage(player) {
                CfValueStorage::None => &[],
                CfValueStorage::Sum => result,
                CfValueStorage::All => cfv_actions.as_slice(),
            };
            if !slice.is_empty() {
                if game.is_compression_enabled() {
                    let dst = node.cfvalues_chance_compressed_mut(player);
                    let cfv_scale = encode_signed_slice(dst, slice);
                    node.set_cfvalue_chance_scale(player, cfv_scale);
                } else {
                    node.cfvalues_chance_mut(player).copy_from_slice(slice);
                }
            }
        }
    }
    // player node
    else if node.player() == player {
        // obtain the strategy
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

        // normalize the strategy
        normalize_strategy(&mut strategy, node.num_actions());

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                save_cfvalues,
            );
        });

        // sum up the counterfactual values
        let cfv_actions = cfv_actions.lock();
        mul_slice(&mut strategy, &cfv_actions);
        strategy.chunks_exact(num_hands).for_each(|row| {
            add_slice(result, row);
        });

        // save the counterfactual values
        if save_cfvalues {
            if game.is_compression_enabled() {
                let cfv_scale = encode_signed_slice(node.cfvalues_compressed_mut(), &cfv_actions);
                node.set_cfvalue_scale(cfv_scale);
            } else {
                node.cfvalues_mut().copy_from_slice(&cfv_actions);
            }
        }
    }
    // opponent node
    else if num_actions == 1 {
        // simply recurse when the number of actions is one
        compute_cfvalue_recursive(
            result,
            game,
            &mut node.play(0),
            player,
            cfreach,
            save_cfvalues,
        );
    } else {
        // obtain the strategy
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

        // normalize the strategy
        normalize_strategy(&mut cfreach_actions, node.num_actions());

        // update the reach probabilities
        let row_size = cfreach_actions.len() / node.num_actions();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_cfvalue_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                save_cfvalues,
            );
        });

        // sum up the counterfactual values
        let cfv_actions = cfv_actions.lock();
        cfv_actions.chunks_exact(num_hands).for_each(|row| {
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
    let num_hands = game.num_private_hands(player);

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        let child = &node.play(0);
        compute_best_cfv_recursive(result, game, child, player, cfreach);
        return;
    }

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(vec::from_elem_in(0.0, num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(vec![0.0; num_actions * num_hands]);

    // chance node
    if node.is_chance() {
        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = vec::from_elem_in(0.0, num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = vec![0.0; num_hands];

        // update the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach = cfreach.to_vec_in(StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_best_cfv_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_hands),
                game,
                &node.play(action),
                player,
                &cfreach,
            )
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        cfv_actions.chunks_exact(num_hands).for_each(|row| {
            result_f64.iter_mut().zip(row).for_each(|(r, v)| {
                *r += *v as f64;
            });
        });

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

        // process isomorphic chances
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);

            for &(i, j) in swap_list {
                unsafe {
                    ptr::swap(
                        tmp.get_unchecked_mut(i as usize),
                        tmp.get_unchecked_mut(j as usize),
                    );
                }
            }

            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });

            for &(i, j) in swap_list {
                unsafe {
                    ptr::swap(
                        tmp.get_unchecked_mut(i as usize),
                        tmp.get_unchecked_mut(j as usize),
                    );
                }
            }
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            *r = v as f32;
        });
    }
    // player node
    else if node.player() == player {
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_best_cfv_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_hands),
                game,
                &node.play(action),
                player,
                cfreach,
            )
        });

        // compute element-wise maximum (take the best response)
        result.fill(f32::MIN);
        let cfv_actions = cfv_actions.lock();
        cfv_actions.chunks_exact(num_hands).for_each(|row| {
            result.iter_mut().zip(row).for_each(|(r, v)| {
                *r = max(*r, *v);
            });
        });
    }
    // opponent node
    else {
        // obtain the strategy
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

        // normalize the strategy
        normalize_strategy(&mut cfreach_actions, node.num_actions());

        // update the reach probabilities
        let row_size = cfreach_actions.len() / node.num_actions();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            let cfreach = row(&cfreach_actions, action, row_size);
            if cfreach.iter().any(|&x| x > 0.0) {
                compute_best_cfv_recursive(
                    row_mut(&mut cfv_actions.lock(), action, num_hands),
                    game,
                    &node.play(action),
                    player,
                    cfreach,
                );
            }
        });

        // sum up the counterfactual values
        let cfv_actions = cfv_actions.lock();
        cfv_actions.chunks_exact(num_hands).for_each(|row| {
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

    slice.chunks_exact(row_size).for_each(|row| {
        add_slice(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    slice.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });
}
