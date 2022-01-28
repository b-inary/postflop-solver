use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::io::{stdout, Write};

const ALPHA: f32 = 1.5;
const BETA: f32 = 0.0;
const GAMMA: f32 = 4.0;

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

/// Performs CFR until the given number of iterations or exploitability is satisfied, and returns
/// the exploitability.
pub fn solve<T: Game>(
    game: &T,
    num_iterations: i32,
    target_exploitability: f32,
    bias: f32,
    show_progress: bool,
) -> f32 {
    let mut root = game.root();
    let reach = [game.initial_reach(0), game.initial_reach(1)];

    if show_progress {
        print!("iteration: 0 / {}", num_iterations);
        stdout().flush().unwrap();
    }

    let mut exploitability = f32::INFINITY;

    for t in 0..num_iterations {
        let mut cfv = [
            vec![0.0; game.num_private_hands(0)],
            vec![0.0; game.num_private_hands(1)],
        ];

        let t_f32 = t as f32;
        let params = DiscountParams {
            alpha_t: t_f32.powf(ALPHA) / (t_f32.powf(ALPHA) + 1.0),
            beta_t: t_f32.powf(BETA) / (t_f32.powf(BETA) + 1.0),
            gamma_t: t_f32.powf(GAMMA),
        };

        // alternating updates
        for player in 0..2 {
            solve_recursive(
                &mut cfv[player],
                game,
                &mut root,
                player,
                reach[player],
                reach[player ^ 1],
                &params,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == num_iterations {
            exploitability = compute_exploitability(game, bias, false);
            if show_progress {
                print!("\riteration: {} / {} ", t + 1, num_iterations);
                print!("(exploitability = {:.4e}[bb])", exploitability);
                stdout().flush().unwrap();
            }
            if exploitability <= target_exploitability {
                break;
            }
        } else if show_progress {
            print!("\riteration: {} / {}", t + 1, num_iterations);
            stdout().flush().unwrap();
        }
    }

    if show_progress {
        println!();
        stdout().flush().unwrap();
    }

    normalize_strategy(game);

    if num_iterations > 0 {
        exploitability
    } else {
        compute_exploitability(game, bias, true)
    }
}

/// Recursively solves the counterfactual values.
fn solve_recursive<T: Game>(
    result: &mut [f32],
    game: &T,
    node: &mut T::Node,
    player: usize,
    reach: &[f32],
    cfreach: &[f32],
    params: &DiscountParams,
) {
    // returns the counterfactual values when the `node` is terminal
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_private_hands = game.num_private_hands(player);

    // allocates memory for storing the counterfactual values
    let cfv_actions = MutexLike::new(vec![0.0; num_actions * num_private_hands]);

    // if the `node` is chance
    if node.is_chance() {
        // updates the reach probabilities
        let mut cfreach = cfreach.to_vec();
        mul_slice_scalar(&mut cfreach, node.chance_factor());

        // computes the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &mut node.play(action),
                player,
                reach,
                &cfreach,
                params,
            );
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
    // if the current player is `player`
    else if node.player() == player {
        // computes the strategy by regret-maching algorithm
        let strategy = regret_matching(node.cum_regret(), num_actions);

        // updates the reach probabilities
        let mut reach_actions = strategy.clone();
        reach_actions.chunks_mut(num_private_hands).for_each(|row| {
            mul_slice(row, reach);
        });

        // computes the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                game,
                &mut node.play(action),
                player,
                row(&reach_actions, action, num_private_hands),
                cfreach,
                params,
            );
        });

        // sums up the counterfactual values
        let cfv_actions = cfv_actions.lock();
        let mut cfv_strategy = strategy;
        mul_slice(&mut cfv_strategy, &cfv_actions);
        cfv_strategy.chunks(num_private_hands).for_each(|row| {
            add_slice(result, row);
        });

        // updates the cumulative regret
        let cum_regret = node.cum_regret_mut();
        cum_regret.iter_mut().for_each(|el| {
            *el *= if *el >= 0.0 {
                params.alpha_t
            } else {
                params.beta_t
            };
        });
        add_slice(cum_regret, &cfv_actions);
        cum_regret.chunks_mut(num_private_hands).for_each(|row| {
            sub_slice(row, result);
        });

        // updates the cumulative strategy
        let cum_strategy = node.strategy_mut();
        mul_slice_scalar(&mut reach_actions, params.gamma_t);
        add_slice(cum_strategy, &reach_actions);
    }
    // if the current player is not `player`
    else {
        // computes the strategy by regret-matching algorithm
        let mut cfreach_actions = regret_matching(node.cum_regret(), num_actions);
        let row_size = cfreach_actions.len() / num_actions;

        // updates the reach probabilities
        cfreach_actions.chunks_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // computes the counterfactual values of each action
        for_each_child(node, |action| {
            let cfreach = row(&cfreach_actions, action, row_size);
            if cfreach.iter().any(|&x| x > 0.0) {
                solve_recursive(
                    row_mut(&mut cfv_actions.lock(), action, num_private_hands),
                    game,
                    &mut node.play(action),
                    player,
                    reach,
                    cfreach,
                    params,
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

/// Computes the strategy by regret-mathcing algorithm.
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32> {
    let mut strategy = regret.to_vec();
    let row_size = strategy.len() / num_actions;

    strategy.iter_mut().for_each(|el| *el = el.max(0.0));

    let mut denom = vec![0.0; row_size];
    strategy.chunks(row_size).for_each(|row| {
        add_slice(&mut denom, row);
    });

    let default = 1.0 / num_actions as f32;
    strategy.chunks_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}
