use crate::interface::*;
use crate::utility::*;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::io::{stdout, Write};

const ALPHA: f32 = 1.5;
const BETA: f32 = 0.0;
const GAMMA: f32 = 2.0;

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
            Array1::zeros(game.num_private_hands(0)),
            Array1::zeros(game.num_private_hands(1)),
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
                &mut cfv[player].view_mut(),
                game,
                &mut root,
                player,
                &reach[player].view(),
                &reach[player ^ 1].view(),
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
    result: &mut ArrayViewMut1<f32>,
    game: &T,
    node: &mut T::Node,
    player: usize,
    reach: &ArrayView1<f32>,
    cfreach: &ArrayView1<f32>,
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
    let mut cfv_actions = Array2::zeros((num_actions, num_private_hands));

    // if the `node` is chance
    if node.is_chance() {
        // updates the reach probabilities
        let cfreach = cfreach * node.chance_factor();

        // computes the counterfactual values of each action
        cfv_actions
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(action, mut cfv)| {
                solve_recursive(
                    &mut cfv,
                    game,
                    &mut node.play(action),
                    player,
                    reach,
                    &cfreach.view(),
                    params,
                );
            });

        // sums up the counterfactual values
        cfv_actions.outer_iter().for_each(|cfv| {
            *result += &cfv;
        });

        // get information about isomorphic chances
        let iso_chances = node.isomorphic_chances();

        // processes isomorphic chances
        let mut tmp = Array1::zeros(num_private_hands);
        for iso_chance in iso_chances {
            tmp.assign(&cfv_actions.row(iso_chance.index));
            for (i, j) in &iso_chance.swap_list[player] {
                tmp.swap(*i, *j);
            }
            *result += &tmp;
        }
    }
    // if the current player is `player`
    else if node.player() == player {
        // computes the strategy by regret-maching algorithm
        let strategy = regret_matching(node.cum_regret());

        // updates the reach probabilities
        let reach_actions = reach * &strategy;

        // computes the counterfactual values of each action
        cfv_actions
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(action, mut cfv)| {
                solve_recursive(
                    &mut cfv,
                    game,
                    &mut node.play(action),
                    player,
                    &reach_actions.row(action),
                    cfreach,
                    params,
                );
            });

        // sums up the counterfactual values
        let cfv_strategy = &cfv_actions * &strategy;
        cfv_strategy.outer_iter().for_each(|cfv| {
            *result += &cfv;
        });

        // updates the cumulative regret
        let cum_regret = node.cum_regret_mut();
        cum_regret.mapv_inplace(|el| {
            if el >= 0.0 {
                el * params.alpha_t
            } else {
                el * params.beta_t
            }
        });
        *cum_regret += &cfv_actions;
        *cum_regret -= &*result;

        // updates the cumulative strategy
        let cum_strategy = node.strategy_mut();
        let weighted_strategy = reach_actions * params.gamma_t;
        *cum_strategy += &weighted_strategy;
    }
    // if the current player is not `player`
    else {
        // computes the strategy by regret-matching algorithm
        let strategy = regret_matching(node.cum_regret());

        // updates the reach probabilities
        let cfreach_actions = cfreach * strategy;

        // computes the counterfactual values of each action
        cfv_actions
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(action, mut cfv)| {
                if cfreach_actions.row(action).sum() > 0.0 {
                    solve_recursive(
                        &mut cfv,
                        game,
                        &mut node.play(action),
                        player,
                        reach,
                        &cfreach_actions.row(action),
                        params,
                    );
                }
            });

        // sums up the counterfactual values
        cfv_actions.outer_iter().for_each(|cfv| {
            *result += &cfv;
        })
    }
}

/// Computes the strategy by regret-mathcing algorithm.
#[inline]
fn regret_matching(regret: &Array2<f32>) -> Array2<f32> {
    let mut strategy = regret.clone();

    strategy.mapv_inplace(|el| el.max(0.0));
    strategy /= &strategy.sum_axis(Axis(0));

    let default = 1.0 / strategy.nrows() as f32;
    strategy.mapv_inplace(|el| if el.is_nan() { default } else { el });

    strategy
}
