use crate::interface::*;
use ndarray::prelude::*;
use rayon::prelude::*;

/// Normalizes the cumulative strategy.
pub fn normalize_strategy<T: Game>(game: &T) {
    normalize_strategy_recursive::<T>(&mut game.root());
}

/// Computes the expected value of `player`'s payoff.
pub fn compute_ev<T: Game>(game: &T, player: usize) -> f32 {
    let reach = [game.initial_reach(0), game.initial_reach(1)];
    compute_ev_recursive(
        game,
        &game.root(),
        player,
        &reach[player].view(),
        &reach[1 - player].view(),
    )
}

/// Computes the exploitability of the strategy.
pub fn compute_exploitability<T: Game>(game: &T, is_normalized: bool) -> f32 {
    let mut cfv = [
        Array1::zeros(game.num_private_hands(0)),
        Array1::zeros(game.num_private_hands(1)),
    ];
    let reach = [game.initial_reach(0), game.initial_reach(1)];
    for player in 0..2 {
        compute_best_cfv_recursive(
            &mut cfv[player].view_mut(),
            game,
            &game.root(),
            player,
            &reach[1 - player].view(),
            is_normalized,
        );
    }
    (cfv[0].sum() + cfv[1].sum()) / 2.0
}

/// The recursive helper function for normalizing the strategy.
fn normalize_strategy_recursive<T: Game>(node: &mut T::Node) {
    if node.is_terminal() {
        return;
    }

    if !node.is_chance() {
        let strategy = node.strategy_mut();
        *strategy /= &strategy.sum_axis(Axis(0));

        let default = 1.0 / strategy.nrows() as f32;
        strategy.mapv_inplace(|el| if el.is_nan() { default } else { el });
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
    reach: &ArrayView1<f32>,
    cfreach: &ArrayView1<f32>,
) -> f32 {
    // terminal node
    if node.is_terminal() {
        let mut cfv = Array1::zeros(game.num_private_hands(player));
        game.evaluate(&mut cfv.view_mut(), node, player, cfreach);
        cfv.dot(reach)
    }
    // chance node
    else if node.is_chance() {
        let cfreach = cfreach * node.chance_factor();
        let evs = node
            .actions()
            .into_par_iter()
            .map(|action| {
                compute_ev_recursive(game, &node.play(action), player, reach, &cfreach.view())
            })
            .collect::<Vec<_>>();
        let mut ev = evs.iter().sum();
        for iso_chance in node.isomorphic_chances() {
            ev += evs[iso_chance.index];
        }
        ev
    }
    // player node
    else if node.player() == player {
        let reach_actions = reach * node.strategy();
        node.actions()
            .into_par_iter()
            .map(|action| {
                compute_ev_recursive(
                    game,
                    &node.play(action),
                    player,
                    &reach_actions.row(action),
                    cfreach,
                )
            })
            .sum()
    }
    // opponent node
    else {
        let cfreach_actions = cfreach * node.strategy();
        node.actions()
            .into_par_iter()
            .map(|action| {
                compute_ev_recursive(
                    game,
                    &node.play(action),
                    player,
                    reach,
                    &cfreach_actions.row(action),
                )
            })
            .sum()
    }
}

/// The recursive helper function for computing the counterfactual values of best response.
fn compute_best_cfv_recursive<T: Game>(
    result: &mut ArrayViewMut1<f32>,
    game: &T,
    node: &T::Node,
    player: usize,
    cfreach: &ArrayView1<f32>,
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
    let mut cfv_actions = Array2::zeros((num_actions, num_private_hands));

    // chance node
    if node.is_chance() {
        // updates the reach probabilities
        let cfreach = cfreach * node.chance_factor();

        // computes the counterfactual values of each action
        cfv_actions
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(action, mut cfv)| {
                compute_best_cfv_recursive(
                    &mut cfv,
                    game,
                    &node.play(action),
                    player,
                    &cfreach.view(),
                    is_normalized,
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
    // player node
    else if node.player() == player {
        // computes the counterfactual values of each action
        cfv_actions
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(action, mut cfv)| {
                compute_best_cfv_recursive(
                    &mut cfv,
                    game,
                    &node.play(action),
                    player,
                    cfreach,
                    is_normalized,
                );
            });

        // computes element-wise maximum (takes the best response)
        result.fill(f32::NEG_INFINITY);
        cfv_actions.outer_iter().for_each(|cfv| {
            result.zip_mut_with(&cfv, |l, r| {
                *l = l.max(*r);
            });
        });
    }
    // opponent node
    else {
        // updates the reach probabilities
        let cfreach_actions = if is_normalized {
            cfreach * node.strategy()
        } else {
            // if the strategy is not normalized, we need to normalize it
            let mut strategy = node.strategy().clone();
            let default = 1.0 / strategy.nrows() as f32;
            strategy /= &strategy.sum_axis(Axis(0));
            strategy.mapv_inplace(|el| if el.is_nan() { default } else { el });
            cfreach * strategy
        };

        // computes the counterfactual values of each action
        cfv_actions
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(action, mut cfv)| {
                compute_best_cfv_recursive(
                    &mut cfv,
                    game,
                    &node.play(action),
                    player,
                    &cfreach_actions.row(action),
                    is_normalized,
                );
            });

        // sums up the counterfactual values
        cfv_actions.outer_iter().for_each(|cfv| {
            *result += &cfv;
        });
    }
}
