use crate::mutex_like::*;
use ndarray::prelude::*;
use std::ops::Range;

/// The trait representing a game.
pub trait Game: Sync {
    /// The type representing a node in game tree.
    type Node: GameNode;

    /// Returns the root node of game tree.
    fn root(&self) -> MutexGuardLike<Self::Node>;

    /// Returns the number of private hands of given player.
    fn num_private_hands(&self, player: usize) -> usize;

    /// Returns the initial reach probabilities of given player.
    fn initial_reach(&self, player: usize) -> &Array1<f32>;

    /// Computes the counterfactual values of given node.
    fn evaluate(
        &self,
        result: &mut ArrayViewMut1<f32>,
        node: &Self::Node,
        player: usize,
        cfreach: &ArrayView1<f32>,
    );
}

/// The trait representing a node in game tree.
pub trait GameNode: Sync {
    /// Returns whether the node is terminal.
    fn is_terminal(&self) -> bool;

    /// Returns whether the node is chance.
    fn is_chance(&self) -> bool;

    /// Returns the current player.
    fn player(&self) -> usize;

    /// Returns the number of actions.
    fn num_actions(&self) -> usize;

    /// Returns the range struct of actions.
    fn actions(&self) -> Range<usize> {
        0..self.num_actions()
    }

    /// Returns the effective coefficient of chance.
    fn chance_factor(&self) -> f32;

    /// Returns the list of isomorphic chances.
    fn isomorphic_chances(&self) -> &Vec<IsomorphicChance>;

    /// Returns the node after taking the given action.
    fn play(&self, action: usize) -> MutexGuardLike<Self>;

    /// Returns the cumulative regrets.
    fn cum_regret(&self) -> &Array2<f32>;

    /// Returns the mutable reference of the cumulative regrets.
    fn cum_regret_mut(&mut self) -> &mut Array2<f32>;

    /// Returns the cumulative strategy.
    fn strategy(&self) -> &Array2<f32>;

    /// Returns the mutable reference of the cumulative strategy.
    fn strategy_mut(&mut self) -> &mut Array2<f32>;
}

/// The struct representing an isomorphic chance branch.
pub struct IsomorphicChance {
    /// The index of isomorphic chance.
    pub index: usize,

    /// The swap indices of each player.
    pub swap_list: [Vec<(usize, usize)>; 2],
}
