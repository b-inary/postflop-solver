use crate::mutex_like::*;
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
    fn initial_reach(&self, player: usize) -> &[f32];

    /// Computes the counterfactual values of given node.
    fn evaluate(&self, result: &mut [f32], node: &Self::Node, player: usize, cfreach: &[f32]);

    /// Returns whether the compression is enabled.
    fn is_compression_enabled(&self) -> bool {
        false
    }
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
    fn cum_regret(&self) -> &[f32];

    /// Returns the mutable reference to the cumulative regrets.
    fn cum_regret_mut(&mut self) -> &mut [f32];

    /// Returns the strategy.
    fn strategy(&self) -> &[f32];

    /// Returns the mutable reference to the strategy.
    fn strategy_mut(&mut self) -> &mut [f32];

    /// Returns the compressed cumulative regrets.
    fn cum_regret_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed cumulative regrets.
    fn cum_regret_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns the compressed strategy.
    fn strategy_compressed(&self) -> &[u16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed strategy.
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unreachable!()
    }

    /// Returns the scale of the compressed cumulative regrets.
    fn cum_regret_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed cumulative regrets.
    fn set_cum_regret_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed strategy.
    fn strategy_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed strategy.
    fn set_strategy_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Hint for parallelization. By default, it is set to `false`.
    fn enable_parallelization(&self) -> bool {
        false
    }
}

/// The struct representing an isomorphic chance branch.
#[derive(Debug, Clone)]
pub struct IsomorphicChance {
    /// The index of isomorphic chance.
    pub index: usize,

    /// The swap indices of each player.
    pub swap_list: [Vec<(usize, usize)>; 2],
}
