use crate::mutex_like::*;
use std::mem::MaybeUninit;
use std::ops::Range;

/// The trait representing a game.
pub trait Game: Send + Sync {
    /// The type representing a node in game tree.
    #[doc(hidden)]
    type Node: GameNode;

    /// Returns the root node of game tree.
    #[doc(hidden)]
    fn root(&self) -> MutexGuardLike<Self::Node>;

    /// Returns the number of private hands of given player.
    #[doc(hidden)]
    fn num_private_hands(&self, player: usize) -> usize;

    /// Returns the initial reach probabilities of given player.
    #[doc(hidden)]
    fn initial_weights(&self, player: usize) -> &[f32];

    /// Computes the counterfactual values of given node.
    #[doc(hidden)]
    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    );

    /// Returns the effective number of chances.
    #[doc(hidden)]
    fn chance_factor(&self, node: &Self::Node) -> usize;

    /// Returns whether the instance is solved.
    #[doc(hidden)]
    fn is_solved(&self) -> bool;

    /// Sets the instance to be solved.
    #[doc(hidden)]
    fn set_solved(&mut self);

    /// Returns whether the instance is ready to be solved.
    #[doc(hidden)]
    fn is_ready(&self) -> bool {
        true
    }

    /// Returns whether the game is raked.
    #[doc(hidden)]
    fn is_raked(&self) -> bool {
        false
    }

    /// Returns the list of indices that isomorphic chances refer to.
    #[doc(hidden)]
    fn isomorphic_chances(&self, _node: &Self::Node) -> &[u8] {
        &[]
    }

    /// Returns the swap list of the given isomorphic chance.
    #[doc(hidden)]
    fn isomorphic_swap(&self, _node: &Self::Node, _index: usize) -> &[Vec<(u16, u16)>; 2] {
        unreachable!()
    }

    /// Returns the locking strategy.
    #[doc(hidden)]
    fn locking_strategy(&self, _node: &Self::Node) -> &[f32] {
        &[]
    }

    /// Returns whether the compression is enabled.
    #[doc(hidden)]
    fn is_compression_enabled(&self) -> bool {
        false
    }
}

/// The trait representing a node in game tree.
pub trait GameNode: Send + Sync {
    /// Returns whether the node is terminal.
    #[doc(hidden)]
    fn is_terminal(&self) -> bool;

    /// Returns whether the node is chance.
    #[doc(hidden)]
    fn is_chance(&self) -> bool;

    /// Returns the current player.
    #[doc(hidden)]
    fn player(&self) -> usize;

    /// Returns the number of actions.
    #[doc(hidden)]
    fn num_actions(&self) -> usize;

    /// Returns the node after taking the given action.
    #[doc(hidden)]
    fn play(&self, action: usize) -> MutexGuardLike<Self>;

    /// Returns the strategy.
    #[doc(hidden)]
    fn strategy(&self) -> &[f32];

    /// Returns the mutable reference to the strategy.
    #[doc(hidden)]
    fn strategy_mut(&mut self) -> &mut [f32];

    /// Returns the cumulative regrets.
    #[doc(hidden)]
    fn regrets(&self) -> &[f32];

    /// Returns the mutable reference to the cumulative regrets.
    #[doc(hidden)]
    fn regrets_mut(&mut self) -> &mut [f32];

    /// Returns the counterfactual values.
    #[doc(hidden)]
    fn cfvalues(&self) -> &[f32];

    /// Returns the mutable reference to the counterfactual values.
    #[doc(hidden)]
    fn cfvalues_mut(&mut self) -> &mut [f32];

    /// Returns whether IP's counterfactual values are stored.
    #[doc(hidden)]
    fn has_cfvalues_ip(&self) -> bool {
        false
    }

    /// Returns IP's counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip(&self) -> &[f32] {
        unreachable!()
    }

    /// Returns the mutable reference to IP's counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_mut(&mut self) -> &mut [f32] {
        unreachable!()
    }

    /// Returns the player whose counterfactual values are stored (for chance node).
    #[doc(hidden)]
    fn cfvalue_storage_player(&self) -> Option<usize> {
        None
    }

    /// Returns the buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance(&self) -> &[f32] {
        unreachable!()
    }

    /// Returns the mutable reference to the buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_mut(&mut self) -> &mut [f32] {
        unreachable!()
    }

    /// Returns the [`Range`] struct of actions.
    #[doc(hidden)]
    fn action_indices(&self) -> Range<usize> {
        0..self.num_actions()
    }

    /// Returns the compressed strategy.
    #[doc(hidden)]
    fn strategy_compressed(&self) -> &[u16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed strategy.
    #[doc(hidden)]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unreachable!()
    }

    /// Returns the compressed cumulative regrets.
    #[doc(hidden)]
    fn regrets_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed cumulative regrets.
    #[doc(hidden)]
    fn regrets_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns the compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns IP's compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to IP's compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns the scale of the compressed strategy.
    #[doc(hidden)]
    fn strategy_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed strategy.
    #[doc(hidden)]
    fn set_strategy_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed cumulative regrets.
    #[doc(hidden)]
    fn regret_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed cumulative regrets.
    #[doc(hidden)]
    fn set_regret_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalue_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed counterfactual values.
    #[doc(hidden)]
    fn set_cfvalue_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed counterfactual values for IP.
    #[doc(hidden)]
    fn cfvalue_ip_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed counterfactual values for IP.
    #[doc(hidden)]
    fn set_cfvalue_ip_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalue_chance_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn set_cfvalue_chance_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Hint for parallelization. By default, it is set to `false`.
    #[doc(hidden)]
    fn enable_parallelization(&self) -> bool {
        false
    }
}
