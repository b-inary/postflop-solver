use crate::bet_size::*;
use crate::mutex_like::*;

#[cfg(feature = "bincode")]
use bincode::{error::DecodeError, Decode, Encode};

pub(crate) const PLAYER_OOP: u8 = 0;
// pub(crate) const PLAYER_IP: u8 = 1;
pub(crate) const PLAYER_CHANCE: u8 = 2;
pub(crate) const PLAYER_MASK: u8 = 3;
pub(crate) const PLAYER_TERMINAL_FLAG: u8 = 4;
pub(crate) const PLAYER_FOLD_FLAG: u8 = 12;

/// Available actions of the postflop game.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub enum Action {
    /// (Default value)
    #[default]
    None,

    /// Fold action.
    Fold,

    /// Check action.
    Check,

    /// Call action.
    Call,

    /// Bet action with a specified amount.
    Bet(i32),

    /// Raise action with a specified amount.
    Raise(i32),

    /// All-in action with a specified amount.
    AllIn(i32),

    /// Chance action with a card ID (of range [`0`, `52`)).
    Chance(u8),
}

/// An enum representing the board state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub enum BoardState {
    #[default]
    Flop,
    Turn,
    River,
}

/// A struct containing the game tree configuration.
///
/// # Examples
/// ```
/// use postflop_solver::*;
///
/// let bet_sizes = BetSizeCandidates::try_from(("60%, e, a", "2.5x")).unwrap();
/// let donk_sizes = DonkSizeCandidates::try_from("50%").unwrap();
///
/// let tree_config = TreeConfig {
///     initial_state: BoardState::Turn,
///     starting_pot: 200,
///     effective_stack: 900,
///     flop_bet_sizes: Default::default(),
///     turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
///     turn_donk_sizes: None,
///     river_donk_sizes: Some(donk_sizes),
///     add_allin_threshold: 1.5,
///     force_allin_threshold: 0.15,
///     merging_threshold: 0.1,
/// };
/// ```
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct TreeConfig {
    /// Initial state of the game (flop, turn, or river).
    pub initial_state: BoardState,

    /// Starting pot size. Must be greater than `0`.
    pub starting_pot: i32,

    /// Initial effective stack. Must be greater than `0`.
    pub effective_stack: i32,

    /// Bet size candidates of each player for the flop.
    pub flop_bet_sizes: [BetSizeCandidates; 2],

    /// Bet size candidates of each player for the turn.
    pub turn_bet_sizes: [BetSizeCandidates; 2],

    /// Bet size candidates of each player for the river.
    pub river_bet_sizes: [BetSizeCandidates; 2],

    /// Donk size candidates for the turn (set `None` to use default sizes).
    pub turn_donk_sizes: Option<DonkSizeCandidates>,

    /// Donk size candidates for the river (set `None` to use default sizes).
    pub river_donk_sizes: Option<DonkSizeCandidates>,

    /// Add all-in action if the ratio of maximum bet size to the pot is below or equal to this
    /// value (set `0.0` to disable).
    pub add_allin_threshold: f32,

    /// Force all-in action if the SPR (stack/pot) after the opponent's call is below or equal to
    /// this value (set `0.0` to disable).
    ///
    /// Personal recommendation: between `0.1` and `0.2`
    pub force_allin_threshold: f32,

    /// Merge bet actions if there are bet actions with "close" values (set `0.0` to disable).
    ///
    /// Algorithm: The same as PioSOLVER. That is, select the highest bet size (= X% of the pot) and
    /// remove all bet actions with a value (= Y% of the pot) satisfying the following inequality:
    ///   (100 + X) / (100 + Y) < 1.0 + threshold.
    /// Continue this process with the next highest bet size.
    ///
    /// Personal recommendation: around `0.1`
    pub merging_threshold: f32,
}

/// A struct representing an abstract game tree.
#[derive(Default)]
pub struct ActionTree {
    config: TreeConfig,
    added_lines: Vec<Vec<Action>>,
    removed_lines: Vec<Vec<Action>>,
    root: Box<MutexLike<ActionTreeNode>>,
    action_history: Vec<Action>,
    node_history: Vec<*const ActionTreeNode>,
}

// automatic derive of `Decode` does not work (2.0.0-rc.2)
#[derive(Default)]
#[cfg_attr(feature = "bincode", derive(Encode))]
pub(crate) struct ActionTreeNode {
    pub(crate) player: u8,
    pub(crate) board_state: BoardState,
    pub(crate) amount: i32,
    pub(crate) actions: Vec<Action>,
    pub(crate) children: Vec<MutexLike<ActionTreeNode>>,
}

#[derive(Default)]
struct BuildTreeInfo {
    last_action: Action,
    allin_flag: bool,
    oop_call_flag: bool,
    bet_amount: [i32; 2],
}

type EjectedActionTree = (
    TreeConfig,
    Vec<Vec<Action>>,
    Vec<Vec<Action>>,
    Box<MutexLike<ActionTreeNode>>,
);

impl ActionTree {
    /// Creates a new [`ActionTree`] with the specified configuration.
    #[inline]
    pub fn new(config: TreeConfig) -> Result<Self, String> {
        Self::check_config(&config)?;
        let mut ret = Self {
            config,
            ..Default::default()
        };
        ret.build_tree();
        ret.node_history.push(&*ret.root.lock());
        Ok(ret)
    }

    /// Obtains the configuration of the game tree.
    #[inline]
    pub fn config(&self) -> &TreeConfig {
        &self.config
    }

    /// Obtains the list of added lines.
    #[inline]
    pub fn added_lines(&self) -> &[Vec<Action>] {
        &self.added_lines
    }

    /// Obtains the list of removed lines.
    #[inline]
    pub fn removed_lines(&self) -> &[Vec<Action>] {
        &self.removed_lines
    }

    /// Adds a given line to the action tree.
    ///
    /// - `line` except the last action must exist in the current tree.
    /// - The last action of the `line` must not exist in the current tree.
    /// - Except for the case that the `line` is in the removed lines, the last action of the `line`
    ///   must be a bet action (including raise and all-in action).
    /// - Chance actions (i.e., dealing turn and river cards) must be omitted from the `line`.
    #[inline]
    pub fn add_line(&mut self, line: &[Action]) -> Result<(), String> {
        let removed_index = self.removed_lines.iter().position(|x| x == line);
        let info = BuildTreeInfo::default();
        self.add_line_recursive(&mut self.root.lock(), line, removed_index.is_some(), info)?;
        if let Some(index) = removed_index {
            self.removed_lines.remove(index);
        } else {
            self.added_lines.push(line.to_vec());
        }
        Ok(())
    }

    /// Removes a given line from the action tree.
    ///
    /// - `line` must exist in the current tree.
    /// - Chance actions (i.e., dealing turn and river cards) must be omitted from the `line`.
    /// - If the current node is removed by this method, the current node is moved to the nearest
    ///   ancestor node that is not removed.
    #[inline]
    pub fn remove_line(&mut self, line: &[Action]) -> Result<(), String> {
        Self::remove_line_recursive(&mut self.root.lock(), line)?;
        let added_index = self.added_lines.iter().position(|x| x == line);
        if let Some(index) = added_index {
            self.added_lines.remove(index);
        } else {
            self.removed_lines.push(line.to_vec());
        }
        if self.action_history.starts_with(line) {
            self.action_history.truncate(line.len() - 1);
            self.node_history.truncate(line.len());
        }
        Ok(())
    }

    /// Obtains all possible actions at the current node.
    ///
    /// If the current node is a chance node, returns possible actions after the chance event.
    #[inline]
    pub fn actions(&self) -> &[Action] {
        &self.current_node_skip_chance().actions
    }

    /// Returns a list of booleans indicating whether the corresponding action is terminal.
    #[inline]
    pub fn is_terminal_action(&self) -> Vec<bool> {
        let node = self.current_node_skip_chance();
        node.children
            .iter()
            .map(|child| {
                let child = child.lock();
                child.is_terminal() || child.amount == self.config.effective_stack
            })
            .collect()
    }

    /// Plays the given action. Returns `Ok(())` if the action is valid.
    ///
    /// The `action` must be one of the possible actions at the current node.
    /// If the current node is a chance node, the chance action is automatically played before
    /// playing the given action.
    #[inline]
    pub fn play(&mut self, action: Action) -> Result<(), String> {
        let node = self.current_node_skip_chance();
        let search_result = node.actions.binary_search(&action);
        if search_result.is_err() {
            return Err(format!("Action `{:?}` is not available", action));
        }

        let index = search_result.unwrap();
        self.node_history.push(&*node.children[index].lock());
        self.action_history.push(action);

        Ok(())
    }

    /// Undoes the last action. Returns `Ok(())` if the action is successfully undone.
    #[inline]
    pub fn undo(&mut self) -> Result<(), String> {
        if self.action_history.is_empty() {
            return Err("No action to undo".to_string());
        }

        self.action_history.pop();
        self.node_history.pop();

        Ok(())
    }

    /// Moves back to the root node.
    #[inline]
    pub fn back_to_root(&mut self) {
        self.action_history.clear();
        self.node_history = vec![&*self.root.lock()];
    }

    /// Obtains the current action history.
    #[inline]
    pub fn history(&self) -> &[Action] {
        &self.action_history
    }

    /// Applies the given action history from the root node.
    #[inline]
    pub fn apply_history(&mut self, history: &[Action]) -> Result<(), String> {
        self.back_to_root();
        for action in history {
            self.play(*action)?;
        }
        Ok(())
    }

    /// Returns whether the current node is a chance node.
    #[inline]
    pub fn is_chance_node(&self) -> bool {
        self.current_node().is_chance()
    }

    /// Adds a given action to the current node.
    ///
    /// Internally, this method calls [`add_line`] with the current action history and the given
    /// action. See [`add_line`] for the details.
    ///
    /// [`add_line`]: #method.add_line
    #[inline]
    pub fn add_action(&mut self, action: Action) -> Result<(), String> {
        let mut action_line = self.action_history.clone();
        action_line.push(action);
        self.add_line(&action_line)
    }

    /// Removes a given action from the current node.
    ///
    /// Internally, this method calls [`remove_line`] with the current action history and the given
    /// action. See [`remove_line`] for the details.
    ///
    /// [`remove_line`]: #method.remove_line
    #[inline]
    pub fn remove_action(&mut self, action: Action) -> Result<(), String> {
        let mut action_line = self.action_history.clone();
        action_line.push(action);
        self.remove_line(&action_line)
    }

    /// Removes the current node.
    ///
    /// Internally, this method calls [`remove_line`] with the current action history. See
    /// [`remove_line`] for the details.
    ///
    /// [`remove_line`]: #method.remove_line
    #[inline]
    pub fn remove_current_node(&mut self) -> Result<(), String> {
        let action_history = self.action_history.clone();
        self.remove_line(&action_history)
    }

    /// Ejects the fields.
    #[inline]
    pub(crate) fn eject(self) -> EjectedActionTree {
        (self.config, self.added_lines, self.removed_lines, self.root)
    }

    /// Returns the reference to the current node.
    #[inline]
    fn current_node(&self) -> &ActionTreeNode {
        unsafe { &**self.node_history.last().unwrap() }
    }

    /// Returns the reference to the current node skipping chance nodes.
    #[inline]
    fn current_node_skip_chance(&self) -> &ActionTreeNode {
        unsafe {
            let mut node = *self.node_history.last().unwrap();
            while (*node).is_chance() {
                node = &*(*node).children[0].lock();
            }
            &*node
        }
    }

    /// Checks the configuration.
    #[inline]
    fn check_config(config: &TreeConfig) -> Result<(), String> {
        if config.starting_pot <= 0 {
            return Err(format!(
                "Starting pot must be positive: {}",
                config.starting_pot
            ));
        }

        if config.effective_stack <= 0 {
            return Err(format!(
                "Effective stack must be positive: {}",
                config.effective_stack
            ));
        }

        if config.add_allin_threshold < 0.0 {
            return Err(format!(
                "Add all-in threshold must be non-negative: {}",
                config.add_allin_threshold
            ));
        }

        if config.force_allin_threshold < 0.0 {
            return Err(format!(
                "Force all-in threshold must be non-negative: {}",
                config.force_allin_threshold
            ));
        }

        if config.merging_threshold < 0.0 {
            return Err(format!(
                "Merging threshold must be non-negative: {}",
                config.merging_threshold
            ));
        }

        Ok(())
    }

    /// Builds the action tree.
    #[inline]
    fn build_tree(&mut self) {
        let mut root = self.root.lock();
        *root = ActionTreeNode::default();
        root.board_state = self.config.initial_state;
        self.build_tree_recursive(&mut root, BuildTreeInfo::default());
    }

    /// Recursively builds the action tree.
    fn build_tree_recursive(&self, node: &mut ActionTreeNode, info: BuildTreeInfo) {
        if node.is_terminal() {
            // Do nothing
        } else if node.is_chance() {
            let next_state = match node.board_state {
                BoardState::Flop => BoardState::Turn,
                BoardState::Turn => BoardState::River,
                BoardState::River => unreachable!(),
            };

            let next_player = match (info.allin_flag, node.board_state) {
                (false, _) => PLAYER_OOP,
                (true, BoardState::Flop) => PLAYER_CHANCE,
                (true, _) => PLAYER_TERMINAL_FLAG,
            };

            node.actions.push(Action::Chance(0));
            node.children.push(MutexLike::new(ActionTreeNode {
                player: next_player,
                board_state: next_state,
                amount: node.amount,
                ..Default::default()
            }));

            self.build_tree_recursive(
                &mut node.children[0].lock(),
                info.create_next(PLAYER_CHANCE, Action::Chance(0)),
            );
        } else {
            self.push_actions(node, &info);
            for (action, child) in node.actions.iter().zip(node.children.iter()) {
                self.build_tree_recursive(
                    &mut child.lock(),
                    info.create_next(node.player, *action),
                );
            }
        }
    }

    /// Pushes all possible actions to the given node.
    fn push_actions(&self, node: &mut ActionTreeNode, info: &BuildTreeInfo) {
        let player = node.player;
        let opponent = node.player ^ 1;

        let player_amount = info.bet_amount[player as usize];
        let opponent_amount = info.bet_amount[opponent as usize];

        let amount_diff = opponent_amount - player_amount;
        let pot = self.config.starting_pot + 2 * (node.amount + amount_diff);

        let max_amount = self.config.effective_stack - node.amount + player_amount;
        let min_amount = (opponent_amount + amount_diff).clamp(1, max_amount);

        let stack_after_call = self.config.effective_stack - node.amount - amount_diff;
        let spr_after_call = stack_after_call as f64 / pot as f64;
        let compute_geometric = |num_streets: i32, max_ratio: f32| {
            let ratio = ((2.0 * spr_after_call + 1.0).powf(1.0 / num_streets as f64) - 1.0) / 2.0;
            (pot as f64 * ratio.min(max_ratio as f64)).round() as i32
        };

        let (candidates, donk_candidates, num_remaining_streets) = match node.board_state {
            BoardState::Flop => (&self.config.flop_bet_sizes, &None, 3),
            BoardState::Turn => (&self.config.turn_bet_sizes, &self.config.turn_donk_sizes, 2),
            BoardState::River => (
                &self.config.river_bet_sizes,
                &self.config.river_donk_sizes,
                1,
            ),
        };

        let mut actions = Vec::new();

        if donk_candidates.is_some()
            && matches!(info.last_action, Action::Chance(_))
            && info.oop_call_flag
        {
            // check
            actions.push(Action::Check);

            // donk bet
            for &donk_size in &donk_candidates.as_ref().unwrap().donk {
                match donk_size {
                    BetSize::PotRelative(ratio) => {
                        let amount = (pot as f32 * ratio).round() as i32;
                        actions.push(Action::Bet(amount));
                    }
                    BetSize::PrevBetRelative(_) => panic!("Unexpected `PrevBetRelative`"),
                    BetSize::Additive(adder) => actions.push(Action::Bet(adder)),
                    BetSize::Geometric(num_streets, max_ratio) => {
                        let num_streets = match num_streets {
                            0 => num_remaining_streets,
                            _ => num_streets,
                        };
                        let amount = compute_geometric(num_streets, max_ratio);
                        actions.push(Action::Bet(amount));
                    }
                    BetSize::AllIn => actions.push(Action::AllIn(max_amount)),
                }
            }

            // all-in
            if max_amount <= (pot as f32 * self.config.add_allin_threshold).round() as i32 {
                actions.push(Action::AllIn(max_amount));
            }
        } else if matches!(
            info.last_action,
            Action::None | Action::Check | Action::Chance(_)
        ) {
            // check
            actions.push(Action::Check);

            // bet
            for &bet_size in &candidates[player as usize].bet {
                match bet_size {
                    BetSize::PotRelative(ratio) => {
                        let amount = (pot as f32 * ratio).round() as i32;
                        actions.push(Action::Bet(amount));
                    }
                    BetSize::PrevBetRelative(_) => panic!("Unexpected `PrevBetRelative`"),
                    BetSize::Additive(adder) => actions.push(Action::Bet(adder)),
                    BetSize::Geometric(num_streets, max_ratio) => {
                        let num_streets = match num_streets {
                            0 => num_remaining_streets,
                            _ => num_streets,
                        };
                        let amount = compute_geometric(num_streets, max_ratio);
                        actions.push(Action::Bet(amount));
                    }
                    BetSize::AllIn => actions.push(Action::AllIn(max_amount)),
                }
            }

            // all-in
            if max_amount <= (pot as f32 * self.config.add_allin_threshold).round() as i32 {
                actions.push(Action::AllIn(max_amount));
            }
        } else {
            // fold
            actions.push(Action::Fold);

            // call
            actions.push(Action::Call);

            if !info.allin_flag {
                // raise
                for &bet_size in &candidates[player as usize].raise {
                    match bet_size {
                        BetSize::PotRelative(ratio) => {
                            let amount = opponent_amount + (pot as f32 * ratio).round() as i32;
                            actions.push(Action::Raise(amount));
                        }
                        BetSize::PrevBetRelative(ratio) => {
                            let amount = (opponent_amount as f32 * ratio).round() as i32;
                            actions.push(Action::Raise(amount));
                        }
                        BetSize::Additive(adder) => {
                            actions.push(Action::Raise(opponent_amount + adder));
                        }
                        BetSize::Geometric(num_streets, max_ratio) => {
                            let num_streets = match num_streets {
                                0 => num_remaining_streets,
                                _ => num_streets,
                            };
                            let amount = compute_geometric(num_streets, max_ratio);
                            actions.push(Action::Raise(opponent_amount + amount));
                        }
                        BetSize::AllIn => actions.push(Action::AllIn(max_amount)),
                    }
                }

                // all-in
                let all_in_threshold = pot as f32 * self.config.add_allin_threshold;
                if max_amount <= opponent_amount + all_in_threshold.round() as i32 {
                    actions.push(Action::AllIn(max_amount));
                }
            }
        }

        let is_above_threshold = |amount: i32| {
            let new_amount_diff = amount - opponent_amount;
            let new_pot = pot + 2 * new_amount_diff;
            let threshold = (new_pot as f32 * self.config.force_allin_threshold).round() as i32;
            max_amount <= amount + threshold
        };

        // clamp bet amounts
        for action in actions.iter_mut() {
            match *action {
                Action::Bet(amount) => {
                    let clamped = amount.clamp(min_amount, max_amount);
                    if is_above_threshold(clamped) {
                        *action = Action::AllIn(max_amount);
                    } else if clamped != amount {
                        *action = Action::Bet(clamped);
                    }
                }
                Action::Raise(amount) => {
                    let clamped = amount.clamp(min_amount, max_amount);
                    if is_above_threshold(clamped) {
                        *action = Action::AllIn(max_amount);
                    } else if clamped != amount {
                        *action = Action::Raise(clamped);
                    }
                }
                _ => {}
            }
        }

        // remove duplicates
        actions.sort_unstable();
        actions.dedup();

        // merge bet actions with close amounts
        actions = merge_bet_actions(actions, pot, opponent_amount, self.config.merging_threshold);

        let player_after_call = match num_remaining_streets {
            1 => PLAYER_TERMINAL_FLAG,
            _ => PLAYER_CHANCE,
        };

        let player_after_check = match player {
            PLAYER_OOP => opponent,
            _ => player_after_call,
        };

        // push actions
        for action in actions {
            let mut amount = node.amount;
            let next_player = match action {
                Action::Fold => PLAYER_FOLD_FLAG | player,
                Action::Check => player_after_check,
                Action::Call => {
                    amount += amount_diff;
                    player_after_call
                }
                Action::Bet(_) | Action::Raise(_) | Action::AllIn(_) => {
                    amount += amount_diff;
                    opponent
                }
                _ => panic!("Unexpected action: {:?}", action),
            };

            node.actions.push(action);
            node.children.push(MutexLike::new(ActionTreeNode {
                player: next_player,
                board_state: node.board_state,
                amount,
                ..Default::default()
            }));
        }

        node.actions.shrink_to_fit();
        node.children.shrink_to_fit();
    }

    /// Recursive function to add a given line to the tree.
    fn add_line_recursive(
        &self,
        node: &mut ActionTreeNode,
        line: &[Action],
        was_removed: bool,
        info: BuildTreeInfo,
    ) -> Result<(), String> {
        if line.is_empty() {
            return Err("Empty line".to_string());
        }

        if node.is_terminal() {
            return Err("Unexpected terminal node".to_string());
        }

        if node.is_chance() {
            return self.add_line_recursive(
                &mut node.children[0].lock(),
                line,
                was_removed,
                info.create_next(PLAYER_CHANCE, Action::Chance(0)),
            );
        }

        let action = line[0];
        let search_result = node.actions.binary_search(&action);

        let player = node.player;
        let opponent = node.player ^ 1;

        if line.len() > 1 {
            if search_result.is_err() {
                return Err(format!("Action does not exist: {:?}", action));
            }

            return self.add_line_recursive(
                &mut node.children[search_result.unwrap()].lock(),
                &line[1..],
                was_removed,
                info.create_next(player, action),
            );
        }

        if search_result.is_ok() {
            return Err(format!("Action already exists: {:?}", action));
        }

        let player_bet = info.bet_amount[player as usize];
        let opponent_bet = info.bet_amount[opponent as usize ^ 1];
        let bet_diff = opponent_bet - player_bet;

        let max_amount = self.config.effective_stack - node.amount + player_bet;
        let min_amount = (opponent_bet + bet_diff).clamp(1, max_amount);

        let action = match action {
            Action::Bet(amount) | Action::Raise(amount) if amount == max_amount => {
                Action::AllIn(amount)
            }
            _ => action,
        };

        let is_valid_bet = match action {
            Action::Bet(amount) if amount >= min_amount && amount < max_amount => {
                matches!(
                    info.last_action,
                    Action::None | Action::Check | Action::Chance(_)
                )
            }
            Action::Raise(amount) if amount >= min_amount && amount < max_amount => {
                matches!(info.last_action, Action::Bet(_) | Action::Raise(_))
            }
            Action::AllIn(amount) => amount == max_amount,
            _ => false,
        };

        if !was_removed && !is_valid_bet {
            match action {
                Action::Bet(amount) | Action::Raise(amount) => {
                    return Err(format!(
                        "Invalid bet amount: {} (min: {}, max: {})",
                        amount, min_amount, max_amount
                    ));
                }
                Action::AllIn(amount) => {
                    return Err(format!(
                        "Invalid all-in amount: {} (expected: {})",
                        amount, max_amount
                    ));
                }
                _ => {
                    return Err(format!("Invalid action: {:?}", action));
                }
            };
        }

        let player_after_call = match node.board_state {
            BoardState::River => PLAYER_TERMINAL_FLAG,
            _ => PLAYER_CHANCE,
        };

        let player_after_check = match player {
            PLAYER_OOP => opponent,
            _ => player_after_call,
        };

        let mut amount = node.amount;
        let next_player = match action {
            Action::Fold => PLAYER_FOLD_FLAG | player,
            Action::Check => player_after_check,
            Action::Call => {
                amount += bet_diff;
                player_after_call
            }
            Action::Bet(_) | Action::Raise(_) | Action::AllIn(_) => {
                amount += bet_diff;
                opponent
            }
            _ => panic!("Unexpected action: {:?}", action),
        };

        let index = search_result.unwrap_err();
        node.actions.insert(index, action);
        node.children.insert(
            index,
            MutexLike::new(ActionTreeNode {
                player: next_player,
                board_state: node.board_state,
                amount,
                ..Default::default()
            }),
        );

        node.actions.shrink_to_fit();
        node.children.shrink_to_fit();

        self.build_tree_recursive(
            &mut node.children[index].lock(),
            info.create_next(player, action),
        );

        Ok(())
    }

    /// Recursive function to remove a given line from the tree.
    fn remove_line_recursive(node: &mut ActionTreeNode, line: &[Action]) -> Result<(), String> {
        if line.is_empty() {
            return Err("Empty line".to_string());
        }

        if node.is_terminal() {
            return Err("Unexpected terminal node".to_string());
        }

        if node.is_chance() {
            return Self::remove_line_recursive(&mut node.children[0].lock(), line);
        }

        let action = line[0];
        let search_result = node.actions.binary_search(&action);
        if search_result.is_err() {
            return Err(format!("Action does not exist: {:?}", action));
        }

        if line.len() > 1 {
            return Self::remove_line_recursive(
                &mut node.children[search_result.unwrap()].lock(),
                &line[1..],
            );
        }

        let index = search_result.unwrap();
        node.actions.remove(index);
        node.children.remove(index);

        node.actions.shrink_to_fit();
        node.children.shrink_to_fit();

        Ok(())
    }
}

impl ActionTreeNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    #[inline]
    fn is_chance(&self) -> bool {
        self.player == PLAYER_CHANCE
    }
}

impl BuildTreeInfo {
    #[inline]
    fn create_next(&self, player: u8, action: Action) -> Self {
        let mut allin_flag = self.allin_flag;
        let mut oop_call_flag = self.oop_call_flag;
        let mut bet_amount = self.bet_amount;

        match action {
            Action::Check => {
                oop_call_flag = false;
            }
            Action::Call => {
                oop_call_flag = player == PLAYER_OOP;
                bet_amount[player as usize] = bet_amount[player as usize ^ 1];
            }
            Action::Bet(amount) | Action::Raise(amount) => {
                bet_amount[player as usize] = amount;
            }
            Action::AllIn(amount) => {
                allin_flag = true;
                bet_amount[player as usize] = amount;
            }
            Action::Chance(_) => {
                bet_amount = [0, 0];
            }
            _ => {}
        }

        BuildTreeInfo {
            last_action: action,
            allin_flag,
            oop_call_flag,
            bet_amount,
        }
    }
}

#[cfg(feature = "bincode")]
impl Decode for ActionTreeNode {
    #[inline]
    fn decode<D: bincode::de::Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(ActionTreeNode {
            player: Decode::decode(decoder)?,
            board_state: Decode::decode(decoder)?,
            amount: Decode::decode(decoder)?,
            actions: Decode::decode(decoder)?,
            children: Decode::decode(decoder)?,
        })
    }
}

fn merge_bet_actions(actions: Vec<Action>, pot: i32, offset: i32, param: f32) -> Vec<Action> {
    const EPS: f32 = 2e-7; // 2 ulps

    let get_amount = |action: Action| match action {
        Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => amount,
        _ => -1,
    };

    let mut cur_amount = i32::MAX;
    let mut ret = Vec::new();

    for &action in actions.iter().rev() {
        let amount = get_amount(action);
        if amount > 0 {
            let ratio = (amount - offset) as f32 / pot as f32;
            let cur_ratio = (cur_amount - offset) as f32 / pot as f32;
            let threshold_ratio = (cur_ratio - param) / (1.0 + param);
            if ratio < threshold_ratio * (1.0 - EPS) {
                ret.push(action);
                cur_amount = amount;
            }
        } else {
            ret.push(action);
        }
    }

    ret.reverse();
    ret
}
