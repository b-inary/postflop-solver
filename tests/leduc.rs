extern crate postflop_solver;
use postflop_solver::*;
use std::mem::MaybeUninit;
use std::slice;

struct LeducGame {
    root: MutexLike<LeducNode>,
    initial_weight: Vec<f32>,
    isomorphism: Vec<u8>,
    isomorphism_swap: [Vec<(u16, u16)>; 2],
    is_solved: bool,
    is_compression_enabled: bool,
}

struct LeducNode {
    player: usize,
    board: usize,
    amount: i32,
    children: Vec<(Action, MutexLike<LeducNode>)>,
    strategy: Vec<f32>,
    storage: Vec<f32>,
    strategy_scale: f32,
    storage_scale: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    None,
    Fold,
    Check,
    Call,
    Bet(i32),
    Raise(i32),
    Chance(usize),
}

const NUM_PRIVATE_HANDS: usize = 6;

#[allow(dead_code)]
const PLAYER_OOP: usize = 0;

#[allow(dead_code)]
const PLAYER_IP: usize = 1;

const PLAYER_CHANCE: usize = 0xff;
const PLAYER_MASK: usize = 0xff;
const PLAYER_TERMINAL_FLAG: usize = 0x100;
const PLAYER_FOLD_FLAG: usize = 0x300;

const NOT_DEALT: usize = 0xff;

impl Game for LeducGame {
    type Node = LeducNode;

    #[inline]
    fn root(&self) -> MutexGuardLike<Self::Node> {
        self.root.lock()
    }

    #[inline]
    fn num_private_hands(&self, _player: usize) -> usize {
        NUM_PRIVATE_HANDS
    }

    #[inline]
    fn initial_weights(&self, _player: usize) -> &[f32] {
        &self.initial_weight
    }

    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    ) {
        result.iter_mut().for_each(|x| {
            x.write(0.0);
        });
        let result = unsafe { &mut *(result as *mut _ as *mut [f32]) };

        let num_hands = NUM_PRIVATE_HANDS * (NUM_PRIVATE_HANDS - 1);
        let num_hands_inv = 1.0 / num_hands as f32;
        let amount_normalized = node.amount as f32 * num_hands_inv;

        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let sign = [1.0, -1.0][(player == folded_player) as usize];
            let payoff_normalized = amount_normalized * sign;
            for my_card in 0..NUM_PRIVATE_HANDS {
                if my_card != node.board {
                    for opp_card in 0..NUM_PRIVATE_HANDS {
                        if my_card != opp_card && opp_card != node.board {
                            result[my_card] += payoff_normalized * cfreach[opp_card];
                        }
                    }
                }
            }
        } else {
            for my_card in 0..NUM_PRIVATE_HANDS {
                if my_card != node.board {
                    for opp_card in 0..NUM_PRIVATE_HANDS {
                        if my_card != opp_card && opp_card != node.board {
                            let sign = match () {
                                _ if my_card / 2 == node.board / 2 => 1.0,
                                _ if opp_card / 2 == node.board / 2 => -1.0,
                                _ if my_card / 2 == opp_card / 2 => 0.0,
                                _ if my_card > opp_card => 1.0,
                                _ => -1.0,
                            };
                            let payoff_normalized = amount_normalized * sign;
                            result[my_card] += payoff_normalized * cfreach[opp_card];
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn chance_factor(&self, _node: &Self::Node) -> usize {
        4
    }

    #[inline]
    fn isomorphic_chances(&self, _node: &Self::Node) -> &[u8] {
        &self.isomorphism
    }

    #[inline]
    fn isomorphic_swap(&self, _node: &Self::Node, _index: usize) -> &[Vec<(u16, u16)>; 2] {
        &self.isomorphism_swap
    }

    #[inline]
    fn is_solved(&self) -> bool {
        self.is_solved
    }

    #[inline]
    fn set_solved(&mut self) {
        self.is_solved = true;
    }

    #[inline]
    fn is_compression_enabled(&self) -> bool {
        self.is_compression_enabled
    }
}

impl LeducGame {
    #[inline]
    pub fn new(is_compression_enabled: bool) -> Self {
        Self {
            root: Self::build_tree(),
            initial_weight: vec![1.0; NUM_PRIVATE_HANDS],
            isomorphism: vec![0, 1, 2],
            isomorphism_swap: [vec![(0, 1), (2, 3), (4, 5)], vec![(0, 1), (2, 3), (4, 5)]],
            is_solved: false,
            is_compression_enabled,
        }
    }

    fn build_tree() -> MutexLike<LeducNode> {
        let mut root = LeducNode {
            player: PLAYER_OOP,
            board: NOT_DEALT,
            amount: 1,
            children: Vec::new(),
            strategy: Default::default(),
            storage: Default::default(),
            strategy_scale: 0.0,
            storage_scale: 0.0,
        };
        Self::build_tree_recursive(&mut root, Action::None, [0, 0]);
        Self::allocate_memory_recursive(&mut root);
        MutexLike::new(root)
    }

    fn build_tree_recursive(node: &mut LeducNode, prev_action: Action, prev_amount: [i32; 2]) {
        if node.is_terminal() {
            return;
        }

        if node.is_chance() {
            Self::push_chance_actions(node);
            for action in node.action_indices() {
                Self::build_tree_recursive(&mut node.play(action), Action::Chance(action), [0, 0]);
            }
            return;
        }

        let actions = Self::get_actions(node, prev_action, node.board != NOT_DEALT);

        let mut next_amounts = Vec::new();
        let prev_amount_min = prev_amount.iter().min().unwrap();

        for (action, next_player) in &actions {
            let mut next_amount = prev_amount;
            if *action == Action::Call {
                next_amount[node.player] = next_amount[node.player ^ 1];
            }
            if let Action::Bet(amount) = action {
                next_amount[node.player] = *amount;
            }
            if let Action::Raise(amount) = action {
                next_amount[node.player] = *amount;
            }

            next_amounts.push(next_amount);
            let amount_diff = next_amount.iter().min().unwrap() - prev_amount_min;

            node.children.push((
                *action,
                MutexLike::new(LeducNode {
                    player: *next_player,
                    board: node.board,
                    amount: node.amount + amount_diff,
                    children: Vec::new(),
                    strategy: Default::default(),
                    storage: Default::default(),
                    strategy_scale: 0.0,
                    storage_scale: 0.0,
                }),
            ));
        }

        for action in node.action_indices() {
            Self::build_tree_recursive(
                &mut node.play(action),
                actions[action].0,
                next_amounts[action],
            );
        }
    }

    fn push_chance_actions(node: &mut LeducNode) {
        for index in 0..3 {
            node.children.push((
                Action::Chance(index * 2),
                MutexLike::new(LeducNode {
                    player: PLAYER_OOP,
                    board: index * 2,
                    amount: node.amount,
                    children: Vec::new(),
                    strategy: Default::default(),
                    storage: Default::default(),
                    strategy_scale: 0.0,
                    storage_scale: 0.0,
                }),
            ));
        }
    }

    fn get_actions(
        node: &LeducNode,
        prev_action: Action,
        is_second_round: bool,
    ) -> Vec<(Action, usize)> {
        let raise_amount = [2, 4][is_second_round as usize];

        let player = node.player;
        let player_opponent = player ^ 1;

        let player_after_call = if is_second_round {
            PLAYER_TERMINAL_FLAG | player
        } else {
            PLAYER_CHANCE
        };

        let player_after_check = if player == PLAYER_OOP {
            player_opponent
        } else {
            player_after_call
        };

        let mut actions = Vec::new();

        match prev_action {
            Action::None | Action::Check | Action::Chance(_) => {
                actions.push((Action::Check, player_after_check));
                actions.push((Action::Bet(raise_amount), player_opponent));
            }
            Action::Bet(amount) => {
                actions.push((Action::Fold, PLAYER_FOLD_FLAG | player));
                actions.push((Action::Call, player_after_call));
                actions.push((Action::Raise(amount + raise_amount), player_opponent));
            }
            Action::Raise(_) => {
                actions.push((Action::Fold, PLAYER_FOLD_FLAG | player));
                actions.push((Action::Call, player_after_call));
            }
            Action::Fold | Action::Call => unreachable!(),
        };

        actions
    }

    fn allocate_memory_recursive(node: &mut LeducNode) {
        if node.is_terminal() {
            return;
        }

        if !node.is_chance() {
            let num_actions = node.num_actions();
            node.strategy = vec![0.0; num_actions * NUM_PRIVATE_HANDS];
            node.storage = vec![0.0; num_actions * NUM_PRIVATE_HANDS];
        }

        for action in node.action_indices() {
            Self::allocate_memory_recursive(&mut node.play(action));
        }
    }
}

impl GameNode for LeducNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    #[inline]
    fn is_chance(&self) -> bool {
        self.player == PLAYER_CHANCE
    }

    #[inline]
    fn player(&self) -> usize {
        self.player
    }

    #[inline]
    fn num_actions(&self) -> usize {
        self.children.len()
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<Self> {
        self.children[action].1.lock()
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        &self.strategy
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        &mut self.strategy
    }

    #[inline]
    fn regrets(&self) -> &[f32] {
        &self.storage
    }

    #[inline]
    fn regrets_mut(&mut self) -> &mut [f32] {
        &mut self.storage
    }

    #[inline]
    fn cfvalues(&self) -> &[f32] {
        &self.storage
    }

    #[inline]
    fn cfvalues_mut(&mut self) -> &mut [f32] {
        &mut self.storage
    }

    #[inline]
    fn strategy_compressed(&self) -> &[u16] {
        let ptr = self.strategy.as_ptr() as *const u16;
        unsafe { slice::from_raw_parts(ptr, self.strategy.len()) }
    }

    #[inline]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        let ptr = self.strategy.as_mut_ptr() as *mut u16;
        unsafe { slice::from_raw_parts_mut(ptr, self.strategy.len()) }
    }

    #[inline]
    fn regrets_compressed(&self) -> &[i16] {
        let ptr = self.storage.as_ptr() as *const i16;
        unsafe { slice::from_raw_parts(ptr, self.storage.len()) }
    }

    #[inline]
    fn regrets_compressed_mut(&mut self) -> &mut [i16] {
        let ptr = self.storage.as_mut_ptr() as *mut i16;
        unsafe { slice::from_raw_parts_mut(ptr, self.storage.len()) }
    }

    #[inline]
    fn cfvalues_compressed(&self) -> &[i16] {
        let ptr = self.storage.as_ptr() as *const i16;
        unsafe { slice::from_raw_parts(ptr, self.storage.len()) }
    }

    #[inline]
    fn cfvalues_compressed_mut(&mut self) -> &mut [i16] {
        let ptr = self.storage.as_mut_ptr() as *mut i16;
        unsafe { slice::from_raw_parts_mut(ptr, self.storage.len()) }
    }

    #[inline]
    fn strategy_scale(&self) -> f32 {
        self.strategy_scale
    }

    #[inline]
    fn set_strategy_scale(&mut self, scale: f32) {
        self.strategy_scale = scale;
    }

    #[inline]
    fn regret_scale(&self) -> f32 {
        self.storage_scale
    }

    #[inline]
    fn set_regret_scale(&mut self, scale: f32) {
        self.storage_scale = scale;
    }

    #[inline]
    fn cfvalue_scale(&self) -> f32 {
        self.storage_scale
    }

    #[inline]
    fn set_cfvalue_scale(&mut self, scale: f32) {
        self.storage_scale = scale;
    }
}

#[test]
fn leduc() {
    let target = 1e-4;
    let mut game = LeducGame::new(false);
    solve(&mut game, 10000, target, false);

    let root = game.root();

    let mut strategy = root.strategy().to_vec();
    for i in 0..NUM_PRIVATE_HANDS {
        let j = i + NUM_PRIVATE_HANDS;
        let sum = strategy[i] + strategy[j];
        strategy[i] /= sum;
        strategy[j] /= sum;
    }

    let root_ev = root
        .cfvalues()
        .iter()
        .zip(strategy.iter())
        .fold(0.0, |acc, (&ev, &strategy)| acc + ev * strategy);

    let expected_ev = -0.0856; // verified by OpenSpiel
    assert!((root_ev - expected_ev).abs() < 2.0 * target);
}

#[test]
fn leduc_compressed() {
    let target = 1e-3;
    let mut game = LeducGame::new(true);
    solve(&mut game, 10000, target, false);

    let root = game.root();

    let mut strategy = [0.0; NUM_PRIVATE_HANDS * 2];
    let raw_strategy = root.strategy_compressed();
    for i in 0..NUM_PRIVATE_HANDS {
        let j = i + NUM_PRIVATE_HANDS;
        let sum = (raw_strategy[i] as u32 + raw_strategy[j] as u32) as f32;
        strategy[i] = raw_strategy[i] as f32 / sum;
        strategy[j] = raw_strategy[j] as f32 / sum;
    }

    let ev_decoder = root.cfvalue_scale() / i16::MAX as f32;
    let root_ev = root
        .cfvalues_compressed()
        .iter()
        .zip(strategy.iter())
        .fold(0.0, |acc, (&raw_ev, &strategy)| {
            acc + ev_decoder * raw_ev as f32 * strategy
        });

    let expected_ev = -0.0856; // verified by OpenSpiel
    assert!((root_ev - expected_ev).abs() < 2.0 * target);
}
