extern crate postflop_solver;
use postflop_solver::*;

#[cfg(feature = "custom_alloc")]
use std::sync::atomic::Ordering;

struct LeducGame {
    root: MutexLike<LeducNode>,
    initial_weight: Vec<f32>,
    isomorphism: Vec<usize>,
    isomorphism_swap: [Vec<(usize, usize)>; 2],
    is_solved: bool,
}

struct LeducNode {
    player: usize,
    board: usize,
    amount: i32,
    children: Vec<(Action, MutexLike<LeducNode>)>,
    storage: Vec<f32>,
    strategy: Vec<f32>,
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
    fn initial_weight(&self, _player: usize) -> &[f32] {
        &self.initial_weight
    }

    fn evaluate(
        &self,
        result: &mut [f32],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
        compute_equity: bool,
    ) {
        let num_hands = NUM_PRIVATE_HANDS * (NUM_PRIVATE_HANDS - 1);
        let num_hands_inv = 1.0 / num_hands as f32;

        let amount = if compute_equity {
            0.5
        } else {
            node.amount as f32
        };
        let amount_normalized = amount * num_hands_inv;

        if !compute_equity && node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
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
        } else if node.board != NOT_DEALT {
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
        } else {
            for board in 0..NUM_PRIVATE_HANDS {
                for my_card in 0..NUM_PRIVATE_HANDS {
                    if my_card != board {
                        for opp_card in 0..NUM_PRIVATE_HANDS {
                            if my_card != opp_card && opp_card != board {
                                let sign = match () {
                                    _ if my_card / 2 == board / 2 => 1.0,
                                    _ if opp_card / 2 == board / 2 => -1.0,
                                    _ if my_card / 2 == opp_card / 2 => 0.0,
                                    _ if my_card > opp_card => 1.0,
                                    _ => -1.0,
                                };
                                let payoff_normalized = amount_normalized * sign / 4.0;
                                result[my_card] += payoff_normalized * cfreach[opp_card];
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn isomorphic_chances(&self, _node: &Self::Node) -> &[usize] {
        &self.isomorphism
    }

    #[inline]
    fn isomorphic_swap(&self, _node: &Self::Node, _index: usize) -> &[Vec<(usize, usize)>; 2] {
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
}

impl LeducGame {
    #[inline]
    pub fn new() -> Self {
        Self {
            root: Self::build_tree(),
            initial_weight: vec![1.0; NUM_PRIVATE_HANDS],
            isomorphism: vec![0, 1, 2],
            isomorphism_swap: [vec![(0, 1), (2, 3), (4, 5)], vec![(0, 1), (2, 3), (4, 5)]],
            is_solved: false,
        }
    }

    fn build_tree() -> MutexLike<LeducNode> {
        let mut root = LeducNode {
            player: PLAYER_OOP,
            board: NOT_DEALT,
            amount: 1,
            children: Vec::new(),
            storage: Default::default(),
            strategy: Default::default(),
        };
        Self::build_tree_recursive(&mut root, Action::None, [0, 0]);
        Self::allocate_memory_recursive(&mut root);
        #[cfg(feature = "custom_alloc")]
        STACK_UNIT_SIZE.store(1 << 20, Ordering::Relaxed);
        MutexLike::new(root)
    }

    fn build_tree_recursive(node: &mut LeducNode, last_action: Action, last_bet: [i32; 2]) {
        if node.is_terminal() {
            return;
        }

        if node.is_chance() {
            Self::push_chance_actions(node);
            for action in node.actions() {
                Self::build_tree_recursive(&mut node.play(action), Action::Chance(action), [0, 0]);
            }
            return;
        }

        let actions = Self::get_actions(node, last_action, node.board != NOT_DEALT);

        let mut last_bets = Vec::new();
        let prev_min_bet = last_bet.iter().min().unwrap();

        for (action, next_player) in &actions {
            let mut last_bet = last_bet;
            if *action == Action::Call {
                last_bet[node.player] = last_bet[node.player ^ 1];
            }
            if let Action::Bet(amount) = action {
                last_bet[node.player] = *amount;
            }
            if let Action::Raise(amount) = action {
                last_bet[node.player] = *amount;
            }
            last_bets.push(last_bet);

            let bet_diff = last_bet.iter().min().unwrap() - prev_min_bet;
            node.children.push((
                *action,
                MutexLike::new(LeducNode {
                    player: *next_player,
                    board: node.board,
                    amount: node.amount + bet_diff,
                    children: Vec::new(),
                    storage: Default::default(),
                    strategy: Default::default(),
                }),
            ));
        }

        for action in node.actions() {
            Self::build_tree_recursive(
                &mut node.play(action),
                actions[action].0,
                last_bets[action],
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
                    storage: Default::default(),
                    strategy: Default::default(),
                }),
            ));
        }
    }

    fn get_actions(
        node: &LeducNode,
        last_action: Action,
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

        match last_action {
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
            node.storage = vec![0.0; num_actions * NUM_PRIVATE_HANDS];
            node.strategy = vec![0.0; num_actions * NUM_PRIVATE_HANDS];
        }

        for action in node.actions() {
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
    fn chance_factor(&self) -> f32 {
        1.0 / 4.0
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<Self> {
        self.children[action].1.lock()
    }

    #[inline]
    fn cum_regret(&self) -> &[f32] {
        &self.storage
    }

    #[inline]
    fn cum_regret_mut(&mut self) -> &mut [f32] {
        &mut self.storage
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
    fn expected_values(&self) -> &[f32] {
        &self.storage[..NUM_PRIVATE_HANDS]
    }

    #[inline]
    fn expected_values_mut(&mut self) -> &mut [f32] {
        &mut self.storage[..NUM_PRIVATE_HANDS]
    }

    #[inline]
    fn equity(&self) -> &[f32] {
        &self.storage[NUM_PRIVATE_HANDS..2 * NUM_PRIVATE_HANDS]
    }

    #[inline]
    fn equity_mut(&mut self) -> &mut [f32] {
        &mut self.storage[NUM_PRIVATE_HANDS..2 * NUM_PRIVATE_HANDS]
    }
}

#[test]
fn leduc() {
    let target = 1e-4;
    let mut game = LeducGame::new();
    solve(&mut game, 10000, target, false);

    let root_ev = game.root().expected_values().iter().sum::<f32>();
    let root_equity = game.root().equity().iter().sum::<f32>();

    let expected_ev = -0.0856; // verified by OpenSpiel
    assert!((root_ev - expected_ev).abs() < 2.0 * target);
    assert!(root_equity.abs() < 1e-5);
}
