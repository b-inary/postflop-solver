extern crate postflop_solver;
use postflop_solver::*;
use std::mem::MaybeUninit;

struct KuhnGame {
    root: MutexLike<KuhnNode>,
    initial_weight: Vec<f32>,
    is_solved: bool,
}

struct KuhnNode {
    player: usize,
    amount: i32,
    children: Vec<(Action, MutexLike<KuhnNode>)>,
    strategy: Vec<f32>,
    storage: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    None,
    Fold,
    Check,
    Call,
    Bet,
}

const NUM_PRIVATE_HANDS: usize = 3;

#[allow(dead_code)]
const PLAYER_OOP: usize = 0;

#[allow(dead_code)]
const PLAYER_IP: usize = 1;

const PLAYER_MASK: usize = 0xff;
const PLAYER_TERMINAL_FLAG: usize = 0x100;
const PLAYER_FOLD_FLAG: usize = 0x300;

impl Game for KuhnGame {
    type Node = KuhnNode;

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
                for opp_card in 0..NUM_PRIVATE_HANDS {
                    if my_card != opp_card {
                        result[my_card] += payoff_normalized * cfreach[opp_card];
                    }
                }
            }
        } else {
            for my_card in 0..NUM_PRIVATE_HANDS {
                for opp_card in 0..NUM_PRIVATE_HANDS {
                    if my_card != opp_card {
                        let sign = [1.0, -1.0][(my_card < opp_card) as usize];
                        let payoff_normalized = amount_normalized * sign;
                        result[my_card] += payoff_normalized * cfreach[opp_card];
                    }
                }
            }
        }
    }

    #[inline]
    fn chance_factor(&self, _node: &Self::Node) -> usize {
        unreachable!()
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

impl KuhnGame {
    #[inline]
    pub fn new() -> Self {
        Self {
            root: Self::build_tree(),
            initial_weight: vec![1.0; NUM_PRIVATE_HANDS],
            is_solved: false,
        }
    }

    fn build_tree() -> MutexLike<KuhnNode> {
        let mut root = KuhnNode {
            player: PLAYER_OOP,
            amount: 1,
            children: Vec::new(),
            strategy: Default::default(),
            storage: Default::default(),
        };
        Self::build_tree_recursive(&mut root, Action::None);
        Self::allocate_memory_recursive(&mut root);
        MutexLike::new(root)
    }

    fn build_tree_recursive(node: &mut KuhnNode, prev_action: Action) {
        if node.is_terminal() {
            return;
        }

        let actions = match prev_action {
            Action::None | Action::Check => vec![Action::Check, Action::Bet],
            Action::Bet => vec![Action::Fold, Action::Call],
            _ => unreachable!(),
        };

        for action in &actions {
            let next_player = match (action, prev_action) {
                (Action::Check, Action::Check) => PLAYER_TERMINAL_FLAG,
                (Action::Fold, _) => PLAYER_FOLD_FLAG | node.player,
                (Action::Call, _) => PLAYER_TERMINAL_FLAG,
                _ => node.player ^ 1,
            };
            node.children.push((
                *action,
                MutexLike::new(KuhnNode {
                    player: next_player,
                    amount: node.amount + (*action == Action::Call) as i32,
                    children: Vec::new(),
                    strategy: Default::default(),
                    storage: Default::default(),
                }),
            ));
        }

        for (action, child) in &node.children {
            Self::build_tree_recursive(&mut child.lock(), *action);
        }
    }

    fn allocate_memory_recursive(node: &mut KuhnNode) {
        if node.is_terminal() {
            return;
        }

        let num_actions = node.num_actions();
        node.strategy = vec![0.0; num_actions * NUM_PRIVATE_HANDS];
        node.storage = vec![0.0; num_actions * NUM_PRIVATE_HANDS];

        for action in node.action_indices() {
            Self::allocate_memory_recursive(&mut node.play(action));
        }
    }
}

impl GameNode for KuhnNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    #[inline]
    fn is_chance(&self) -> bool {
        false
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
}

#[test]
fn kuhn() {
    let target = 1e-4;
    let mut game = KuhnGame::new();
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
        .fold(0.0, |acc, (&cfv, &strategy)| acc + cfv * strategy);

    let expected_ev = -1.0 / 18.0;
    assert!((root_ev - expected_ev).abs() < 2.0 * target);
}
