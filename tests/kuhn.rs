extern crate postflop_solver;
use postflop_solver::*;

struct KuhnGame {
    root: MutexLike<KuhnNode>,
    initial_reach: Vec<f32>,
}

struct KuhnNode {
    player: usize,
    amount: i32,
    children: Vec<(Action, MutexLike<KuhnNode>)>,
    cum_regret: Vec<f32>,
    strategy: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    fn initial_reach(&self, _player: usize) -> &[f32] {
        &self.initial_reach
    }

    fn evaluate(&self, result: &mut [f32], node: &Self::Node, player: usize, cfreach: &[f32]) {
        let num_hands = NUM_PRIVATE_HANDS * (NUM_PRIVATE_HANDS - 1);
        let num_hands_inv = 1.0 / num_hands as f32;

        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let payoff = node.amount * [1, -1][(player == folded_player) as usize];
            let payoff_normalized = payoff as f32 * num_hands_inv;
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
                        let payoff = node.amount * [1, -1][(my_card < opp_card) as usize];
                        let payoff_normalized = payoff as f32 * num_hands_inv;
                        result[my_card] += payoff_normalized * cfreach[opp_card];
                    }
                }
            }
        }
    }
}

impl KuhnGame {
    #[inline]
    pub fn new() -> Self {
        Self {
            root: Self::build_tree(),
            initial_reach: vec![1.0; NUM_PRIVATE_HANDS],
        }
    }

    fn build_tree() -> MutexLike<KuhnNode> {
        let mut root = KuhnNode {
            player: PLAYER_OOP,
            amount: 1,
            children: Vec::new(),
            cum_regret: Default::default(),
            strategy: Default::default(),
        };
        Self::build_tree_recursive(&mut root, Action::None);
        Self::allocate_memory_recursive(&mut root);
        MutexLike::new(root)
    }

    fn build_tree_recursive(node: &mut KuhnNode, last_action: Action) {
        if node.is_terminal() {
            return;
        }

        let actions = match last_action {
            Action::None | Action::Check => vec![Action::Check, Action::Bet],
            Action::Bet => vec![Action::Fold, Action::Call],
            _ => {
                println!("{:?}", last_action);
                unreachable!()
            }
        };

        for action in &actions {
            let next_player = match (action, last_action) {
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
                    cum_regret: Default::default(),
                    strategy: Default::default(),
                }),
            ));
        }

        for_each_child(node, |action| {
            Self::build_tree_recursive(&mut node.play(action), actions[action]);
        });
    }

    fn allocate_memory_recursive(node: &mut KuhnNode) {
        if node.is_terminal() {
            return;
        }

        let num_actions = node.num_actions();
        node.cum_regret = vec![0.0; num_actions * NUM_PRIVATE_HANDS];
        node.strategy = vec![0.0; num_actions * NUM_PRIVATE_HANDS];

        for_each_child(node, |action| {
            Self::allocate_memory_recursive(&mut node.play(action));
        });
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
    fn chance_factor(&self) -> f32 {
        unreachable!()
    }

    #[inline]
    fn isomorphic_chances(&self) -> &[IsomorphicChance] {
        unreachable!()
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<Self> {
        self.children[action].1.lock()
    }

    #[inline]
    fn cum_regret(&self) -> &[f32] {
        &self.cum_regret
    }

    #[inline]
    fn cum_regret_mut(&mut self) -> &mut [f32] {
        &mut self.cum_regret
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        &self.strategy
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        &mut self.strategy
    }
}

#[test]
fn kuhn() {
    let target = 1e-4;
    let game = KuhnGame::new();
    solve(&game, 10000, target, false);
    compute_ev(&game);

    let ev = compute_ev_scalar(&game, &game.root());
    let expected_ev = -1.0 / 18.0;
    assert!((ev - expected_ev).abs() <= 2.0 * target);
}
