use super::*;
use crate::hand::{classify_hand, DrawType, HandCategory};
use crate::range::*;
use crate::solver::*;
use crate::utility::*;
use crate::BunchingData;

#[test]
fn all_check_all_range() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    assert!(game.is_chance_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(usize::MAX);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);
}

#[test]
fn one_raise_all_range() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    assert!(game.is_chance_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(usize::MAX);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(1);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 75.0).abs() < 1e-4);
    assert!((ev_ip - 15.0).abs() < 1e-4);

    game.play(1);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 60.0).abs() < 1e-4);
}

#[test]
fn one_raise_all_range_compressed() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(true);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(0);
    assert!(game.is_chance_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(usize::MAX);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(1);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 75.0).abs() < 1e-2);
    assert!((ev_ip - 15.0).abs() < 1e-2);

    game.play(1);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 60.0).abs() < 1e-2);
    assert!((ev_ip - 60.0).abs() < 1e-2);
}

#[test]
fn one_raise_all_range_with_turn() {
    let card_config = CardConfig {
        flop: flop_from_str("Td9d6h").unwrap(),
        range: [Range::ones(); 2],
        turn: card_from_str("Qc").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.5).abs() < 1e-5);
    assert!((root_equity_ip - 0.5).abs() < 1e-5);
    assert!((root_ev_oop - 37.5).abs() < 1e-4);
    assert!((root_ev_ip - 22.5).abs() < 1e-4);
}

#[test]
fn one_raise_all_range_with_river() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("Qc").unwrap(),
        river: card_from_str("7s").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.back_to_root();
    game.play(1);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 75.0).abs() < 1e-4);
    assert!((ev_ip - 15.0).abs() < 1e-4);

    game.play(0);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!(game.is_terminal_node());
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 90.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
fn always_win() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
fn always_win_raked() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((ev_oop - 57.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((ev_oop - 57.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
fn always_lose() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: [lose_range_str.parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.0).abs() < 1e-5);
    assert!((root_equity_ip - 1.0).abs() < 1e-5);
    assert!((root_ev_oop - 0.0).abs() < 1e-4);
    assert!((root_ev_ip - 60.0).abs() < 1e-4);
}

#[test]
fn always_lose_raked() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: [lose_range_str.parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_ev_oop - 0.0).abs() < 1e-4);
    assert!((root_ev_ip - 57.0).abs() < 1e-4);
}

#[test]
fn always_tie() {
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.5).abs() < 1e-5);
    assert!((root_equity_ip - 0.5).abs() < 1e-5);
    assert!((root_ev_oop - 30.0).abs() < 1e-4);
    assert!((root_ev_ip - 30.0).abs() < 1e-4);
}

#[test]
fn always_tie_raked() {
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_ev_oop - 28.5).abs() < 1e-4);
    assert!((root_ev_ip - 28.5).abs() < 1e-4);
}

#[test]
fn no_assignment() {
    let card_config = CardConfig {
        range: ["TT".parse().unwrap(), "TT".parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let game = PostFlopGame::with_config(card_config, action_tree);
    assert!(game.is_err());
}

#[test]
fn remove_lines() {
    use crate::bet_size::BetSizeOptions;
    let card_config = CardConfig {
        range: ["TT+,AKo,AQs+".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    // simple tree: force checks on flop, and only use 1/2 pot bets on turn and river
    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        turn_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            Default::default(),
        ],
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            Default::default(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let lines = vec![
        vec![
            Action::Check,
            Action::Check,
            Action::Chance(2),
            Action::Check,
        ],
        vec![
            Action::Check,
            Action::Check,
            Action::Chance(2),
            Action::Bet(30),
            Action::Call,
            Action::Chance(3),
            Action::Bet(60),
        ],
    ];

    let res = game.remove_lines(&lines);
    assert!(res.is_ok());

    game.allocate_memory(false);

    // check that the turn line is removed
    game.apply_history(&[0, 0, 2]);
    assert_eq!(game.available_actions(), vec![Action::Bet(30)]);

    // check that other turn lines are correct
    game.apply_history(&[0, 0, 3]);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(30)]
    );

    // check that the river line is removed
    game.apply_history(&[0, 0, 2, 0, 1, 3]);
    assert_eq!(game.available_actions(), vec![Action::Check]);

    // check that other river lines are correct
    game.apply_history(&[0, 0, 2, 0, 1, 4]);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(60)]
    );

    game.apply_history(&[0, 0, 3, 1, 1, 4]);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(60)]
    );

    // check that `solve()` does not crash
    solve(&mut game, 10, 0.01, false);
}

#[test]
fn isomorphism_monotone() {
    let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QhJh2h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 100,
        effective_stack: 100,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let mut check = |history: &[usize],
                     expected_turn_swap: Option<u8>,
                     expected_river_swap: Option<(u8, u8)>| {
        game.apply_history(history);
        game.cache_normalized_weights();
        let weights = game.normalized_weights(0);
        let ev = game.expected_values(0);
        weights.iter().zip(ev.iter()).for_each(|(&w, &v)| {
            assert!(!(w > 0.0 && v == 50.0));
        });
        assert_eq!(game.turn_swap, expected_turn_swap);
        assert_eq!(game.river_swap, expected_river_swap);
    };

    check(&[0, 0, 4], None, None);
    check(&[0, 0, 5], Some(1), None);
    check(&[0, 0, 6], None, None);
    check(&[0, 0, 7], Some(3), None);

    check(&[0, 0, 4, 0, 0, 8], None, None);
    check(&[0, 0, 4, 0, 0, 9], None, None);
    check(&[0, 0, 4, 0, 0, 10], None, None);
    check(&[0, 0, 4, 0, 0, 11], None, Some((0, 3)));

    check(&[0, 0, 5, 0, 0, 8], Some(1), None);
    check(&[0, 0, 5, 0, 0, 9], Some(1), None);
    check(&[0, 0, 5, 0, 0, 10], Some(1), None);
    check(&[0, 0, 5, 0, 0, 11], Some(1), Some((1, 3)));

    check(&[0, 0, 6, 0, 0, 8], None, None);
    check(&[0, 0, 6, 0, 0, 9], None, Some((2, 1)));
    check(&[0, 0, 6, 0, 0, 10], None, None);
    check(&[0, 0, 6, 0, 0, 11], None, Some((2, 3)));

    check(&[0, 0, 7, 0, 0, 8], Some(3), Some((3, 1)));
    check(&[0, 0, 7, 0, 0, 9], Some(3), None);
    check(&[0, 0, 7, 0, 0, 10], Some(3), None);
    check(&[0, 0, 7, 0, 0, 11], Some(3), None);
}

#[test]
fn node_locking() {
    let card_config = CardConfig {
        range: ["AsAh,QsQh".parse().unwrap(), "KsKh".parse().unwrap()],
        flop: flop_from_str("2s3h4d").unwrap(),
        turn: card_from_str("6c").unwrap(),
        river: card_from_str("7c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 20,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.play(1); // all-in
    game.lock_current_strategy(&[0.25, 0.75]); // 25% fold, 75% call
    game.back_to_root();

    solve(&mut game, 1000, 0.0, false);
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    assert!((ev_oop[0] - 0.0).abs() < 1e-2);
    assert!((ev_oop[1] - 27.5).abs() < 5e-2);
    assert!((ev_ip[0] - 6.25).abs() < 1e-2);

    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 1.0).abs() < 1e-3); // QQ check
    assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA check
    assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // QQ bet
    assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA bet

    game.allocate_memory(false);
    game.play(1); // all-in
    game.lock_current_strategy(&[0.5, 0.5]); // 50% fold, 50% call
    game.back_to_root();

    solve(&mut game, 1000, 0.0, false);
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    assert!((ev_oop[0] - 5.0).abs() < 1e-2);
    assert!((ev_oop[1] - 25.0).abs() < 5e-2);
    assert!((ev_ip[0] - 5.0).abs() < 1e-2);

    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 0.0).abs() < 1e-3); // QQ check
    assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA check
    assert!((strategy_oop[2] - 1.0).abs() < 1e-3); // QQ bet
    assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA bet
}

#[test]
fn node_locking_partial() {
    let card_config = CardConfig {
        range: ["AsAh,QsQh,JsJh".parse().unwrap(), "KsKh".parse().unwrap()],
        flop: flop_from_str("2s3h4d").unwrap(),
        turn: card_from_str("6c").unwrap(),
        river: card_from_str("7c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 10,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.lock_current_strategy(&[0.8, 0.0, 0.0, 0.2, 0.0, 0.0]); // JJ -> 80% check, 20% all-in

    solve(&mut game, 1000, 0.0, false);
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    assert!((ev_oop[0] - 0.0).abs() < 1e-2);
    assert!((ev_oop[1] - 0.0).abs() < 1e-2);
    assert!((ev_oop[2] - 15.0).abs() < 5e-2);
    assert!((ev_ip[0] - 5.0).abs() < 1e-2);

    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 0.8).abs() < 1e-3); // JJ check
    assert!((strategy_oop[1] - 0.7).abs() < 1e-3); // QQ check
    assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // AA check
    assert!((strategy_oop[3] - 0.2).abs() < 1e-3); // JJ bet
    assert!((strategy_oop[4] - 0.3).abs() < 1e-3); // QQ bet
    assert!((strategy_oop[5] - 1.0).abs() < 1e-3); // AA bet
}

#[test]
fn node_locking_isomorphism() {
    let card_config = CardConfig {
        range: ["AKs".parse().unwrap(), "AKs".parse().unwrap()],
        flop: flop_from_str("2c3c4c").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 10,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.apply_history(&[0, 0, 15, 0, 0, 14]); // Turn: Spades, River: Hearts
    game.lock_current_strategy(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // AhKh -> check

    finalize(&mut game);

    game.apply_history(&[0, 0, 13, 0, 0, 14]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5]
    );

    game.apply_history(&[0, 0, 13, 0, 0, 15]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]
    );

    game.apply_history(&[0, 0, 14, 0, 0, 13]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
    );

    game.apply_history(&[0, 0, 14, 0, 0, 15]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]
    );

    game.apply_history(&[0, 0, 15, 0, 0, 13]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
    );

    game.apply_history(&[0, 0, 15, 0, 0, 14]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5]
    );
}

#[test]
fn set_bunching_effect() {
    let flop = flop_from_str("Td9d6h").unwrap();
    let card_config = CardConfig {
        flop,
        range: [Range::ones(); 2],
        turn: card_from_str("Qc").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let co_range = "33:0.59,22:0.635,A8o:0.265,A7o-A6o,A5o:0.445,A4o-A2o,K2s,K9o:0.905,K8o-K2o,Q4s-Q2s,Q9o-Q2o,J6s-J2s,J9o:0.88,J8o-J2o,T7s:0.405,T6s-T2s,T9o:0.96,T8o-T2o,96s-92s,92o+,86s:0.57,85s-82s,82o+,76s:0.37,75s-72s,72o+,65s:0.475,64s-62s,62o+,54s:0.68,53s-52s,52o+,42+,32";
    let sb_range = "66:0.46,55:0.821,44:0.92,33:0.93,22:0.925,A6s:0.73,A3s:0.47,A2s,ATo:0.105,A9o-A2o,K8s:0.795,K7s,K6s:0.85,K5s:0.965,K4s-K2s,KJo:0.085,KTo:0.645,K9o-K2o,Q8s-Q2s,QJo:0.765,QTo-Q2o,J8s-J2s,J2o+,T8s:0.69,T7s-T2s,T2o+,98s:0.905,97s-92s,92o+,87s:0.78,86s-82s,82o+,76s:0.77,75s-72s,72o+,65s:0.845,64s-62s,62o+,54s:0.735,53s-52s,52o+,42+,32";

    let mut bunching_data = BunchingData::new(
        &[co_range.parse().unwrap(), sb_range.parse().unwrap()],
        flop,
    )
    .unwrap();

    bunching_data.process(false);
    game.set_bunching_effect(&bunching_data).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let current_ev = compute_current_ev(&game);
    assert!((current_ev[0] - 7.5).abs() < 1e-4);
    assert!((current_ev[1] - -7.5).abs() < 1e-4);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.5).abs() < 1e-5);
    assert!((root_equity_ip - 0.5).abs() < 1e-5);
    assert!((root_ev_oop - 37.5).abs() < 1e-4);
    assert!((root_ev_ip - 22.5).abs() < 1e-4);
}

#[test]
fn set_bunching_effect_always_win() {
    let flop = flop_from_str("AcAdKh").unwrap();
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";

    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop,
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let co_range = "33:0.59,22:0.635,A8o:0.265,A7o-A6o,A5o:0.445,A4o-A2o,K2s,K9o:0.905,K8o-K2o,Q4s-Q2s,Q9o-Q2o,J6s-J2s,J9o:0.88,J8o-J2o,T7s:0.405,T6s-T2s,T9o:0.96,T8o-T2o,96s-92s,92o+,86s:0.57,85s-82s,82o+,76s:0.37,75s-72s,72o+,65s:0.475,64s-62s,62o+,54s:0.68,53s-52s,52o+,42+,32";
    let sb_range = "66:0.46,55:0.821,44:0.92,33:0.93,22:0.925,A6s:0.73,A3s:0.47,A2s,ATo:0.105,A9o-A2o,K8s:0.795,K7s,K6s:0.85,K5s:0.965,K4s-K2s,KJo:0.085,KTo:0.645,K9o-K2o,Q8s-Q2s,QJo:0.765,QTo-Q2o,J8s-J2s,J2o+,T8s:0.69,T7s-T2s,T2o+,98s:0.905,97s-92s,92o+,87s:0.78,86s-82s,82o+,76s:0.77,75s-72s,72o+,65s:0.845,64s-62s,62o+,54s:0.735,53s-52s,52o+,42+,32";

    let mut bunching_data = BunchingData::new(
        &[co_range.parse().unwrap(), sb_range.parse().unwrap()],
        flop,
    )
    .unwrap();

    bunching_data.process(false);
    game.set_bunching_effect(&bunching_data).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let current_ev = compute_current_ev(&game);
    assert!((current_ev[0] - 30.0).abs() < 1e-4);
    assert!((current_ev[1] - -30.0).abs() < 1e-4);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
#[ignore]
fn solve_pio_preset_normal() {
    let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QsJh2h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 180,
        effective_stack: 910,
        flop_bet_sizes: [
            ("52%", "45%").try_into().unwrap(),
            ("52%", "45%").try_into().unwrap(),
        ],
        turn_bet_sizes: [
            ("55%", "45%").try_into().unwrap(),
            ("55%", "45%").try_into().unwrap(),
        ],
        river_bet_sizes: [
            ("70%", "45%").try_into().unwrap(),
            ("70%", "45%").try_into().unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    println!(
        "memory usage: {:.2}GB",
        game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    game.allocate_memory(false);

    solve(&mut game, 1000, 180.0 * 0.001, true);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    // verified by PioSOLVER Free
    assert!((root_equity_oop - 0.55347).abs() < 1e-5);
    assert!((root_equity_ip - 0.44653).abs() < 1e-5);
    assert!((root_ev_oop - 105.11).abs() < 0.2);
    assert!((root_ev_ip - 74.89).abs() < 0.2);
}

#[test]
#[ignore]
fn solve_pio_preset_raked() {
    let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QsJh2h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 180,
        effective_stack: 910,
        rake_rate: 0.05,
        rake_cap: 30.0,
        flop_bet_sizes: [
            ("52%", "45%").try_into().unwrap(),
            ("52%", "45%").try_into().unwrap(),
        ],
        turn_bet_sizes: [
            ("55%", "45%").try_into().unwrap(),
            ("55%", "45%").try_into().unwrap(),
        ],
        river_bet_sizes: [
            ("70%", "45%").try_into().unwrap(),
            ("70%", "45%").try_into().unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    println!(
        "memory usage: {:.2}GB",
        game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    game.allocate_memory(false);

    solve(&mut game, 1000, 180.0 * 0.001, true);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    // verified by PioSOLVER Free (but not theoretically guaranteed to be the same)
    assert!((root_ev_oop - 95.57).abs() < 0.2);
    assert!((root_ev_ip - 66.98).abs() < 0.2);
}

#[test]
fn depth_limit_flop_all_check() {
    // 测试深度限制求解：depth_limit = Flop
    // 使用较小的 range 以加速权益矩阵计算
    let range_str = "TT+,AK";
    let card_config = CardConfig {
        range: [range_str.parse().unwrap(), range_str.parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        depth_limit: Some(BoardState::Flop),
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);

    // 深度限制树应该非常小
    let (uncompressed, _) = game.memory_usage();
    assert!(
        uncompressed < 1024 * 1024,
        "Depth-limited tree should be small, got {} bytes",
        uncompressed
    );

    // 求解并验证收敛
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);

    // 对称 range + 无 rake，EV 应接近 half-pot = 30
    // TT+ 在 Td9d6h 上有 equity 差异，容忍较大误差
    assert!(
        (ev_oop - ev_ip).abs() < 10.0,
        "OOP and IP EV should be similar: OOP={}, IP={}",
        ev_oop,
        ev_ip
    );
}

#[test]
fn depth_limit_turn() {
    // 测试 depth_limit = Turn
    let range_str = "TT+,AK";
    let card_config = CardConfig {
        range: [range_str.parse().unwrap(), range_str.parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("Qc").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 970,
        depth_limit: Some(BoardState::Turn),
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);

    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);

    // 对称 range + 无 rake
    assert!(
        (ev_oop - ev_ip).abs() < 10.0,
        "OOP and IP EV should be similar: OOP={}, IP={}",
        ev_oop,
        ev_ip
    );
}

#[test]
fn classify_monster() {
    let flop = flop_from_str("Td9d6h").unwrap();

    // 暗三条: 66 on Td9d6h → trips → Monster
    let cards = (card_from_str("6c").unwrap(), card_from_str("6s").unwrap());
    let (cat, _) = classify_hand(cards, flop, NOT_DEALT);
    assert_eq!(cat, HandCategory::Monster);

    // 两对: T9 on Td9d6h → two pair → Monster
    let cards2 = (card_from_str("Tc").unwrap(), card_from_str("9c").unwrap());
    let (cat2, _) = classify_hand(cards2, flop, NOT_DEALT);
    assert_eq!(cat2, HandCategory::Monster);

    // 同花: 5 diamonds → flush → Monster
    let flop_d = flop_from_str("2d3d4d").unwrap();
    let cards_d = (card_from_str("Td").unwrap(), card_from_str("Ad").unwrap());
    let (cat3, _) = classify_hand(cards_d, flop_d, NOT_DEALT);
    assert_eq!(cat3, HandCategory::Monster);
}

#[test]
fn classify_overpair() {
    let flop = flop_from_str("8d7h3c").unwrap();
    // AA → overpair (rank 12 > max board rank 6)
    let cards = (card_from_str("Ac").unwrap(), card_from_str("As").unwrap());
    let (cat, draw) = classify_hand(cards, flop, NOT_DEALT);
    assert_eq!(cat, HandCategory::Overpair);
    assert_eq!(draw, DrawType::None);
}

#[test]
fn classify_top_pair() {
    let flop = flop_from_str("Kh9d4c").unwrap(); // ranks: 11, 7, 2

    // KQ → top pair K, kicker Q (rank 10) >= median (7) → TPTK
    let cards = (card_from_str("Kc").unwrap(), card_from_str("Qd").unwrap());
    let (cat, _) = classify_hand(cards, flop, NOT_DEALT);
    assert_eq!(cat, HandCategory::TopPairGoodKicker);

    // K5 → top pair K, kicker 5 (rank 3) < median (7) → TPWK
    let cards2 = (card_from_str("Kc").unwrap(), card_from_str("5d").unwrap());
    let (cat2, _) = classify_hand(cards2, flop, NOT_DEALT);
    assert_eq!(cat2, HandCategory::TopPairBadKicker);
}

#[test]
fn classify_draws() {
    // 同花听牌: AhJh on Kh9h4c → 4 hearts
    let flop = flop_from_str("Kh9h4c").unwrap();
    let cards = (card_from_str("Ah").unwrap(), card_from_str("Jh").unwrap());
    let (cat, draw) = classify_hand(cards, flop, NOT_DEALT);
    // Ah=rank12, Jh=rank9. max_board=K(rank11). 12>11 but 9<11 → not TwoOvercards → Air
    assert_eq!(cat, HandCategory::Air);
    assert_eq!(draw, DrawType::FlushDraw);

    // OESD: JT on 9d8h3c → ranks {1,6,7,8,9}
    // window [5..9]={5,6,7,8,9}: has 6,7,8,9=4, missing 5 (endpoint) → OESD
    let flop2 = flop_from_str("9d8h3c").unwrap();
    let cards2 = (card_from_str("Jc").unwrap(), card_from_str("Tc").unwrap());
    let (cat2, draw2) = classify_hand(cards2, flop2, NOT_DEALT);
    // J=rank9, T=rank8. max_board=9d(rank7). 9>7 and 8>7 → TwoOvercards
    // Actually: 9d=rank7, 8h=rank6, 3c=rank1. max_board=7. J=9>7, T=8>7 → TwoOvercards
    assert_eq!(cat2, HandCategory::TwoOvercards);
    assert_eq!(draw2, DrawType::OESD);
}

#[test]
fn classify_gutshot() {
    // Gutshot: QT on 5c3d2h → ranks: Q(10),T(8),5(3),3(1),2(0) = {0,1,3,8,10}
    // window [8..12]={8,9,10,11,12}: has 8,10 = 2. No.
    // window [7..11]={7,8,9,10,11}: has 8,10 = 2. No.
    // Hmm. Let me try QT on 5c4d2h: ranks {0,2,3,8,10}
    // window [8..12]={8,9,10,11,12}: 8,10=2.
    // Actually need 4 in a 5-window. Try AQ on Jc5d3h:
    // A(12),Q(10),J(9),5(3),3(1) = {1,3,9,10,12}
    // window [8..12]={8,9,10,11,12}: 9,10,12=3. No.
    // Try KQ on Jc5d3h: K(11),Q(10),J(9),5(3),3(1) = {1,3,9,10,11}
    // window [8..12]={8,9,10,11,12}: 9,10,11=3.
    // window [7..11]={7,8,9,10,11}: 9,10,11=3.
    // Still 3. Need exactly 4 in a window.
    // KJ on QT5: K(11),J(9),Q(10),T(8),5(3) = {3,8,9,10,11}
    // window [7..11]={7,8,9,10,11}: 8,9,10,11=4! missing 7 (endpoint) → OESD
    // That's OESD, not gutshot.
    // For gutshot: need internal gap. AT on KJ2:
    // A(12),T(8),K(11),J(9),2(0) = {0,8,9,11,12}
    // window [8..12]={8,9,10,11,12}: 8,9,11,12=4! missing 10 (internal) → Gutshot!
    let flop = flop_from_str("KdJh2c").unwrap();
    let cards = (card_from_str("Ac").unwrap(), card_from_str("Tc").unwrap());
    let (cat, draw) = classify_hand(cards, flop, NOT_DEALT);
    // A=12, T=8. max_board=K(11). 12>11 but 8<11 → not TwoOvercards → Air
    assert_eq!(cat, HandCategory::Air);
    assert_eq!(draw, DrawType::Gutshot);
}

#[test]
fn classify_combo_draw() {
    // 超对: TcTh on 9h8h3c → pocket TT, pair_rank=8 > max_board=7 → Overpair, no draw
    let flop = flop_from_str("9h8h3c").unwrap();
    let cards = (card_from_str("Tc").unwrap(), card_from_str("Th").unwrap());
    let (cat, draw) = classify_hand(cards, flop, NOT_DEALT);
    assert_eq!(cat, HandCategory::Overpair);
    assert_eq!(draw, DrawType::None);

    // 顶对弱踢脚 + 同花听: Kc9c on Kd8c3c → pairs K, 4 clubs → TPWK + FlushDraw
    // K(11)=max_board. Kicker=9(rank7). Board ranks: K(11),8(6),3(1). median=6. 7>=6 → TPTK actually
    // Let me recalculate: sorted_board_ranks = [1,6,11], median = sorted[3/2] = sorted[1] = 6
    // kicker=7 >= 6 → TPTK. Hmm.
    // Use Kc6c on Kd8c3c: kicker=6(rank4). median=6(rank of 8). 4<6 → TPWK
    let flop2 = flop_from_str("Kd8c3c").unwrap();
    let cards2 = (card_from_str("Kc").unwrap(), card_from_str("6c").unwrap());
    let (cat2, draw2) = classify_hand(cards2, flop2, NOT_DEALT);
    // Clubs: Kc, 6c, 8c, 3c = 4 clubs → FlushDraw
    assert_eq!(cat2, HandCategory::TopPairBadKicker);
    assert_eq!(draw2, DrawType::FlushDraw);
}

#[test]
fn classify_two_overcards() {
    let flop = flop_from_str("8d7h3c").unwrap(); // max board rank = 6 (8)
                                                 // AK → both ranks (12, 11) > max board rank (6) → TwoOvercards
    let cards = (card_from_str("Ac").unwrap(), card_from_str("Kc").unwrap());
    let (cat, draw) = classify_hand(cards, flop, NOT_DEALT);
    assert_eq!(cat, HandCategory::TwoOvercards);
    assert_eq!(draw, DrawType::None);
}

#[test]
fn classify_medium_bottom_pair() {
    let flop = flop_from_str("Kh9d4c").unwrap(); // ranks: 11, 7, 2

    // 99 on K94 → pocket 9s, one 9 also on board → num_pairs=2 (9 and board has no other pair)
    // Wait: hero has 9c,9s. Board has 9d. rank_count[7]=3 → num_trips=1 → Monster!
    // That's correct: hero + board = three 9s = trips.
    let cards = (card_from_str("9c").unwrap(), card_from_str("9s").unwrap());
    let (cat, _) = classify_hand(cards, flop, NOT_DEALT);
    assert_eq!(cat, HandCategory::Monster); // trips (99 + board 9)

    // Use a different board without matching hero ranks
    let flop2 = flop_from_str("Kh7d3c").unwrap(); // ranks: 11, 5, 1
                                                  // 99 → pocket pair, no board match → pair_rank=7, 7>1 and 7<11 → MediumPair
    let cards2 = (card_from_str("9c").unwrap(), card_from_str("9s").unwrap());
    let (cat2, _) = classify_hand(cards2, flop2, NOT_DEALT);
    assert_eq!(cat2, HandCategory::MediumPair);

    // 44 → pocket pair, pair_rank=2, 2>1(min) and 2<11(max) → MediumPair
    let cards3 = (card_from_str("4s").unwrap(), card_from_str("4h").unwrap());
    let (cat3, _) = classify_hand(cards3, flop2, NOT_DEALT);
    assert_eq!(cat3, HandCategory::MediumPair);

    // For BottomPair: use a board where hero pairs the lowest card
    let flop3 = flop_from_str("Kh9d4c").unwrap(); // ranks: 11, 7, 2
                                                  // 55 → pair_rank=3, 3>2(min) and 3<11(max) → MediumPair
    let cards4 = (card_from_str("5c").unwrap(), card_from_str("5h").unwrap());
    let (cat4, _) = classify_hand(cards4, flop3, NOT_DEALT);
    assert_eq!(cat4, HandCategory::MediumPair);

    // 22 → pair_rank=0, 0<2(min_board) → Overpair? No, 0 < 2 → BottomPair
    // Actually: 0 < min_board(2). In classify_pair: pair_rank(0) < min_board_rank(2) → BottomPair
    // Wait, pair_rank(0) > min_board_rank(2) is false. pair_rank(0) == min_board_rank(2) is false.
    // pair_rank(0) > max_board_rank(11) → no. pair_rank == max → no. pair_rank > min → 0>2? no.
    // So falls to else → BottomPair. But 22 is actually BELOW the bottom pair of the board.
    // This is correct classification: 22 on K94 is a bottom pair (below everything).
    let cards5 = (card_from_str("2s").unwrap(), card_from_str("2h").unwrap());
    let (cat5, _) = classify_hand(cards5, flop3, NOT_DEALT);
    assert_eq!(cat5, HandCategory::BottomPair);
}

#[test]
fn depth_limit_with_category_correction() {
    let range_str = "TT+,AK";
    let card_config = CardConfig {
        range: [range_str.parse().unwrap(), range_str.parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let mut flop_coefs = CategoryCoefficients::default();
    flop_coefs.monster = 1.15;
    flop_coefs.air = 0.75;

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        depth_limit: Some(BoardState::Flop),
        flop_category_correction: flop_coefs,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);

    assert!(!game.hand_categories[0].is_empty());
    assert!(!game.hand_categories[1].is_empty());

    finalize(&mut game);

    game.cache_normalized_weights();
    let ev_oop = compute_average(&game.expected_values(0), game.normalized_weights(0));
    let ev_ip = compute_average(&game.expected_values(1), game.normalized_weights(1));

    assert!(
        (ev_oop - ev_ip).abs() < 15.0,
        "OOP and IP EV should be similar: OOP={}, IP={}",
        ev_oop,
        ev_ip
    );
}
