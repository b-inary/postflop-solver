use super::*;
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
