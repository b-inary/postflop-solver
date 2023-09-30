use postflop_solver::*;

fn main() {
    normal_node_locking();
    partial_node_locking();
}

fn normal_node_locking() {
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

    // node locking must be performed after allocating memory and before solving
    game.play(1); // OOP all-in
    game.lock_current_strategy(&[0.25, 0.75]); // lock IP's strategy: 25% fold, 75% call
    game.back_to_root();

    solve(&mut game, 1000, 0.001, false);
    game.cache_normalized_weights();

    // check OOP's strategy
    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 1.0).abs() < 1e-3); // QQ always check
    assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA never check
    assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // QQ never all-in
    assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA always all-in

    game.allocate_memory(false);
    game.play(1);
    game.lock_current_strategy(&[0.5, 0.5]); // lock IP's strategy: 50% fold, 50% call
    game.back_to_root();

    solve(&mut game, 1000, 0.001, false);
    game.cache_normalized_weights();

    // check OOP's strategy
    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 0.0).abs() < 1e-3); // QQ never check
    assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA never check
    assert!((strategy_oop[2] - 1.0).abs() < 1e-3); // QQ always bet
    assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA always bet
}

fn partial_node_locking() {
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

    // lock OOP's strategy: only JJ is locked and the rest is not
    game.lock_current_strategy(&[0.8, 0.0, 0.0, 0.2, 0.0, 0.0]); // JJ: 80% check, 20% all-in

    solve(&mut game, 1000, 0.001, false);
    game.cache_normalized_weights();

    // check OOP's strategy
    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 0.8).abs() < 1e-3); // JJ check 80% (locked)
    assert!((strategy_oop[1] - 0.7).abs() < 1e-3); // QQ check 70%
    assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // AA never check
    assert!((strategy_oop[3] - 0.2).abs() < 1e-3); // JJ bet 20% (locked)
    assert!((strategy_oop[4] - 0.3).abs() < 1e-3); // QQ bet 30%
    assert!((strategy_oop[5] - 1.0).abs() < 1e-3); // AA always bet
}
