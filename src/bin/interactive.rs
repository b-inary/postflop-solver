use postflop_solver::*;
use std::io::{self, Write};

const DEFAULT_OOP_RANGE: &str = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
const DEFAULT_IP_RANGE: &str = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";
const DEFAULT_BET_SIZES: &str = "60%, e, a";
const DEFAULT_RAISE_SIZES: &str = "2.5x";
const DEFAULT_STARTING_POT: i32 = 200;
const DEFAULT_EFFECTIVE_STACK: i32 = 900;

fn main() {
    println!("=== Postflop Solver Interactive Mode ===\n");

    let (hero, hero_c1, hero_c2, card_config, tree_config) = prompt_config();

    println!("\nBuilding game tree and solving...");
    let mut game = build_and_solve(&tree_config, &card_config);

    let hero_hand_index = find_hero_hand_index(&game, hero, hero_c1, hero_c2);
    let hero_str = holes_to_strings(&[(hero_c1, hero_c2)]).unwrap();
    println!(
        "\nHero's hand: {} (index {} in {} private hands)",
        hero_str[0],
        hero_hand_index,
        game.private_cards(hero).len()
    );

    run_interactive_loop(&mut game, hero, hero_hand_index);
}

fn prompt_config() -> (usize, Card, Card, CardConfig, TreeConfig) {
    // 1. Hero position
    let hero = prompt_input("Hero position (oop/ip):", |s| match s.to_lowercase().as_str() {
        "oop" | "0" => Ok(0usize),
        "ip" | "1" => Ok(1usize),
        _ => Err("Enter 'oop' or 'ip'".into()),
    });
    println!(
        "  Hero is {}\n",
        if hero == 0 { "OOP" } else { "IP" }
    );

    // 2. Ranges
    let hero_range_str = prompt_with_default(
        &format!(
            "Hero ({}) range (Enter for default):",
            if hero == 0 { "OOP" } else { "IP" }
        ),
        if hero == 0 {
            DEFAULT_OOP_RANGE
        } else {
            DEFAULT_IP_RANGE
        },
    );
    let opp_range_str = prompt_with_default(
        &format!(
            "Opponent ({}) range (Enter for default):",
            if hero == 0 { "IP" } else { "OOP" }
        ),
        if hero == 0 {
            DEFAULT_IP_RANGE
        } else {
            DEFAULT_OOP_RANGE
        },
    );

    let oop_range: Range = if hero == 0 {
        hero_range_str.parse().unwrap()
    } else {
        opp_range_str.parse().unwrap()
    };
    let ip_range: Range = if hero == 0 {
        opp_range_str.parse().unwrap()
    } else {
        hero_range_str.parse().unwrap()
    };

    // 3. Hero's hole cards
    let (hero_c1, hero_c2) = prompt_input("Hero's hole cards (e.g. AhKd):", |s| {
        if s.len() != 4 {
            return Err("Expected 4 characters, e.g. AhKd".into());
        }
        let c1 = card_from_str(&s[0..2])?;
        let c2 = card_from_str(&s[2..4])?;
        if c1 == c2 {
            return Err("Cards must be different".into());
        }
        let hero_range = if hero == 0 { &oop_range } else { &ip_range };
        if hero_range.get_weight_by_cards(c1, c2) == 0.0 {
            return Err("Hand is not in hero's range".into());
        }
        Ok((c1, c2))
    });
    let hero_hand_str = holes_to_strings(&[(hero_c1, hero_c2)]).unwrap();
    println!("  Parsed: {}\n", hero_hand_str[0]);

    // 4. Flop
    let flop = prompt_input("Flop (e.g. Td9d6h):", flop_from_str);
    println!(
        "  Flop: {} {} {}\n",
        card_to_string(flop[0]).unwrap(),
        card_to_string(flop[1]).unwrap(),
        card_to_string(flop[2]).unwrap()
    );

    // 5. Pot and stack
    let starting_pot = prompt_with_default_parse(
        &format!("Starting pot (Enter for {}):", DEFAULT_STARTING_POT),
        DEFAULT_STARTING_POT,
        |s| s.parse::<i32>().map_err(|e| e.to_string()),
    );

    let effective_stack = prompt_with_default_parse(
        &format!(
            "Effective stack (Enter for {}):",
            DEFAULT_EFFECTIVE_STACK
        ),
        DEFAULT_EFFECTIVE_STACK,
        |s| s.parse::<i32>().map_err(|e| e.to_string()),
    );
    println!();

    // 6. Bet sizes
    let bet_sizes = prompt_input(
        &format!(
            "Bet sizes (Enter for default \"{}\" / \"{}\"):",
            DEFAULT_BET_SIZES, DEFAULT_RAISE_SIZES
        ),
        |s| {
            if s.is_empty() {
                return BetSizeOptions::try_from((DEFAULT_BET_SIZES, DEFAULT_RAISE_SIZES))
                    .map_err(|e| e.to_string());
            }
            // Accept "bet_sizes, raise_sizes" format
            let parts: Vec<&str> = s.splitn(2, '/').collect();
            if parts.len() == 2 {
                BetSizeOptions::try_from((parts[0].trim(), parts[1].trim()))
                    .map_err(|e| e.to_string())
            } else {
                BetSizeOptions::try_from((s, DEFAULT_RAISE_SIZES)).map_err(|e| e.to_string())
            }
        },
    );
    println!();

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot,
        effective_stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    (hero, hero_c1, hero_c2, card_config, tree_config)
}

fn prompt_input<T, F>(prompt: &str, parse: F) -> T
where
    F: Fn(&str) -> Result<T, String>,
{
    let stdin = io::stdin();
    loop {
        print!("{} ", prompt);
        io::stdout().flush().unwrap();
        let mut input = String::new();
        stdin.read_line(&mut input).unwrap();
        let trimmed = input.trim();
        match parse(trimmed) {
            Ok(val) => return val,
            Err(e) => println!("  Error: {}", e),
        }
    }
}

fn prompt_with_default(prompt: &str, default: &str) -> String {
    let stdin = io::stdin();
    print!("{} ", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    stdin.read_line(&mut input).unwrap();
    let trimmed = input.trim();
    if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    }
}

fn prompt_with_default_parse<T, F>(prompt: &str, default: T, parse: F) -> T
where
    F: Fn(&str) -> Result<T, String>,
{
    let stdin = io::stdin();
    loop {
        print!("{} ", prompt);
        io::stdout().flush().unwrap();
        let mut input = String::new();
        stdin.read_line(&mut input).unwrap();
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return default;
        }
        match parse(trimmed) {
            Ok(val) => return val,
            Err(e) => println!("  Error: {}", e),
        }
    }
}

fn format_action(action: &Action) -> String {
    match action {
        Action::Check => "Check".into(),
        Action::Fold => "Fold".into(),
        Action::Call => "Call".into(),
        Action::Bet(x) => format!("Bet({})", x),
        Action::Raise(x) => format!("Raise({})", x),
        Action::AllIn(x) => format!("AllIn({}) [ALL-IN]", x),
        _ => format!("{:?}", action),
    }
}

fn find_hero_hand_index(game: &PostFlopGame, hero: usize, c1: Card, c2: Card) -> usize {
    let lo = c1.min(c2);
    let hi = c1.max(c2);
    game.private_cards(hero)
        .binary_search(&(lo, hi))
        .expect("Hero's hand not found in private cards")
}

fn build_and_solve(tree_config: &TreeConfig, card_config: &CardConfig) -> PostFlopGame {
    let action_tree = ActionTree::new(tree_config.clone()).expect("Failed to build action tree");
    let mut game =
        PostFlopGame::with_config(card_config.clone(), action_tree).expect("Failed to create game");

    let (mem, mem_compressed) = game.memory_usage();
    println!(
        "  Memory: {:.2}MB (uncompressed) / {:.2}MB (compressed)",
        mem as f64 / (1024.0 * 1024.0),
        mem_compressed as f64 / (1024.0 * 1024.0)
    );

    game.allocate_memory(false);

    let target_exploitability = tree_config.starting_pot as f32 * 0.005;
    let exploitability = solve(&mut game, 1000, target_exploitability, true);

    println!("  Final exploitability: {:.4}", exploitability);
    game
}

fn run_interactive_loop(game: &mut PostFlopGame, hero: usize, hero_hand_index: usize) {
    println!("\n=== Starting Interactive Play ===\n");

    loop {
        game.cache_normalized_weights();

        if game.is_terminal_node() {
            handle_terminal(game, hero, hero_hand_index);
            break;
        }

        if game.is_chance_node() {
            handle_chance(game);
            continue;
        }

        let current = game.current_player();
        if current == hero {
            handle_hero_turn(game, hero, hero_hand_index);
        } else {
            handle_opponent_turn(game);
        }
    }
}

fn print_game_state(game: &PostFlopGame) {
    let board = game.current_board();
    let board_str: Vec<String> = board.iter().map(|&c| card_to_string(c).unwrap()).collect();
    let bets = game.total_bet_amount();
    println!(
        "  Board: {}  |  Pot: {}  |  Bets: OOP={}, IP={}",
        board_str.join(" "),
        game.tree_config().starting_pot + bets[0] + bets[1],
        bets[0],
        bets[1]
    );
}

fn handle_hero_turn(game: &mut PostFlopGame, hero: usize, hero_hand_index: usize) {
    let actions = game.available_actions();
    let strategy = game.strategy();
    let ev_detail = game.expected_values_detail(hero);
    let num_hands = game.private_cards(hero).len();

    print_game_state(game);
    println!("  [HERO's turn]\n");

    let mut best_ev = f32::NEG_INFINITY;
    let mut best_action = 0;

    println!("  {:<4} {:<14} {:>10} {:>8}", "  #", "Action", "EV", "Prob");
    println!("  {}", "-".repeat(42));

    for (i, action) in actions.iter().enumerate() {
        let ev = ev_detail[i * num_hands + hero_hand_index];
        let prob = strategy[i * num_hands + hero_hand_index];
        if ev > best_ev {
            best_ev = ev;
            best_action = i;
        }
        println!(
            "  {:<4} {:<14} {:>10.2} {:>7.1}%",
            i,
            format_action(action),
            ev,
            prob * 100.0
        );
    }

    println!();
    println!(
        "  >> Recommended: [{}] {} (EV = {:.2})",
        best_action,
        format_action(&actions[best_action]),
        best_ev
    );

    let chosen = prompt_input(
        &format!("  HERO's action (Enter for recommended [{}]):", best_action),
        |s| {
            if s.is_empty() {
                return Ok(best_action);
            }
            match s.parse::<usize>() {
                Ok(idx) if idx < actions.len() => Ok(idx),
                Ok(_) => Err(format!("Index must be 0-{}", actions.len() - 1)),
                Err(_) => Err("Enter a number or press Enter".into()),
            }
        },
    );

    println!(
        "  -> HERO plays: [{}] {}\n",
        chosen,
        format_action(&actions[chosen])
    );
    game.play(chosen);
}

fn handle_opponent_turn(game: &mut PostFlopGame) {
    let actions = game.available_actions();

    print_game_state(game);
    println!("  [OPPONENT's turn]\n");

    println!("  Available actions:");
    for (i, action) in actions.iter().enumerate() {
        println!("    {}: {}", i, format_action(action));
    }

    let chosen = prompt_input("  Opponent's action (number or name):", |s| {
        parse_action_input(s, &actions)
    });

    println!(
        "  -> OPPONENT plays: {}\n",
        format_action(&actions[chosen])
    );
    game.play(chosen);
}

fn handle_chance(game: &mut PostFlopGame) {
    let board = game.current_board();
    let street = if board.len() == 3 { "Turn" } else { "River" };

    print_game_state(game);
    println!("  [{} card]\n", street);

    let possible = game.possible_cards();
    let possible_count = possible.count_ones() as usize;

    // Show a few possible cards as examples
    let examples: Vec<String> = (0..52u8)
        .filter(|c| possible & (1 << c) != 0)
        .take(8)
        .map(|c| card_to_string(c).unwrap())
        .collect();
    println!(
        "  {} possible cards (e.g. {}{})",
        possible_count,
        examples.join(", "),
        if possible_count > 8 { ", ..." } else { "" }
    );

    let card = prompt_input(&format!("  {} card:", street), |s| {
        let c = card_from_str(s)?;
        if possible & (1 << c) == 0 {
            Err("This card cannot be dealt".into())
        } else {
            Ok(c)
        }
    });

    println!(
        "  -> Dealt: {}\n",
        card_to_string(card).unwrap()
    );
    game.play(card as usize);
}

fn handle_terminal(game: &mut PostFlopGame, hero: usize, hero_hand_index: usize) {
    game.cache_normalized_weights();

    let board = game.current_board();
    let board_str: Vec<String> = board.iter().map(|&c| card_to_string(c).unwrap()).collect();
    let bets = game.total_bet_amount();
    let final_pot = game.tree_config().starting_pot + bets[0] + bets[1];

    println!("  Board: {}  |  Final pot: {}", board_str.join(" "), final_pot);
    println!("\n  === Game Over ===\n");

    let equity = game.equity(hero);
    let ev = game.expected_values(hero);
    let hero_equity = equity[hero_hand_index];
    let hero_ev = ev[hero_hand_index];

    let hero_str =
        holes_to_strings(&[game.private_cards(hero)[hero_hand_index]]).unwrap();
    println!("  Hero's hand: {}", hero_str[0]);
    println!("  Equity: {:.1}%", hero_equity * 100.0);
    println!("  EV: {:.2}", hero_ev);
}

fn parse_action_input(input: &str, available: &[Action]) -> Result<usize, String> {
    // Try as number first
    if let Ok(idx) = input.parse::<usize>() {
        if idx < available.len() {
            return Ok(idx);
        }
        return Err(format!("Index must be 0-{}", available.len() - 1));
    }

    let lower = input.to_lowercase();
    let parts: Vec<&str> = lower.split_whitespace().collect();
    if parts.is_empty() {
        return Err("Enter an action number or name".into());
    }

    let keyword = parts[0];
    let amount: Option<i32> = if parts.len() > 1 {
        parts[1].parse().ok()
    } else {
        None
    };

    for (i, action) in available.iter().enumerate() {
        let matches = match (keyword, action, amount) {
            ("check" | "c" | "x", Action::Check, None) => true,
            ("fold" | "f", Action::Fold, None) => true,
            ("call", Action::Call, None) => true,
            ("bet" | "b", Action::Bet(x), Some(amt)) if *x == amt => true,
            ("raise" | "r", Action::Raise(x), Some(amt)) if *x == amt => true,
            ("allin" | "ai", Action::AllIn(x), Some(amt)) if *x == amt => true,
            ("allin" | "ai", Action::AllIn(_), None) => true,
            _ => false,
        };
        if matches {
            return Ok(i);
        }
    }

    Err(format!(
        "Unknown action '{}'. Available: {}",
        input,
        available
            .iter()
            .enumerate()
            .map(|(i, a)| format!("{}={}", i, format_action(a)))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}
