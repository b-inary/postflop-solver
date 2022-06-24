# postflop-solver

An open-source postflop solver written in Rust

Web interface repository: https://github.com/b-inary/wasm-postflop

Documentation: https://b-inary.github.io/postflop_solver/postflop_solver/

## Usage

`Cargo.toml`

```toml
[dependencies]
postflop-solver = { git = "https://github.com/b-inary/postflop-solver" }
```

`example.rs`

```rust
use postflop_solver::*;

// configure game specification
let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";
let bet_sizes = BetSizeCandidates::try_from(("50%", "50%")).unwrap();
let config = GameConfig {
    flop: flop_from_str("Td9d6h").unwrap(),
    turn: NOT_DEALT, // or `card_from_str("Qh").unwrap()`
    river: NOT_DEALT,
    starting_pot: 200,
    effective_stack: 900,
    range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
    flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
    turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
    river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
    add_all_in_threshold: 1.2,
    force_all_in_threshold: 0.1,
    adjust_last_two_bet_sizes: true,
};

// build game tree
let mut game = PostFlopGame::with_config(&config).unwrap();

// obtain private hands
let oop_hands = game.private_hand_cards(0);
println!(
    "oop_hands[0]: {}{}",
    card_to_string(oop_hands[0].1).unwrap(), // 5c
    card_to_string(oop_hands[0].0).unwrap()  // 4c
);

// check memory usage
let (mem_usage, mem_usage_compressed) = game.memory_usage();
println!(
    "Memory usage without compression: {:.2}GB",
    mem_usage as f64 / (1024.0 * 1024.0 * 1024.0)
);
println!(
    "Memory usage with compression: {:.2}GB",
    mem_usage_compressed as f64 / (1024.0 * 1024.0 * 1024.0)
);

// allocate memory without compression
game.allocate_memory(false);

// allocate memory with compression
// game.allocate_memory(true);

// solve the game
let max_num_iterations = 1000;
let target_exploitability = config.starting_pot as f32 * 0.005;
let exploitability = solve(&mut game, max_num_iterations, target_exploitability, true);
println!("Exploitability: {:.2}", exploitability);

// create result interpreter
let mut interpreter = Interpreter::new(&game, 0.0);

// get EV and equity of a specific hand
interpreter.cache_normalized_weights();
let ev = interpreter.expected_values();
let equity = interpreter.equity();
println!("EV of oop_hands[0]: {:.2}", ev[0]);
println!("Equity of oop_hands[0]: {:.2}%", 100.0 * equity[0]);

// get EV and equity of whole hand
let weights = interpreter.normalized_weights(interpreter.current_player());
let average_ev = compute_average(&ev, weights);
let average_equity = compute_average(&equity, weights);
println!("Average EV: {:.2}", average_ev);
println!("Average equity: {:.2}%", 100.0 * average_equity);

// get available actions
let actions = interpreter.available_actions();
println!("Available actions: {:?}", actions); // [Check, Bet(100)]

// play `Bet(100)`
interpreter.play(1);

// get available actions
let actions = interpreter.available_actions();
println!("Available actions: {:?}", actions); // [Fold, Call, Raise(300)]

// play `Call`
interpreter.play(1);

// confirm that the current node is a chance node
assert!(interpreter.is_chance_node());

// confirm that "7s" may be dealt
let card = card_from_str("7s").unwrap();
assert!(interpreter.possible_cards() & (1 << card) != 0);

// deal "7s"
interpreter.play(card as usize);
```

## Features

- `custom_alloc`: Uses custom memory allocator in solving process.
  It significantly reduces the number of calls of the default allocator, so it is recommended to use this feature when the default allocator is not so efficient.
  Disabled by default.
- `holdem-hand-evaluator`: Uses [holdem-hand-evaluator] crate to evaluate hands.
  It makes the tree construction slightly faster, but the program size will increase by about 200KB.
  Enabled by default.
- `rayon`: Uses [rayon] crate for parallelization.
  Enabled by default.

[holdem-hand-evaluator]: https://github.com/b-inary/holdem-hand-evaluator
[rayon]: https://github.com/rayon-rs/rayon

## License

Copyright (C) 2022 Wataru Inariba

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
