# postflop-solver

An open-source postflop solver written in Rust

Web interface repository: https://github.com/b-inary/wasm-postflop

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
    turn: NOT_DEALT, // or `card_from_str("As").unwrap()`
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

// check memory usage
let (mem_usage, mem_usage_compressed) = game.memory_usage();
println!(
    "memory usage without compression: {:.2}GB",
    mem_usage as f64 / (1024.0 * 1024.0 * 1024.0)
);
println!(
    "memory usage with compression: {:.2}GB",
    mem_usage_compressed as f64 / (1024.0 * 1024.0 * 1024.0)
);

// allocate memory without compression
game.allocate_memory(false);

// allocate memory with compression
// game.allocate_memory(true);

// solve game
let max_num_iterations = 1000;
let target_exploitability = config.starting_pot as f32 * 0.005;
let exploitability = solve(&mut game, max_num_iterations, target_exploitability, true);

// compute OOP's EV and equity
let bias = config.starting_pot as f32 * 0.5;
let ev = get_root_ev(&game) + bias;
let equity = get_root_equity(&game) + 0.5;
```

## License

Copyright (C) 2022 Wataru Inariba

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
