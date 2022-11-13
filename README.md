# postflop-solver

An open-source postflop solver library written in Rust

Documentation: https://b-inary.github.io/postflop_solver/postflop_solver/

**Related repositories**
- Web application: https://github.com/b-inary/wasm-postflop
- Desktop application: https://github.com/b-inary/desktop-postflop

## Usage

- `Cargo.toml`

```toml
[dependencies]
postflop-solver = { git = "https://github.com/b-inary/postflop-solver" }
```

- Examples

```rust
use postflop_solver::*;

// ranges of OOP and IP in string format
let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";

// bet size -> 70% of pot / raise size -> 45% of pot
// (multiple sizes can be specified by using a comma-separated string)
let bet_sizes = BetSizeCandidates::try_from(("70%", "45%")).unwrap();

// donk size -> 50% of pot
let donk_sizes = DonkSizeCandidates::try_from("50%").unwrap();

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
    turn_donk_sizes: None, // do not distinguish between donk bets and other bets
    river_donk_sizes: Some(donk_sizes.clone()), // use 50% size for donk bets
    add_all_in_threshold: 1.2,
    force_all_in_threshold: 0.1,
    adjust_last_two_bet_sizes: false,
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

// solve the game manually
// for i in 0..max_num_iterations {
//     solve_step(&game, i);
//     if (i + 1) % 10 == 0 {
//         let exploitability = compute_exploitability(&game);
//         if exploitability <= target_exploitability {
//             println!("Exploitability: {:.2}", exploitability);
//             break;
//         }
//     }
// }
// finalize(&mut game);

// get equity and EV of a specific hand
game.cache_normalized_weights();
let equity = game.equity(game.current_player());
let ev = game.expected_values();
println!("Equity of oop_hands[0]: {:.2}%", 100.0 * equity[0]);
println!("EV of oop_hands[0]: {:.2}", ev[0]);

// get equity and EV of whole hand
let weights = game.normalized_weights(game.current_player());
let average_equity = compute_average(&equity, weights);
let average_ev = compute_average(&ev, weights);
println!("Average equity: {:.2}%", 100.0 * average_equity);
println!("Average EV: {:.2}", average_ev);

// get available actions
let actions = game.available_actions();
println!("Available actions: {:?}", actions); // [Check, Bet(140)]

// play `Bet(100)`
game.play(1);

// get available actions
let actions = game.available_actions();
println!("Available actions: {:?}", actions); // [Fold, Call, Raise(356)]

// play `Call`
game.play(1);

// confirm that the current node is a chance node
assert!(game.is_chance_node());

// confirm that "7s" may be dealt
let card = card_from_str("7s").unwrap();
assert!(game.possible_cards() & (1 << card) != 0);

// deal "7s"
game.play(card as usize);

// back to the root node
game.back_to_root();
```

## Implementation details

- **Algorithm**: The solver uses [Discounted CFR] algorithm.
  Currently, the value of γ is set to 5.0, rather than the 2.0 recommended by the original paper.
- **Precision**: 32-bit floating-point numbers are used in most places.
  When calculating summations, temporal values use 64-bit floating-point numbers.
  If the compression feature is enabled, each game node stores the values by 16-bit integers with a single 32-bit floating-point scaling factor.
- **Handling isomorphism**: The solver does not perform any abstraction.
  However, isomorphic chances (turn and river deals) are combined into one.
  For example, if the flop is monotone, the three non-dealt suits are isomorphic, allowing us to skip the calculation for two of the three suits.

[Discounted CFR]: https://arxiv.org/abs/1809.04040

## Crate features

- `bincode`: Uses [bincode] crate (2.0.0-rc.2) to serialize and deserialize the `PostFlopGame` struct.
  Disabled by default.
- `custom-alloc`: Uses custom memory allocator in solving process (only available in nightly Rust).
  It significantly reduces the number of calls of the default allocator, so it is recommended to use this feature when the default allocator is not so efficient.
  Note that this feature assumes that, at most, only one instance of `PostFlopGame` is available when solving in a program.
  Disabled by default.
- `rayon`: Uses [rayon] crate for parallelization.
  Enabled by default.

[bincode]: https://github.com/bincode-org/bincode
[rayon]: https://github.com/rayon-rs/rayon

## License

Copyright (C) 2022 Wataru Inariba

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
