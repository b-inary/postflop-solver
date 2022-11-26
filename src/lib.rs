//! An open-source postflop solver library.
//!
//! # Examples
//! ```
//! use postflop_solver::*;
//!
//! // ranges of OOP and IP in string format
//! let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
//! let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";
//!
//! // bet sizes -> 60% of the pot, geometric size, and all-in
//! // raise sizes -> 2.5x of the previous bet
//! // see the documentation of `BetSizeCandidates` for more details
//! let bet_sizes = BetSizeCandidates::try_from(("60%, e, a", "2.5x")).unwrap();
//!
//! let config = GameConfig {
//!     flop: flop_from_str("Td9d6h").unwrap(),
//!     turn: card_from_str("Qc").unwrap(),
//!     river: NOT_DEALT,
//!     starting_pot: 200,
//!     effective_stack: 900,
//!     range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
//!     flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()], // [OOP, IP]
//!     turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
//!     river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
//!     turn_donk_sizes: None, // use default bet sizes
//!     river_donk_sizes: Some(DonkSizeCandidates::try_from("50%").unwrap()),
//!     add_all_in_threshold: 1.2, // add all-in if the current SPR is less than 1.2
//!     force_all_in_threshold: 0.1, // force all-in if the SPR after call is less than 0.1
//!     adjust_bet_size_before_all_in: false,
//! };
//!
//! // build the game tree
//! let mut game = PostFlopGame::with_config(&config).unwrap();
//!
//! // obtain the private hands
//! let oop_hands = game.private_hand_cards(0);
//!
//! // check oop_hands[0]
//! assert_eq!(card_to_string(oop_hands[0].0).unwrap(), "4c");
//! assert_eq!(card_to_string(oop_hands[0].1).unwrap(), "5c");
//!
//! // check memory usage
//! let (mem_usage, mem_usage_compressed) = game.memory_usage();
//! println!(
//!     "Memory usage without compression: {:.2}GB",
//!     mem_usage as f64 / (1024.0 * 1024.0 * 1024.0)
//! );
//! println!(
//!     "Memory usage with compression: {:.2}GB",
//!     mem_usage_compressed as f64 / (1024.0 * 1024.0 * 1024.0)
//! );
//!
//! // allocate memory without compression
//! game.allocate_memory(false);
//!
//! // allocate memory with compression
//! // game.allocate_memory(true);
//!
//! // solve the game
//! let max_num_iterations = 1000;
//! let target_exploitability = config.starting_pot as f32 * 0.005;
//! let exploitability = solve(&mut game, max_num_iterations, target_exploitability, true);
//! println!("Exploitability: {:.2}", exploitability);
//!
//! // solve the game manually
//! // for i in 0..max_num_iterations {
//! //     solve_step(&game, i);
//! //     if (i + 1) % 10 == 0 {
//! //         let exploitability = compute_exploitability(&game);
//! //         if exploitability <= target_exploitability {
//! //             println!("Exploitability: {:.2}", exploitability);
//! //             break;
//! //         }
//! //     }
//! // }
//! // finalize(&mut game);
//!
//! // get equity and EV of a specific hand
//! game.cache_normalized_weights();
//! let equity = game.equity(game.current_player());
//! let ev = game.expected_values();
//! println!("Equity of oop_hands[0]: {:.2}%", 100.0 * equity[0]);
//! println!("EV of oop_hands[0]: {:.2}", ev[0]);
//!
//! // get equity and EV of whole hand
//! let weights = game.normalized_weights(game.current_player());
//! let average_equity = compute_average(&equity, weights);
//! let average_ev = compute_average(&ev, weights);
//! println!("Average equity: {:.2}%", 100.0 * average_equity);
//! println!("Average EV: {:.2}", average_ev);
//!
//! // get available actions
//! let actions = game.available_actions();
//! assert_eq!(format!("{:?}", actions), "[Check, Bet(120), Bet(216), AllIn(900)]");
//!
//! // play `Bet(120)`
//! game.play(1);
//!
//! // get available actions
//! let actions = game.available_actions();
//! assert_eq!(format!("{:?}", actions), "[Fold, Call, Raise(300)]");
//!
//! // play `Call`
//! game.play(1);
//!
//! // confirm that the current node is a chance node
//! assert!(game.is_chance_node());
//!
//! // confirm that "7s" may be dealt
//! let card_7s = card_from_str("7s").unwrap();
//! assert!(game.possible_cards() & (1 << card_7s) != 0);
//!
//! // deal "7s"
//! game.play(card_7s as usize);
//!
//! // back to the root node
//! game.back_to_root();
//! ```
//!
//! # Implementation details
//! - **Algorithm**: The solver uses [Discounted CFR] algorithm.
//!   Currently, the value of γ is set to 5.0, rather than the 2.0 recommended by the original paper.
//! - **Precision**: 32-bit floating-point numbers are used in most places.
//!   When calculating summations, temporal values use 64-bit floating-point numbers.
//!   If the compression feature is enabled, each game node stores the values by 16-bit integers
//!   with a single 32-bit floating-point scaling factor.
//! - **Handling isomorphism**: The solver does not perform any abstraction.
//!   However, isomorphic chances (turn and river deals) are combined into one.
//!   For example, if the flop is monotone, the three non-dealt suits are isomorphic,
//!   allowing us to skip the calculation for two of the three suits.
//!
//! [Discounted CFR]: https://arxiv.org/abs/1809.04040
//!
//! # Crate features
//! - `bincode`: Uses [bincode] crate (2.0.0-rc.2) to serialize and deserialize the `PostFlopGame` struct.
//!   Disabled by default.
//! - `custom-alloc`: Uses custom memory allocator in solving process (only available in nightly Rust).
//!   It significantly reduces the number of calls of the default allocator,
//!   so it is recommended to use this feature when the default allocator is not so efficient.
//!   Note that this feature assumes that, at most, only one instance of `PostFlopGame` is available
//!   when solving in a program.
//!   Disabled by default.
//! - `rayon`: Uses [rayon] crate for parallelization.
//!   Enabled by default.
//!
//! [bincode]: https://github.com/bincode-org/bincode
//! [rayon]: https://github.com/rayon-rs/rayon

#![cfg_attr(feature = "custom-alloc", feature(allocator_api))]

mod bet_size;
mod game;
mod hand;
mod hand_table;
mod interface;
mod mutex_like;
mod range;
mod sliceop;
mod solver;
mod utility;

#[cfg(feature = "custom-alloc")]
mod alloc;

pub use bet_size::*;
pub use game::*;
pub use interface::*;
pub use mutex_like::*;
pub use range::*;
pub use solver::*;
pub use utility::*;
