//! An open-source postflop solver library.
//!
//! ```
//! use postflop_solver::*;
//!
//! // configure game specification
//! let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
//! let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";
//! let bet_sizes = BetSizeCandidates::try_from(("50%", "50%")).unwrap();
//! let config = GameConfig {
//!     flop: flop_from_str("Td9d6h").unwrap(),
//!     turn: card_from_str("Qh").unwrap(),
//!     river: NOT_DEALT,
//!     starting_pot: 200,
//!     effective_stack: 900,
//!     range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
//!     flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
//!     turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
//!     river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
//!     add_all_in_threshold: 1.2,
//!     force_all_in_threshold: 0.1,
//!     adjust_last_two_bet_sizes: true,
//! };
//!
//! // build game tree
//! let mut game = PostFlopGame::with_config(&config).unwrap();
//!
//! // check memory usage
//! let (mem_usage, mem_usage_compressed) = game.memory_usage();
//! println!(
//!     "memory usage without compression: {:.2}GB",
//!     mem_usage as f64 / (1024.0 * 1024.0 * 1024.0)
//! );
//! println!(
//!     "memory usage with compression: {:.2}GB",
//!     mem_usage_compressed as f64 / (1024.0 * 1024.0 * 1024.0)
//! );
//!
//! // allocate memory without compression
//! game.allocate_memory(false);
//!
//! // allocate memory with compression
//! // game.allocate_memory(true);
//!
//! // solve game
//! let max_num_iterations = 1000;
//! let target_exploitability = config.starting_pot as f32 * 0.005;
//! let exploitability = solve(&mut game, max_num_iterations, target_exploitability, true);
//!
//! // compute OOP's EV and equity
//! let bias = config.starting_pot as f32 * 0.5;
//! let ev = get_root_ev(&game) + bias;
//! let equity = get_root_equity(&game) + 0.5;
//! ```

#![cfg_attr(feature = "custom_alloc", feature(allocator_api))]

mod bet_size;
mod game;
mod interface;
mod mutex_like;
mod range;
mod sliceop;
mod solver;
mod utility;

#[cfg(feature = "custom_alloc")]
mod alloc;

#[cfg(not(feature = "holdem-hand-evaluator"))]
mod hand;

pub use bet_size::*;
pub use game::*;
pub use interface::*;
pub use mutex_like::*;
pub use range::*;
pub use sliceop::*;
pub use solver::*;
pub use utility::*;

#[cfg(feature = "custom_alloc")]
pub use alloc::*;

#[cfg(not(feature = "holdem-hand-evaluator"))]
pub use hand::*;
