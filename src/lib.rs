//! Solver for post-flop game.
//!
//! ```ignore
//! use postflop_solver::*;
//!
//! // configure game specification
//! let oop_range = "22+,A2s+,A8o+,K7s+,K9o+,Q8s+,Q9o+,J8s+,J9o+,T8+,97+,86+,75+,64s+,65o,54,43s";
//! let ip_range = "22+,A4s+,A9o+,K9s+,KTo+,Q9s+,QTo+,J9+,T9,98s,87s,76s,65s";
//! let bet_sizes = bet_sizes_from_str("50%", "50%").unwrap();
//! let config = GameConfig {
//!     flop: flop_from_str("Td9d6h").unwrap(),
//!     initial_pot: 60,
//!     initial_stack: 370,
//!     range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
//!     flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
//!     turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
//!     river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
//!     max_num_bet: 4,
//! };
//!
//! // build game tree
//! let game = PostFlopGame::new(&config, None).unwrap();
//!
//! // solve game
//! let exploitability = solve(&game, 1000, config.initial_pot as f32 * 0.005, true);
//!
//! // compute OOP's EV
//! let ev = compute_ev(&game, 0) + config.initial_pot as f32 / 2.0;
//! ```

mod bet_size;
mod game;
mod interface;
mod mutex_like;
mod range;
mod sliceop;
mod solver;
mod utility;

pub use bet_size::*;
pub use game::*;
pub use interface::*;
pub use mutex_like::*;
pub use range::*;
pub use sliceop::*;
pub use solver::*;
pub use utility::*;
