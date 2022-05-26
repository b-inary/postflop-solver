# postflop-solver

Efficient open-source post-flop solver library written in Rust (work in progress)

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
let oop_range = "22+,A2s+,A8o+,K7s+,K9o+,Q8s+,Q9o+,J8s+,J9o+,T8+,97+,86+,75+,64s+,65o,54,43s";
let ip_range = "22+,A4s+,A9o+,K9s+,KTo+,Q9s+,QTo+,J9+,T9,98s,87s,76s,65s";
let bet_sizes = bet_sizes_from_str("50%", "50%").unwrap();
let config = GameConfig {
    flop: flop_from_str("Td9d6h").unwrap(),
    initial_pot: 60,
    initial_stack: 370,
    range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
    flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
    turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
    river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
    ..Default::default()
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
let target_exploitability = config.initial_pot as f32 * 0.005;
let exploitability = solve(&game, max_num_iterations, target_exploitability, true);

// compute OOP's EV
compute_ev(&game);
let bias = config.initial_pot as f32 * 0.5;
let ev = compute_ev_scalar(&game, &game.root()) + bias;
```
