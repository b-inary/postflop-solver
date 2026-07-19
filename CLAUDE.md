# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 个人习惯
- 所有的回复和分析使用中文。

## Project overview

An open-source Texas hold'em postflop solver library implementing the Discounted CFR algorithm. Designed as a backend engine for GUI apps (WASM Postflop, Desktop Postflop). Breaking changes may occur without version bumps; check `CHANGES.md` for the breaking change log.

## Build and development commands

```sh
# Build (default features: bincode, rayon)
cargo build --release

# Build with all features (requires zstd)
cargo build --release --features zstd

# Build with custom allocator (requires nightly Rust)
cargo +nightly build --release --features custom-alloc

# Run all tests
cargo test --release --features zstd

# Run a single test by name
cargo test --release --features zstd -- <test_name>

# Run tests with custom-alloc (requires nightly, single-threaded)
cargo +nightly test --release --features custom-alloc -- --test-threads 1

# Run examples
cargo run --release --example basic
cargo run --release --example file_io
cargo run --release --example node_locking

# Lint
cargo clippy --release --features zstd -- -A clippy::needless_range_loop

# Format check
cargo fmt --all --check

# Generate docs
cargo doc --release
```

The CI pipeline (`RUSTFLAGS: --deny warnings`) treats all warnings as errors. Ensure `cargo clippy` and `cargo fmt --check` pass cleanly.

## Architecture

### Core abstraction: Game and GameNode traits (`src/interface.rs`)

The solver is generic over two traits:
- `Game`: represents a game (root node, private hands, evaluation, chance factors, isomorphism)
- `GameNode`: represents a tree node (terminal/chance checks, strategy, regrets, counterfactual values, compression variants)

`PostFlopGame` (`src/game/mod.rs`) is the concrete implementation for Texas hold'em.

### Game lifecycle (state machine)

`PostFlopGame` progresses through states: `Uninitialized -> TreeBuilt -> MemoryAllocated -> Solved`.
- Construct with `CardConfig` (ranges, board cards) + `ActionTree` (built from `TreeConfig`)
- Call `allocate_memory(bool)` to allocate storage (bool = compression via i16 with f32 scaling factor)
- Call `solve()` or manually loop `solve_step()` + `compute_exploitability()` + `finalize()`
- After solving, navigate the tree with `play(action_index)`, `back_to_root()`, `apply_history()`

### Key modules

- `src/solver.rs` — Discounted CFR algorithm. `solve()` runs to target exploitability; `solve_step()` runs one iteration. `finalize()` must be called after manual solving loops.
- `src/action_tree.rs` — `ActionTree` builds the game tree from `TreeConfig` (pot, stack, bet sizes, thresholds). `Action` enum defines Fold/Check/Call/Bet/Raise/AllIn/Chance.
- `src/bet_size.rs` — `BetSizeOptions` and `DonkSizeOptions` parsed from strings ("60%, e, a" / "2.5x"). Supports pot-relative, geometric, additive, all-in sizes.
- `src/game/base.rs` — `Game` trait implementation for `PostFlopGame`: tree construction, memory allocation, isomorphism detection.
- `src/game/evaluation.rs` — Terminal node counterfactual value computation, with and without bunching effect.
- `src/game/interpreter.rs` — Tree navigation, querying strategies/EVs/equities, card dealing on chance nodes.
- `src/bunching.rs` — Bunching effect support (up to 4 folded players, 6-max). Combinatorics-based, not heuristic.
- `src/card.rs` — `Card = u8`, encoded as `4 * rank + suit`. `NOT_DEALT = u8::MAX`.
- `src/range.rs` — Hand range parsing ("66+,A8s+,AJo+").
- `src/hand.rs`, `src/hand_table.rs` — Hand evaluation and lookup tables.
- `src/mutex_like.rs` — `MutexLike<T>` wrapper for safe concurrent access during solving.
- `src/atomic_float.rs` — Atomic float for parallel accumulation.
- `src/sliceop.rs` — SIMD-friendly slice operations (dot products, element-wise ops).
- `src/file.rs` — Serialization/deserialization of `PostFlopGame` (requires `bincode` feature; `zstd` adds compression).
- `src/alloc.rs` — Custom stack allocator for hot-path allocations (nightly only, `custom-alloc` feature).

### Performance-critical patterns

- Hot paths use `unsafe` Rust extensively with `get_unchecked` and raw pointer manipulation for SIMD auto-vectorization.
- Compression mode stores strategy/regrets/cfvalues as `i16` with per-node `f32` scale factors, trading precision for memory.
- Isomorphic chances (e.g., equivalent suits on monotone boards) are combined to skip redundant computation.
- Parallelization via `rayon` is opt-in per node (`enable_parallelization()` returns true for sufficiently large subtrees).
- `custom-alloc` feature uses a stack allocator to minimize heap allocations during solving.

## Testing

- Integration tests in `tests/` implement Kuhn poker and Leduc poker using the `Game`/`GameNode` traits to validate the solver on toy games.
- Unit tests in `src/game/tests.rs` validate `PostFlopGame` behavior (all-check scenarios, bet sizing, serialization round-trips).
- Examples in `examples/` serve as smoke tests and are run in CI.
