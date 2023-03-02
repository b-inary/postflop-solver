# postflop-solver

An open-source postflop solver library written in Rust

Documentation: https://b-inary.github.io/postflop_solver/postflop_solver/

**Related repositories**
- Web app (WASM Postflop): https://github.com/b-inary/wasm-postflop
- Desktop app (Desktop Postflop): https://github.com/b-inary/desktop-postflop

**Note:**
The primary purpose of this library is to serve as a backend engine for the GUI applications ([WASM Postflop] and [Desktop Postflop]).
The direct use of this library by the users/developers is not a critical purpose by design.
Therefore, breaking changes are often made without version changes.
See [CHANGES.md](CHANGES.md) for details about breaking changes.

[WASM Postflop]: https://github.com/b-inary/wasm-postflop
[Desktop Postflop]: https://github.com/b-inary/desktop-postflop

## Usage

- `Cargo.toml`

```toml
[dependencies]
postflop-solver = { git = "https://github.com/b-inary/postflop-solver" }
```

- Examples

You can find use cases in [examples/basic.rs](examples/basic.rs).

If you have cloned this repository, you can run the example with the following command:

```sh
$ cargo run --release --example basic
```

## Implementation details

- **Algorithm**: The solver uses [Discounted CFR] algorithm.
  Currently, the value of γ is set to 3.0, rather than the 2.0 recommended by the original paper.
  Also, the solver reset the cumulative strategy when the number of iterations is a power of 4.
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
