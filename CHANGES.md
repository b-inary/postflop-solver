# List of breaking changes

## 2022-11-29

- struct `GameConfig` is split into `CardConfig` and `TreeConfig`.
- new struct `ActionTree` that takes `TreeConfig` for instantiation.
- now `PostFlopGame` takes `CardConfig` and `ActionTree` for instantiation.
- `add_all_in_threshold` and `force_all_in_threshold` are renamed to `add_allin_threshold` and `force_allin_threshold`, respectively (`all_in` -> `allin`).
- `adjust_bet_size_before_all_in` (renamed from `adjust_last_two_bet_sizes`) is removed.

## 2022-11-27

- enum `BetSize` has new variants: `Additive(i32)`, `Geometric(i32, f32)`, and `AllIn`.
- `BetSize::LastBetRelative` is renamed to `BetSize::PrevBetRelative`.
- `BetSizeCandidates::try_from()` method is refactored. See the documentation for details. Now a pot-relative size must be specified with the '%' character, and the `try_from()` method rejects a single floating number.
- `adjust_last_two_bet_sizes` field of `GameConfig` struct is renamed to `adjust_bet_size_before_all_in`.

## 2022-11-14

- struct `GameConfig` has new fields: `turn_donk_sizes` and `river_donk_sizes`. Their types are `Option<DonkSizeCandidates>`. Specify these as `None` to maintain the previous behavior.
