# List of breaking changes

## 2023-10-01

- `BetSizeCandidates` and `DonkSizeCandidates` are renamed to `BetSizeOptions` and `DonkSizeOptions`, respectively.

## 2023-02-23

- `available_actions()` method of `PostFlopGame` now returns `Vec<Action>` instead of `&[Action]`.

## 2022-12-13

- revert: `compute_exploitability` function is back, and `compute_mes_ev_average` function is removed.

## 2022-12-11

- `TreeConfig`: new fields `rake_rate` and `rake_cap` are added.
- real numbers in `BetSize` enum  and `TreeConfig` struct are now represented as `f64` instead of `f32`.
- `compute_exploitability` function is renamed to `compute_mes_ev_average`.

## 2022-12-07

- `PostFlopGame`:
  - `play`: now terminal actions can be played.
  - `is_terminal_action` method is removed and `is_terminal_node` method is added.
  - `expected_values` and `expected_values_detail` methods now take a `player` argument.

## 2022-12-02

- `ActionTree`: `new` constructor now takes a `TreeConfig` argument.
- `ActionTree`: `with_config` and `update_config` methods are removed.

## 2022-11-30

- `TreeConfig`: `merging_threshold` field is added.
- `PostFlopGame`: `private_hand_cards` method is renamed to `private_cards`.

## 2022-11-29

- struct `GameConfig` is split into `CardConfig` and `TreeConfig`.
- new struct `ActionTree` is added: takes `TreeConfig` for instantiation.
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
