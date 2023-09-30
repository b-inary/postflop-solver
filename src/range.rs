use crate::card::*;
use once_cell::sync::Lazy;
use regex::Regex;
use std::fmt::Write;
use std::str::FromStr;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// A struct representing a player's range.
///
/// The [`Range`] struct implements the [`FromStr`] trait, so you can construct a range from a string
/// using `parse::<Range>()`. The string must be in the following format (similar to PioSOLVER):
///
/// - Each group is separated by a comma. (e.g., "AA,AKs")
/// - Each group can have an optional weight separated by a colon. (e.g., "AA:0.5")
/// - Each group must be one of the following:
///   - Singleton (e.g., "AA", "AKs", "AKo", "AsAh")
///   - Plus range (e.g., "TT+", "ATs+", "T9o+")
///   - Dash range (e.g., "QQ-88", "A9s-A6s", "98o-65o")
///
/// # Examples
/// ```
/// use postflop_solver::Range;
///
/// // construct a range from a string
/// let range = "QQ+,AKs".parse::<Range>().unwrap();
///
/// // rank is defined as follows: A => 12, K => 11, ..., 2 => 0
/// let ace_rank = 12;
/// let king_rank = 11;
/// let queen_rank = 10;
///
/// // check that the hand "QQ" is in the range
/// assert_eq!(range.get_weight_pair(queen_rank), 1.0);
///
/// // check that the hand "AKo" is not in the range
/// assert_eq!(range.get_weight_offsuit(ace_rank, king_rank), 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct Range {
    data: [f32; 52 * 51 / 2],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Suitedness {
    Suited,
    Offsuit,
    All,
    Specific(u8, u8),
}

const COMBO_PAT: &str = r"(?:(?:[AaKkQqJjTt2-9]{2}[os]?)|(?:(?:[AaKkQqJjTt2-9][cdhs]){2}))";
const WEIGHT_PAT: &str = r"(?:(?:[01](\.\d*)?)|(?:\.\d+))";

static RANGE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(&format!(
        r"^(?P<range>{COMBO_PAT}(?:\+|(?:-{COMBO_PAT}))?)(?::(?P<weight>{WEIGHT_PAT}))?$"
    ))
    .unwrap()
});

static TRIM_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*([-:,])\s*").unwrap());

#[inline]
fn pair_indices(rank: u8) -> Vec<usize> {
    let mut result = Vec::with_capacity(6);
    for i in 0..4 {
        for j in i + 1..4 {
            result.push(card_pair_to_index(4 * rank + i, 4 * rank + j));
        }
    }
    result
}

#[inline]
fn nonpair_indices(rank1: u8, rank2: u8) -> Vec<usize> {
    let mut result = Vec::with_capacity(16);
    for i in 0..4 {
        for j in 0..4 {
            result.push(card_pair_to_index(4 * rank1 + i, 4 * rank2 + j));
        }
    }
    result
}

#[inline]
fn suited_indices(rank1: u8, rank2: u8) -> Vec<usize> {
    let mut result = Vec::with_capacity(4);
    for i in 0..4 {
        result.push(card_pair_to_index(4 * rank1 + i, 4 * rank2 + i));
    }
    result
}

#[inline]
fn offsuit_indices(rank1: u8, rank2: u8) -> Vec<usize> {
    let mut result = Vec::with_capacity(12);
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                result.push(card_pair_to_index(4 * rank1 + i, 4 * rank2 + j));
            }
        }
    }
    result
}

#[inline]
fn indices_with_suitedness(rank1: u8, rank2: u8, suitedness: Suitedness) -> Vec<usize> {
    if rank1 == rank2 {
        match suitedness {
            Suitedness::All => pair_indices(rank1),
            Suitedness::Specific(suit1, suit2) => {
                vec![card_pair_to_index(4 * rank1 + suit1, 4 * rank1 + suit2)]
            }
            _ => panic!("invalid suitedness with a pair"),
        }
    } else {
        match suitedness {
            Suitedness::Suited => suited_indices(rank1, rank2),
            Suitedness::Offsuit => offsuit_indices(rank1, rank2),
            Suitedness::All => nonpair_indices(rank1, rank2),
            Suitedness::Specific(suit1, suit2) => {
                vec![card_pair_to_index(4 * rank1 + suit1, 4 * rank2 + suit2)]
            }
        }
    }
}

/// Attempts to convert a rank character to a rank index.
///
/// `'A'` => `12`, `'K'` => `11`, ..., `'2'` => `0`.
#[inline]
fn char_to_rank(c: char) -> Result<u8, String> {
    match c {
        'A' | 'a' => Ok(12),
        'K' | 'k' => Ok(11),
        'Q' | 'q' => Ok(10),
        'J' | 'j' => Ok(9),
        'T' | 't' => Ok(8),
        '2'..='9' => Ok(c as u8 - b'2'),
        _ => Err(format!("Expected rank character: {c}")),
    }
}

/// Attempts to conver a suit character to a suit index.
///
/// `'c'` => `0`, `'d'` => `1`, `'h'` => `2`, `'s'` => `3`.
#[inline]
fn char_to_suit(c: char) -> Result<u8, String> {
    match c {
        'c' => Ok(0),
        'd' => Ok(1),
        'h' => Ok(2),
        's' => Ok(3),
        _ => Err(format!("Expected suit character: {c}")),
    }
}

/// Attempts to convert a rank index to a rank character.
///
/// `12` => `'A'`, `11` => `'K'`, ..., `0` => `'2'`.
#[inline]
fn rank_to_char(rank: u8) -> Result<char, String> {
    match rank {
        12 => Ok('A'),
        11 => Ok('K'),
        10 => Ok('Q'),
        9 => Ok('J'),
        8 => Ok('T'),
        0..=7 => Ok((rank + b'2') as char),
        _ => Err(format!("Invalid input: {rank}")),
    }
}

/// Attempts to convert a suit index to a suit character.
///
/// `0` => `'c'`, `1` => `'d'`, `2` => `'h'`, `3` => `'s'`.
#[inline]
fn suit_to_char(suit: u8) -> Result<char, String> {
    match suit {
        0 => Ok('c'),
        1 => Ok('d'),
        2 => Ok('h'),
        3 => Ok('s'),
        _ => Err(format!("Invalid input: {suit}")),
    }
}

/// Attempts to convert a card into a string.
///
/// # Examples
/// ```
/// use postflop_solver::card_to_string;
///
/// assert_eq!(card_to_string(0), Ok("2c".to_string()));
/// assert_eq!(card_to_string(5), Ok("3d".to_string()));
/// assert_eq!(card_to_string(10), Ok("4h".to_string()));
/// assert_eq!(card_to_string(51), Ok("As".to_string()));
/// assert!(card_to_string(52).is_err());
/// ```
#[inline]
pub fn card_to_string(card: Card) -> Result<String, String> {
    check_card(card)?;
    let rank = card >> 2;
    let suit = card & 3;
    Ok(format!("{}{}", rank_to_char(rank)?, suit_to_char(suit)?))
}

/// Attempts to convert hole cards into a string.
///
/// See [`Card`] for encoding of cards.
/// The card order in the input does not matter, but the output string is sorted in descending order
/// of card IDs.
///
/// # Examples
/// ```
/// use postflop_solver::hole_to_string;
///
/// assert_eq!(hole_to_string((0, 5)), Ok("3d2c".to_string()));
/// assert_eq!(hole_to_string((10, 51)), Ok("As4h".to_string()));
/// assert!(hole_to_string((52, 53)).is_err());
/// ```
#[inline]
pub fn hole_to_string(hole: (Card, Card)) -> Result<String, String> {
    let max_card = Card::max(hole.0, hole.1);
    let min_card = Card::min(hole.0, hole.1);
    Ok(format!(
        "{}{}",
        card_to_string(max_card)?,
        card_to_string(min_card)?
    ))
}

/// Attempts to convert a list of hole cards into a list of strings.
///
/// See [`Card`] for encoding of cards.
/// The card order of each pair in the input does not matter, but the output string of each pair is
/// sorted in descending order of card IDs.
///
/// # Examples
/// ```
/// use postflop_solver::holes_to_strings;
///
/// assert_eq!(
///     holes_to_strings(&[(0, 5), (10, 51)]),
///     Ok(vec!["3d2c".to_string(), "As4h".to_string()])
/// );
/// assert!(holes_to_strings(&[(52, 53)]).is_err());
/// ```
#[inline]
pub fn holes_to_strings(holes: &[(Card, Card)]) -> Result<Vec<String>, String> {
    holes.iter().map(|&hole| hole_to_string(hole)).collect()
}

/// Attempts to read the next card from a char iterator.
///
/// # Examples
/// ```
/// use postflop_solver::card_from_chars;
///
/// let mut chars = "2c3d4hAs".chars();
/// assert_eq!(card_from_chars(&mut chars), Ok(0));
/// assert_eq!(card_from_chars(&mut chars), Ok(5));
/// assert_eq!(card_from_chars(&mut chars), Ok(10));
/// assert_eq!(card_from_chars(&mut chars), Ok(51));
/// assert!(card_from_chars(&mut chars).is_err());
/// ```
#[inline]
pub fn card_from_chars<T: Iterator<Item = char>>(chars: &mut T) -> Result<Card, String> {
    let rank_char = chars.next().ok_or_else(|| "Unexpected end".to_string())?;
    let suit_char = chars.next().ok_or_else(|| "Unexpected end".to_string())?;

    let rank = char_to_rank(rank_char)?;
    let suit = char_to_suit(suit_char)?;

    Ok((rank << 2) | suit)
}

/// Attempts to convert a string into a card.
///
/// # Examples
/// ```
/// use postflop_solver::card_from_str;
///
/// assert_eq!(card_from_str("2c"), Ok(0));
/// assert_eq!(card_from_str("3d"), Ok(5));
/// assert_eq!(card_from_str("4h"), Ok(10));
/// assert_eq!(card_from_str("As"), Ok(51));
/// ```
#[inline]
pub fn card_from_str(s: &str) -> Result<Card, String> {
    let mut chars = s.chars();
    let result = card_from_chars(&mut chars)?;

    if chars.next().is_some() {
        return Err("Expected exactly two characters".to_string());
    }

    Ok(result)
}

/// Attempts to convert an optionally space-separated string into a sorted flop array.
///
/// # Examples
/// ```
/// use postflop_solver::flop_from_str;
///
/// assert_eq!(flop_from_str("2c3d4h"), Ok([0, 5, 10]));
/// assert_eq!(flop_from_str("As Ah Ks"), Ok([47, 50, 51]));
/// assert!(flop_from_str("2c3d4h5s").is_err());
/// ```
#[inline]
pub fn flop_from_str(s: &str) -> Result<[Card; 3], String> {
    let mut result = [0; 3];
    let mut chars = s.chars();

    result[0] = card_from_chars(&mut chars)?;
    result[1] = card_from_chars(&mut chars.by_ref().skip_while(|c| c.is_whitespace()))?;
    result[2] = card_from_chars(&mut chars.by_ref().skip_while(|c| c.is_whitespace()))?;

    if chars.next().is_some() {
        return Err("Expected exactly three cards".to_string());
    }

    result.sort_unstable();

    if result[0] == result[1] || result[1] == result[2] {
        return Err("Cards must be unique".to_string());
    }

    Ok(result)
}

#[inline]
fn parse_singleton(combo: &str) -> Result<(u8, u8, Suitedness), String> {
    if combo.len() == 4 {
        parse_simple_singleton(combo)
    } else {
        parse_compound_singleton(combo)
    }
}

#[inline]
fn parse_simple_singleton(combo: &str) -> Result<(u8, u8, Suitedness), String> {
    let mut chars = combo.chars();
    let rank1 = char_to_rank(chars.next().ok_or_else(|| "Unexpected end".to_string())?)?;
    let suit1 = char_to_suit(chars.next().ok_or_else(|| "Unexpected end".to_string())?)?;
    let rank2 = char_to_rank(chars.next().ok_or_else(|| "Unexpected end".to_string())?)?;
    let suit2 = char_to_suit(chars.next().ok_or_else(|| "Unexpected end".to_string())?)?;
    if rank1 < rank2 {
        return Err(format!(
            "The first rank must be equal or higher than the second rank: {combo}"
        ));
    }
    if rank1 == rank2 && suit1 == suit2 {
        return Err(format!("Duplicate cards are not allowed: {combo}"));
    }
    Ok((rank1, rank2, Suitedness::Specific(suit1, suit2)))
}

#[inline]
fn parse_compound_singleton(combo: &str) -> Result<(u8, u8, Suitedness), String> {
    let mut chars = combo.chars();
    let rank1 = char_to_rank(chars.next().ok_or_else(|| "Unexpected end".to_string())?)?;
    let rank2 = char_to_rank(chars.next().ok_or_else(|| "Unexpected end".to_string())?)?;
    let suitedness = chars.next().map_or(Ok(Suitedness::All), |c| match c {
        's' => Ok(Suitedness::Suited),
        'o' => Ok(Suitedness::Offsuit),
        _ => Err(format!("Invalid suitedness: {combo}")),
    })?;
    if rank1 < rank2 {
        return Err(format!(
            "The first rank must be equal or higher than the second rank: {combo}"
        ));
    }
    if rank1 == rank2 && suitedness != Suitedness::All {
        return Err(format!("A pair with suitedness is not allowed: {combo}"));
    }
    Ok((rank1, rank2, suitedness))
}

#[inline]
fn check_card(card: Card) -> Result<(), String> {
    if card < 52 {
        Ok(())
    } else {
        Err(format!("Invalid card: {card}"))
    }
}

#[inline]
fn check_weight(weight: f32) -> Result<(), String> {
    if (0.0..=1.0).contains(&weight) {
        Ok(())
    } else {
        Err(format!("Invalid weight: {weight}"))
    }
}

impl Default for Range {
    #[inline]
    fn default() -> Self {
        Self {
            data: [0.0; 52 * 51 / 2],
        }
    }
}

impl Range {
    /// Creates an empty range.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a full range.
    #[inline]
    pub fn ones() -> Self {
        Self {
            data: [1.0; 52 * 51 / 2],
        }
    }

    /// Attempts to create a range from raw data.
    #[inline]
    pub fn from_raw_data(data: &[f32]) -> Result<Self, String> {
        if data.len() != 52 * 51 / 2 {
            return Err(format!("Expected exactly {} elements", 52 * 51 / 2));
        }

        for &weight in data {
            check_weight(weight)?;
        }

        Ok(Self {
            data: data.try_into().unwrap(),
        })
    }

    /// Obtains the raw data of the range.
    #[inline]
    pub fn raw_data(&self) -> &[f32] {
        &self.data
    }

    /// Attempts to create a range from a list of hands with their weights.
    #[inline]
    pub fn from_hands_weights(hands: &[(Card, Card)], weights: &[f32]) -> Result<Self, String> {
        let mut range = Self::default();
        for (&(card1, card2), &weight) in hands.iter().zip(weights.iter()) {
            check_card(card1)?;
            check_card(card2)?;
            check_weight(weight)?;
            if card1 == card2 {
                return Err("Hand must consist of two different cards".to_string());
            }
            range.set_weight_by_cards(card1, card2, weight);
        }
        Ok(range)
    }

    /// Returns a list of all hands in this range and their associated weights.
    ///
    /// If there are no dead cards, pass `0` to `dead_cards_mask`.
    /// The returned hands are sorted in lexicographical order.
    pub fn get_hands_weights(&self, dead_cards_mask: u64) -> (Vec<(Card, Card)>, Vec<f32>) {
        let mut hands = Vec::with_capacity(128);
        let mut weights = Vec::with_capacity(128);

        for card1 in 0..52 {
            for card2 in card1 + 1..52 {
                let hand_mask: u64 = (1 << card1) | (1 << card2);
                let weight = self.get_weight_by_cards(card1, card2);
                if weight > 0.0 && hand_mask & dead_cards_mask == 0 {
                    hands.push((card1, card2));
                    weights.push(weight);
                }
            }
        }

        hands.shrink_to_fit();
        weights.shrink_to_fit();

        (hands, weights)
    }

    /// Attempts to create a range from a sanitized range string.
    ///
    /// "Sanitized" means that the range string does not contain any invalid patterns and whitespace
    /// characters. Therefore, this method can bypass the regular expression processing. If you want
    /// to create a range from a regular string, use `parse::<Range>()` instead.
    pub fn from_sanitized_str(ranges: &str) -> Result<Self, String> {
        let mut ranges = ranges.split(',').collect::<Vec<_>>();

        // remove last empty element if any
        if ranges.last().unwrap().is_empty() {
            ranges.pop();
        }

        let mut result = Self::new();

        for range in ranges.into_iter().rev() {
            let mut split = range.split(':');
            let range = split.next().unwrap();

            let weight = split
                .next()
                .map_or(Ok(1.0), |s| s.parse::<f32>().map_err(|e| e.to_string()))?;
            check_weight(weight)?;

            if split.next().is_some() {
                return Err(format!("Invalid range: {range}"));
            }

            if range.contains('-') {
                result.update_with_dash_range(range, weight)?;
            } else if range.contains('+') {
                result.update_with_plus_range(range, weight)?;
            } else {
                result.update_with_singleton(range, weight)?;
            }
        }

        Ok(result)
    }

    /// Clears the range.
    #[inline]
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Returns whether the range is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|el| *el == 0.0)
    }

    /// Inverts the range.
    #[inline]
    pub fn invert(&mut self) {
        // we want to obtain 0.1 when the previous value was 0.9, not 0.100000024
        self.data
            .iter_mut()
            .for_each(|el| *el = (1.0 - el.to_string().parse::<f64>().unwrap()) as f32);
    }

    /// Obtains the weight of a specified hand.
    ///
    /// Undefined behavior if:
    ///   - `card1` or `card2` is not less than `52`
    ///   - `card1` is equal to `card2`
    #[inline]
    pub fn get_weight_by_cards(&self, card1: Card, card2: Card) -> f32 {
        self.data[card_pair_to_index(card1, card2)]
    }

    /// Obtains the average weight of specified pair hands.
    ///
    /// Undefined behavior if `rank` is not less than `13`.
    #[inline]
    pub fn get_weight_pair(&self, rank: u8) -> f32 {
        self.get_average_weight(&pair_indices(rank))
    }

    /// Obtains the average weight of specified suited hands.
    ///
    /// Undefined behavior if:
    ///   - `rank1` or `rank2` is not less than `13`
    ///   - `rank1` is equal to `rank2`
    #[inline]
    pub fn get_weight_suited(&self, rank1: u8, rank2: u8) -> f32 {
        self.get_average_weight(&suited_indices(rank1, rank2))
    }

    /// Obtains the average weight of specified offsuit hands.
    ///
    /// Undefined behavior if:
    ///   - `rank1` or `rank2` is not less than `13`
    ///   - `rank1` is equal to `rank2`
    #[inline]
    pub fn get_weight_offsuit(&self, rank1: u8, rank2: u8) -> f32 {
        self.get_average_weight(&offsuit_indices(rank1, rank2))
    }

    /// Sets the weight of a specified hand.
    ///
    /// Undefined behavior if:
    ///   - `card1` or `card2` is not less than `52`
    ///   - `card1` is equal to `card2`
    ///   - `weight` is not in the range `[0.0, 1.0]`
    #[inline]
    pub fn set_weight_by_cards(&mut self, card1: Card, card2: Card, weight: f32) {
        self.data[card_pair_to_index(card1, card2)] = weight;
    }

    /// Sets the weights of specified pair hands.
    ///
    /// Undefined behavior if:
    ///   - `rank` is not less than `13`
    ///   - `weight` is not in the range `[0.0, 1.0]`
    #[inline]
    pub fn set_weight_pair(&mut self, rank: u8, weight: f32) {
        self.set_weight(&pair_indices(rank), weight);
    }

    /// Sets the weights of specified suited hands.
    ///
    /// Undefined behavior if:
    ///   - `rank1` or `rank2` is not less than `13`
    ///   - `rank1` is equal to `rank2`
    ///   - `weight` is not in the range `[0.0, 1.0]`
    #[inline]
    pub fn set_weight_suited(&mut self, rank1: u8, rank2: u8, weight: f32) {
        self.set_weight(&suited_indices(rank1, rank2), weight);
    }

    /// Sets the weights of specified offsuit hands.
    ///
    /// Undefined behavior if:
    ///   - `rank1` or `rank2` is not less than `13`
    ///   - `rank1` is equal to `rank2`
    ///   - `weight` is not in the range `[0.0, 1.0]`
    #[inline]
    pub fn set_weight_offsuit(&mut self, rank1: u8, rank2: u8, weight: f32) {
        self.set_weight(&offsuit_indices(rank1, rank2), weight);
    }

    /// Returns whether the range is valid, i.e., all weights are in the range `[0.0, 1.0]`.
    #[inline]
    pub(crate) fn is_valid(&self) -> bool {
        self.data.iter().all(|el| (0.0..=1.0).contains(el))
    }

    /// Returns whether the all suits are symmetric.
    pub(crate) fn is_suit_symmetric(&self) -> bool {
        for rank1 in 0..13 {
            if !self.is_same_weight(&pair_indices(rank1)) {
                return false;
            }

            for rank2 in rank1 + 1..13 {
                if !self.is_same_weight(&suited_indices(rank1, rank2)) {
                    return false;
                }

                if !self.is_same_weight(&offsuit_indices(rank1, rank2)) {
                    return false;
                }
            }
        }

        true
    }

    /// Returns whether the two suits are isomorphic.
    pub(crate) fn is_suit_isomorphic(&self, suit1: u8, suit2: u8) -> bool {
        let replace_suit = |suit| {
            if suit == suit1 {
                suit2
            } else if suit == suit2 {
                suit1
            } else {
                suit
            }
        };

        for card1 in 0..52 {
            for card2 in card1 + 1..52 {
                let card1_replaced = (card1 & !3) | replace_suit(card1 & 3);
                let card2_replaced = (card2 & !3) | replace_suit(card2 & 3);
                let weight = self.get_weight_by_cards(card1, card2);
                let weight_replaced = self.get_weight_by_cards(card1_replaced, card2_replaced);
                if weight != weight_replaced {
                    return false;
                }
            }
        }

        true
    }

    #[inline]
    fn is_same_weight(&self, indices: &[usize]) -> bool {
        let weight = self.data[indices[0]];
        indices.iter().all(|&i| self.data[i] == weight)
    }

    #[inline]
    fn get_average_weight(&self, indices: &[usize]) -> f32 {
        let mut sum = 0.0;
        for &i in indices {
            sum += self.data[i] as f64;
        }
        (sum / indices.len() as f64) as f32
    }

    #[inline]
    fn set_weight(&mut self, indices: &[usize], weight: f32) {
        for &i in indices {
            self.data[i] = weight;
        }
    }

    #[inline]
    fn update_with_singleton(&mut self, combo: &str, weight: f32) -> Result<(), String> {
        let (rank1, rank2, suitedness) = parse_singleton(combo)?;
        self.set_weight(&indices_with_suitedness(rank1, rank2, suitedness), weight);
        Ok(())
    }

    #[inline]
    fn update_with_plus_range(&mut self, range: &str, weight: f32) -> Result<(), String> {
        let lowest_combo = &range[..range.len() - 1];
        let (rank1, rank2, suitedness) = parse_singleton(lowest_combo)?;
        let gap = rank1 - rank2;
        if gap <= 1 {
            // pair and connector (e.g.,  88+, T9s+)
            for i in rank1..13 {
                self.set_weight(&indices_with_suitedness(i, i - gap, suitedness), weight);
            }
        } else {
            // otherwise (e.g., ATo+)
            for i in rank2..rank1 {
                self.set_weight(&indices_with_suitedness(rank1, i, suitedness), weight);
            }
        }
        Ok(())
    }

    #[inline]
    fn update_with_dash_range(&mut self, range: &str, weight: f32) -> Result<(), String> {
        let combo_pair = range.split('-').collect::<Vec<_>>();
        let (rank11, rank12, suitedness) = parse_singleton(combo_pair[0])?;
        let (rank21, rank22, suitedness2) = parse_singleton(combo_pair[1])?;
        let gap = rank11 - rank12;
        let gap2 = rank21 - rank22;
        if suitedness != suitedness2 {
            Err(format!("Suitedness does not match: {range}"))
        } else if gap == gap2 {
            // same gap (e.g., 88-55, KQo-JTo)
            if rank11 > rank21 {
                for i in rank21..=rank11 {
                    self.set_weight(&indices_with_suitedness(i, i - gap, suitedness), weight);
                }
                Ok(())
            } else {
                Err(format!("Range must be in descending order: {range}"))
            }
        } else if rank11 == rank21 {
            // same first rank (e.g., A5s-A2s)
            if rank12 > rank22 {
                for i in rank22..=rank12 {
                    self.set_weight(&indices_with_suitedness(rank11, i, suitedness), weight);
                }
                Ok(())
            } else {
                Err(format!("Range must be in descending order: {range}"))
            }
        } else {
            Err(format!("Invalid range: {range}"))
        }
    }

    fn pairs_strings(&self, result: &mut Vec<String>) {
        let mut start: Option<(u8, f32)> = None;

        for i in (-1..13).rev() {
            let rank = i as u8;
            let prev_rank = (i + 1) as u8;

            if start.is_some()
                && (i == -1
                    || !self.is_same_weight(&pair_indices(rank))
                    || start.unwrap().1 != self.get_weight_pair(rank))
            {
                let (start_rank, weight) = start.unwrap();
                let s = rank_to_char(start_rank).unwrap();
                let e = rank_to_char(prev_rank).unwrap();
                let mut tmp = if start_rank == prev_rank {
                    format!("{s}{s}")
                } else if start_rank == 12 {
                    format!("{e}{e}+")
                } else {
                    format!("{s}{s}-{e}{e}")
                };
                if weight != 1.0 {
                    write!(tmp, ":{weight}").unwrap();
                }
                result.push(tmp);
                start = None;
            }

            if i >= 0
                && self.is_same_weight(&pair_indices(rank))
                && self.get_weight_pair(rank) > 0.0
                && start.is_none()
            {
                start = Some((rank, self.get_weight_pair(rank)));
            }
        }
    }

    fn nonpairs_strings(&self, result: &mut Vec<String>) {
        for rank1 in (1..13).rev() {
            if self.can_unsuit(rank1) {
                self.high_cards_strings(result, rank1, Suitedness::All);
            } else {
                self.high_cards_strings(result, rank1, Suitedness::Suited);
                self.high_cards_strings(result, rank1, Suitedness::Offsuit);
            }
        }
    }

    fn can_unsuit(&self, rank1: u8) -> bool {
        for rank2 in 0..rank1 {
            let same_suited = self.is_same_weight(&suited_indices(rank1, rank2));
            let same_offsuit = self.is_same_weight(&offsuit_indices(rank1, rank2));
            let weight_suited = self.get_weight_suited(rank1, rank2);
            let weight_offsuit = self.get_weight_offsuit(rank1, rank2);
            if (same_suited && same_offsuit && weight_suited != weight_offsuit)
                || (same_suited != same_offsuit && weight_suited > 0.0 && weight_offsuit > 0.0)
            {
                return false;
            }
        }
        true
    }

    fn high_cards_strings(&self, result: &mut Vec<String>, rank1: u8, suitedness: Suitedness) {
        let rank1_char = rank_to_char(rank1).unwrap();
        let mut start: Option<(u8, f32)> = None;
        type FnPairToIndices = fn(u8, u8) -> Vec<usize>;
        let (getter, suit_char): (FnPairToIndices, &str) = match suitedness {
            Suitedness::Suited => (suited_indices, "s"),
            Suitedness::Offsuit => (offsuit_indices, "o"),
            Suitedness::All => (nonpair_indices, ""),
            _ => panic!("high_cards_strings: invalid suitedness"),
        };

        for i in (-1..rank1 as i32).rev() {
            let rank2 = i as u8;
            let prev_rank2 = (i + 1) as u8;

            if start.is_some()
                && (i == -1
                    || !self.is_same_weight(&getter(rank1, rank2))
                    || start.unwrap().1 != self.get_average_weight(&getter(rank1, rank2)))
            {
                let (start_rank2, weight) = start.unwrap();
                let s = rank_to_char(start_rank2).unwrap();
                let e = rank_to_char(prev_rank2).unwrap();
                let mut tmp = if start_rank2 == prev_rank2 {
                    format!("{rank1_char}{s}{suit_char}")
                } else if start_rank2 == rank1 - 1 {
                    format!("{rank1_char}{e}{suit_char}+")
                } else {
                    format!("{rank1_char}{s}{suit_char}-{rank1_char}{e}{suit_char}")
                };
                if weight != 1.0 {
                    write!(tmp, ":{weight}").unwrap();
                }
                result.push(tmp);
                start = None;
            }

            if i >= 0
                && self.is_same_weight(&getter(rank1, rank2))
                && self.get_average_weight(&getter(rank1, rank2)) > 0.0
                && start.is_none()
            {
                start = Some((rank2, self.get_average_weight(&getter(rank1, rank2))));
            }
        }
    }

    fn suit_specified_strings(&self, result: &mut Vec<String>) {
        // pairs
        for rank in (0..13).rev() {
            if !self.is_same_weight(&pair_indices(rank)) {
                for suit1 in (0..4).rev() {
                    for suit2 in (0..suit1).rev() {
                        let weight = self.get_weight_by_cards(4 * rank + suit1, 4 * rank + suit2);
                        if weight > 0.0 {
                            let mut tmp = format!(
                                "{rank}{suit1}{rank}{suit2}",
                                rank = rank_to_char(rank).unwrap(),
                                suit1 = suit_to_char(suit1).unwrap(),
                                suit2 = suit_to_char(suit2).unwrap(),
                            );
                            if weight != 1.0 {
                                write!(tmp, ":{weight}").unwrap();
                            }
                            result.push(tmp);
                        }
                    }
                }
            }
        }

        // non-pairs
        for rank1 in (0..13).rev() {
            for rank2 in (0..rank1).rev() {
                // suited
                if !self.is_same_weight(&suited_indices(rank1, rank2)) {
                    for suit in (0..4).rev() {
                        let weight = self.get_weight_by_cards(4 * rank1 + suit, 4 * rank2 + suit);
                        if weight > 0.0 {
                            let mut tmp = format!(
                                "{rank1}{suit}{rank2}{suit}",
                                rank1 = rank_to_char(rank1).unwrap(),
                                rank2 = rank_to_char(rank2).unwrap(),
                                suit = suit_to_char(suit).unwrap(),
                            );
                            if weight != 1.0 {
                                write!(tmp, ":{weight}").unwrap();
                            }
                            result.push(tmp);
                        }
                    }
                }

                // offsuit
                if !self.is_same_weight(&offsuit_indices(rank1, rank2)) {
                    for suit1 in (0..4).rev() {
                        for suit2 in (0..4).rev() {
                            if suit1 != suit2 {
                                let weight =
                                    self.get_weight_by_cards(4 * rank1 + suit1, 4 * rank2 + suit2);
                                if weight > 0.0 {
                                    let mut tmp = format!(
                                        "{rank1}{suit1}{rank2}{suit2}",
                                        rank1 = rank_to_char(rank1).unwrap(),
                                        suit1 = suit_to_char(suit1).unwrap(),
                                        rank2 = rank_to_char(rank2).unwrap(),
                                        suit2 = suit_to_char(suit2).unwrap(),
                                    );
                                    if weight != 1.0 {
                                        write!(tmp, ":{weight}").unwrap();
                                    }
                                    result.push(tmp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl FromStr for Range {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = TRIM_REGEX.replace_all(s, "$1").trim().to_string();
        let mut ranges = s.split(',').collect::<Vec<_>>();

        // remove last empty element if any
        if ranges.last().unwrap().is_empty() {
            ranges.pop();
        }

        let mut result = Self::new();

        for range in ranges.into_iter().rev() {
            let caps = RANGE_REGEX
                .captures(range)
                .ok_or_else(|| format!("Failed to parse range: {range}"))?;

            let range = caps.name("range").unwrap().as_str();
            let weight = caps
                .name("weight")
                .map_or(1.0, |s| s.as_str().parse().unwrap());
            check_weight(weight)?;

            if range.contains('-') {
                result.update_with_dash_range(range, weight)?;
            } else if range.contains('+') {
                result.update_with_plus_range(range, weight)?;
            } else {
                result.update_with_singleton(range, weight)?;
            }
        }

        Ok(result)
    }
}

impl ToString for Range {
    #[inline]
    fn to_string(&self) -> String {
        let mut result = Vec::new();
        self.pairs_strings(&mut result);
        self.nonpairs_strings(&mut result);
        self.suit_specified_strings(&mut result);
        result.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_regex() {
        let tests = [
            ("AK", Some(("AK", None))),
            ("ak", Some(("ak", None))),
            ("K9s:.67", Some(("K9s", Some(".67")))),
            ("88+:1.", Some(("88+", Some("1.")))),
            ("98s-65s:0.25", Some(("98s-65s", Some("0.25")))),
            ("AcKh", Some(("AcKh", None))),
            ("8h8s+:.67", Some(("8h8s+", Some(".67")))),
            ("9d8d-6d5d:0.25", Some(("9d8d-6d5d", Some("0.25")))),
            ("AKQ", None),
            ("AK+-AJ", None),
            ("K9s.67", None),
            ("88+:2.0", None),
            ("98s-21s", None),
        ];

        for (s, expected) in tests {
            if let Some((range, weight)) = expected {
                let caps = RANGE_REGEX.captures(s).unwrap();
                assert_eq!(caps.name("range").unwrap().as_str(), range);
                if let Some(weight) = weight {
                    assert_eq!(caps.name("weight").unwrap().as_str(), weight);
                } else {
                    assert!(caps.name("weight").is_none());
                }
            } else {
                assert!(!RANGE_REGEX.is_match(s));
            }
        }
    }

    #[test]
    fn trim_regex() {
        let tests = [
            ("  AK  ", "AK"),
            ("K9s: .67", "K9s:.67"),
            ("88+, AQ+", "88+,AQ+"),
            ("98s - 65s: 0.25", "98s-65s:0.25"),
        ];

        for (s, expected) in tests {
            assert_eq!(TRIM_REGEX.replace_all(s, "$1").trim(), expected);
        }
    }

    #[test]
    fn range_from_str() {
        let pair_plus = "88+".parse::<Range>();
        let pair_plus_equiv = "AA,KK,QQ,JJ,TT,99,88".parse::<Range>();
        assert!(pair_plus.is_ok());
        assert_eq!(pair_plus, pair_plus_equiv);

        let pair_plus_suit = "8s8h+".parse::<Range>();
        let pair_plus_suit_equiv = "AhAs,KhKs,QhQs,JhJs,ThTs,9h9s,8h8s".parse::<Range>();
        assert!(pair_plus_suit.is_ok());
        assert_eq!(pair_plus_suit, pair_plus_suit_equiv);

        let connector_plus = "98s+".parse::<Range>();
        let connector_plus_equiv = "AKs,KQs,QJs,JTs,T9s,98s".parse::<Range>();
        assert!(connector_plus.is_ok());
        assert_eq!(connector_plus, connector_plus_equiv);

        let other_plus = "A8o+".parse::<Range>();
        let other_plus_equiv = "AKo,AQo,AJo,ATo,A9o,A8o".parse::<Range>();
        assert!(other_plus.is_ok());
        assert_eq!(other_plus, other_plus_equiv);

        let pair_dash = "88-55".parse::<Range>();
        let pair_dash_equiv = "88,77,66,55".parse::<Range>();
        assert!(pair_dash.is_ok());
        assert_eq!(pair_dash, pair_dash_equiv);

        let connector_dash = "98s-65s".parse::<Range>();
        let connector_dash_equiv = "98s,87s,76s,65s".parse::<Range>();
        assert!(connector_dash.is_ok());
        assert_eq!(connector_dash, connector_dash_equiv);

        let gapper_dash = "AQo-86o".parse::<Range>();
        let gapper_dash_equiv = "AQo,KJo,QTo,J9o,T8o,97o,86o".parse::<Range>();
        assert!(gapper_dash.is_ok());
        assert_eq!(gapper_dash, gapper_dash_equiv);

        let other_dash = "K5-K2".parse::<Range>();
        let other_dash_equiv = "K5,K4,K3,K2".parse::<Range>();
        assert!(other_dash.is_ok());
        assert_eq!(other_dash, other_dash_equiv);

        let suit_compound = "AhAs-QhQs,JJ".parse::<Range>();
        let suit_compound_equiv = "JJ,AhAs,KhKs,QhQs".parse::<Range>();
        assert!(suit_compound.is_ok());
        assert_eq!(suit_compound, suit_compound_equiv);

        let allow_empty = "".parse::<Range>();
        assert!(allow_empty.is_ok());

        let allow_trailing_comma = "AK,".parse::<Range>();
        assert!(allow_trailing_comma.is_ok());

        let comma_error = "AK,,".parse::<Range>();
        assert!(comma_error.is_err());

        let rank_error = "89".parse::<Range>();
        assert!(rank_error.is_err());

        let pair_error = "AAo".parse::<Range>();
        assert!(pair_error.is_err());

        let weight_error = "AQo:1.1".parse::<Range>();
        assert!(weight_error.is_err());

        let dash_error_1 = "AQo-AQo".parse::<Range>();
        assert!(dash_error_1.is_err());

        let dash_error_2 = "AQo-86s".parse::<Range>();
        assert!(dash_error_2.is_err());

        let dash_error_3 = "AQo-KQo".parse::<Range>();
        assert!(dash_error_3.is_err());

        let dash_error_4 = "K2-K5".parse::<Range>();
        assert!(dash_error_4.is_err());

        let dash_error_5 = "AhAs-QsQh".parse::<Range>();
        assert!(dash_error_5.is_err());

        let data = "85s:0.5".parse::<Range>();
        assert!(data.is_ok());

        let data = data.unwrap();
        assert_eq!(data.get_weight_suited(3, 6), 0.5);
        assert_eq!(data.get_weight_suited(6, 3), 0.5);
        assert_eq!(data.get_weight_offsuit(3, 6), 0.0);
        assert_eq!(data.get_weight_offsuit(6, 3), 0.0);
    }

    #[test]
    fn range_to_string() {
        let tests = [
            ("AA,KK", "KK+"),
            ("KK,QQ", "KK-QQ"),
            ("66-22,TT+", "TT+,66-22"),
            ("AA:0.5, KK:1.0, QQ:1.0, JJ:0.5", "AA:0.5,KK-QQ,JJ:0.5"),
            ("AA,AK,AQ", "AA,AQ+"),
            ("AK,AQ,AJs", "AJs+,AQo+"),
            ("KQ,KT,K9,K8,K6,K5", "KQ,KT-K8,K6-K5"),
            ("AhAs-QhQs,JJ", "JJ,AsAh,KsKh,QsQh"),
            ("KJs+,KQo,KsJh", "KJs+,KQo,KsJh"),
            ("KcQh,KJ", "KJ,KcQh"),
        ];

        for (input, expected) in tests {
            let range = input.parse::<Range>();
            assert!(range.is_ok());
            assert_eq!(range.unwrap().to_string(), expected);
        }
    }
}
