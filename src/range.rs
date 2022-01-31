use once_cell::sync::Lazy;
use regex::Regex;
use std::str::FromStr;

/// A struct representing a player's 13x13 range.
///
/// ```
/// use postflop_solver::*;
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
/// assert_eq!(range.get_prob_pair(queen_rank), 1.0);
///
/// // check that the hand "AKo" is not in the range
/// assert_eq!(range.get_prob_offsuit(ace_rank, king_rank), 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Range {
    data: [[f32; 13]; 13],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Suitedness {
    Suited,
    Offsuit,
    Both,
}

const COMBO_PAT: &str = r"(?:[AKQJT2-9]{2}[os]?)";
const PROB_PAT: &str = r"(?:(?:[01](\.\d*)?)|(?:\.\d+))";

static RANGE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(&format!(
        r"^(?P<range>{COMBO_PAT}(?:\+|(?:-{COMBO_PAT}))?)(?::(?P<prob>{PROB_PAT}))?$"
    ))
    .unwrap()
});

static TRIM_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*([-:,])\s*").unwrap());

/// Attempts to convert a rank character to a rank index.
/// `'A'` => `12`, `'K'` => `11`, ..., `'2'` => `0`.
pub fn char_to_rank(c: char) -> Result<u8, String> {
    match c {
        'A' => Ok(12),
        'K' => Ok(11),
        'Q' => Ok(10),
        'J' => Ok(9),
        'T' => Ok(8),
        '2'..='9' => Ok(c as u8 - b'2'),
        _ => Err(format!("invalid input: {c}")),
    }
}

/// Attempts to convert a rank index to a rank character.
/// `12` => `'A'`, `11` => `'K'`, ..., `0` => `'2'`.
pub fn rank_to_char(rank: u8) -> Result<char, String> {
    match rank {
        12 => Ok('A'),
        11 => Ok('K'),
        10 => Ok('Q'),
        9 => Ok('J'),
        8 => Ok('T'),
        0..=7 => Ok((rank + b'2') as char),
        _ => Err(format!("invalid input: {rank}")),
    }
}

fn parse_singleton(combo: &str) -> Result<(u8, u8, Suitedness), String> {
    let mut chars = combo.chars();
    let rank1 = char_to_rank(chars.next().unwrap())?;
    let rank2 = char_to_rank(chars.next().unwrap())?;
    let suit = chars.next().map_or(Suitedness::Both, |c| match c {
        's' => Suitedness::Suited,
        'o' => Suitedness::Offsuit,
        _ => panic!("parse_singleton: invalid suitedness: {combo}"),
    });
    if rank1 < rank2 {
        return Err(format!(
            "First rank must be equal or higher than second rank: {combo}"
        ));
    }
    if rank1 == rank2 && suit != Suitedness::Both {
        return Err(format!("Pair with suitedness is not allowed: {combo}"));
    }
    Ok((rank1, rank2, suit))
}

fn check_rank(rank: u8) -> Result<(), String> {
    if rank < 13 {
        Ok(())
    } else {
        Err(format!("Invalid rank: {rank}"))
    }
}

fn check_prob(prob: f32) -> Result<(), String> {
    if (0.0..=1.0).contains(&prob) {
        Ok(())
    } else {
        Err(format!("Invalid probability: {prob}"))
    }
}

impl Range {
    /// Creates an empty range.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a full range.
    pub fn ones() -> Self {
        Self {
            data: [[1.0; 13]; 13],
        }
    }

    /// Obtains the probability by card indices.
    ///
    /// The indices are defined as follows:
    /// 2c2d2h2s => `0-3`, 3c3d3h3s => `4-7`, ..., AcAdAhAs => `48-51`.
    pub fn get_prob_by_cards(&self, card1: u8, card2: u8) -> f32 {
        let (rank1, suit1) = (card1 >> 2, card1 & 3);
        let (rank2, suit2) = (card2 >> 2, card2 & 3);
        if rank1 == rank2 {
            self.get_prob_pair(rank1)
        } else if suit1 == suit2 {
            self.get_prob_suited(rank1, rank2)
        } else {
            self.get_prob_offsuit(rank1, rank2)
        }
    }

    /// Obtains the probability of a pair.
    pub fn get_prob_pair(&self, rank: u8) -> f32 {
        self.data[rank as usize][rank as usize]
    }

    /// Obtains the probability of a suited hand.
    pub fn get_prob_suited(&self, rank1: u8, rank2: u8) -> f32 {
        if rank1 > rank2 {
            self.data[rank1 as usize][rank2 as usize]
        } else {
            self.data[rank2 as usize][rank1 as usize]
        }
    }

    /// Obtains the probability of an offsuit hand.
    pub fn get_prob_offsuit(&self, rank1: u8, rank2: u8) -> f32 {
        if rank1 < rank2 {
            self.data[rank1 as usize][rank2 as usize]
        } else {
            self.data[rank2 as usize][rank1 as usize]
        }
    }

    /// Sets the probability of a pair.
    pub fn set_data_pair(&mut self, rank: u8, prob: f32) -> Result<(), String> {
        check_rank(rank)?;
        check_prob(prob)?;
        self.data[rank as usize][rank as usize] = prob;
        Ok(())
    }

    /// Sets the probability of a suited hand.
    pub fn set_data_suited(&mut self, rank1: u8, rank2: u8, prob: f32) -> Result<(), String> {
        check_rank(rank1)?;
        check_rank(rank2)?;
        check_prob(prob)?;
        if rank1 == rank2 {
            return Err(format!(
                "set_data_suited() accepts non-pairs, but got rank1 = rank2 = {rank1}"
            ));
        }
        if rank1 > rank2 {
            self.data[rank1 as usize][rank2 as usize] = prob;
        } else {
            self.data[rank2 as usize][rank1 as usize] = prob;
        }
        Ok(())
    }

    /// Sets the probability of an offsuit hand.
    pub fn set_data_offsuit(&mut self, rank1: u8, rank2: u8, prob: f32) -> Result<(), String> {
        check_rank(rank1)?;
        check_rank(rank2)?;
        check_prob(prob)?;
        if rank1 == rank2 {
            return Err(format!(
                "set_data_offsuit() accepts non-pairs, but got rank1 = rank2 = {rank1}"
            ));
        }
        if rank1 < rank2 {
            self.data[rank1 as usize][rank2 as usize] = prob;
        } else {
            self.data[rank2 as usize][rank1 as usize] = prob;
        }
        Ok(())
    }

    /// Returns whether the range is empty.
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|row| row.iter().all(|&x| x == 0.0))
    }

    fn update(&mut self, rank1: u8, rank2: u8, suit: Suitedness, prob: f32) {
        debug_assert!(rank1 >= rank2);
        if suit != Suitedness::Offsuit {
            self.data[rank1 as usize][rank2 as usize] = prob;
        }
        if rank1 != rank2 && suit != Suitedness::Suited {
            self.data[rank2 as usize][rank1 as usize] = prob;
        }
    }

    fn update_with_singleton(&mut self, combo: &str, prob: f32) -> Result<(), String> {
        let (rank1, rank2, suit) = parse_singleton(combo)?;
        self.update(rank1, rank2, suit, prob);
        Ok(())
    }

    fn update_with_plus_range(&mut self, range: &str, prob: f32) -> Result<(), String> {
        debug_assert!(range.ends_with('+'));
        let lowest_combo = &range[..range.len() - 1];
        let (rank1, rank2, suit) = parse_singleton(lowest_combo)?;
        let gap = rank1 - rank2;
        if gap <= 1 {
            // pair and connector (e.g.,  88+, T9s+)
            for i in rank1..13 {
                self.update(i, i - gap, suit, prob);
            }
        } else {
            // otherwise (e.g., ATo+)
            for i in rank2..rank1 {
                self.update(rank1, i, suit, prob);
            }
        }
        Ok(())
    }

    fn update_with_dash_range(&mut self, range: &str, prob: f32) -> Result<(), String> {
        let combo_pair = range.split('-').collect::<Vec<_>>();
        debug_assert!(combo_pair.len() == 2);
        let (rank11, rank12, suit1) = parse_singleton(combo_pair[0])?;
        let (rank21, rank22, suit2) = parse_singleton(combo_pair[1])?;
        let gap1 = rank11 - rank12;
        let gap2 = rank21 - rank22;
        if suit1 != suit2 {
            Err(format!("Suitedness does not match: {range}"))
        } else if gap1 == gap2 && rank11 > rank21 {
            // same gap (e.g., 88-55, KQo-JTo)
            for i in rank21..=rank11 {
                self.update(i, i - gap1, suit1, prob);
            }
            Ok(())
        } else if rank11 == rank21 && rank12 > rank22 {
            // same first rank (e.g., A5s-A2s)
            for i in rank22..=rank12 {
                self.update(rank11, i, suit1, prob);
            }
            Ok(())
        } else {
            Err(format!("Invalid range: {range}"))
        }
    }

    fn pairs_strings(&self) -> Vec<String> {
        let mut result = Vec::new();
        let mut start: Option<(u8, f32)> = None;

        for i in (-1..13).rev() {
            let rank = i as u8;
            let prev_rank = (i + 1) as u8;

            if start.is_some() && (i == -1 || start.unwrap().1 != self.get_prob_pair(rank)) {
                let (start_rank, prob) = start.unwrap();
                let s = rank_to_char(start_rank).unwrap();
                let e = rank_to_char(prev_rank).unwrap();
                let mut tmp = if start_rank == prev_rank {
                    format!("{s}{s}")
                } else if start_rank == 12 {
                    format!("{e}{e}+")
                } else {
                    format!("{s}{s}-{e}{e}")
                };
                if prob != 1.0 {
                    tmp += &format!(":{prob}");
                }
                result.push(tmp);
                start = None;
            }

            if i >= 0 && self.get_prob_pair(rank) > 0.0 && start.is_none() {
                start = Some((rank, self.get_prob_pair(rank)));
            }
        }

        result
    }

    fn nonpairs_strings(&self) -> Vec<String> {
        let mut result = Vec::new();

        for rank1 in (1..13).rev() {
            let rank1_usize = rank1 as usize;
            let mut unsuit = true;
            for rank2 in 0..rank1 {
                if self.get_prob_suited(rank1, rank2) != self.get_prob_offsuit(rank1, rank2) {
                    unsuit = false;
                    break;
                }
            }

            if unsuit {
                Self::high_cards_strings(
                    &mut result,
                    rank1,
                    &self.data[rank1_usize][..rank1_usize],
                    "",
                );
            } else {
                let offsuit = self.data[..rank1_usize]
                    .iter()
                    .map(|row| row[rank1_usize])
                    .collect::<Vec<_>>();
                Self::high_cards_strings(
                    &mut result,
                    rank1,
                    &self.data[rank1_usize][..rank1_usize],
                    "s",
                );
                Self::high_cards_strings(&mut result, rank1, &offsuit, "o");
            }
        }

        result
    }

    fn high_cards_strings(result: &mut Vec<String>, rank1: u8, data: &[f32], suit: &str) {
        let rank1_char = rank_to_char(rank1).unwrap();
        let mut start: Option<(u8, f32)> = None;

        for i in (-1..rank1 as i32).rev() {
            let rank2 = i as u8;
            let prev_rank2 = (i + 1) as u8;

            if start.is_some() && (i == -1 || start.unwrap().1 != data[rank2 as usize]) {
                let (start_rank2, prob) = start.unwrap();
                let s = rank_to_char(start_rank2).unwrap();
                let e = rank_to_char(prev_rank2).unwrap();
                let mut tmp = if start_rank2 == prev_rank2 {
                    format!("{rank1_char}{s}{suit}")
                } else if start_rank2 == rank1 - 1 {
                    format!("{rank1_char}{e}{suit}+")
                } else {
                    format!("{rank1_char}{s}{suit}-{rank1_char}{e}{suit}")
                };
                if prob != 1.0 {
                    tmp += &format!(":{prob}");
                }
                result.push(tmp);
                start = None;
            }

            if i >= 0 && data[rank2 as usize] > 0.0 && start.is_none() {
                start = Some((rank2, data[rank2 as usize]));
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

        let mut result = Self {
            data: [[0.0; 13]; 13],
        };

        for range in ranges.into_iter().rev() {
            let caps = RANGE_REGEX
                .captures(range)
                .ok_or_else(|| format!("Failed to parse range: {range}"))?;

            let range_str = caps.name("range").unwrap().as_str();
            let prob = caps
                .name("prob")
                .map_or(1.0, |s| s.as_str().parse().unwrap());
            check_prob(prob)?;

            if range_str.contains('-') {
                result.update_with_dash_range(range_str, prob)?;
            } else if range_str.contains('+') {
                result.update_with_plus_range(range_str, prob)?;
            } else {
                result.update_with_singleton(range_str, prob)?;
            }
        }

        Ok(result)
    }
}

impl ToString for Range {
    fn to_string(&self) -> String {
        let mut pairs = self.pairs_strings();
        let mut nonpairs = self.nonpairs_strings();
        pairs.append(&mut nonpairs);
        pairs.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_regex() {
        let tests = [
            ("AK", Some(("AK", None))),
            ("K9s:.67", Some(("K9s", Some(".67")))),
            ("88+:1.", Some(("88+", Some("1.")))),
            ("98s-65s:0.25", Some(("98s-65s", Some("0.25")))),
            ("ak", None),
            ("AKQ", None),
            ("AK+-AJ", None),
            ("K9s.67", None),
            ("88+:2.0", None),
            ("98s-21s", None),
        ];

        for (s, expected) in tests {
            if let Some((range, prob)) = expected {
                let caps = RANGE_REGEX.captures(s).unwrap();
                assert_eq!(caps.name("range").unwrap().as_str(), range);
                if let Some(prob) = prob {
                    assert_eq!(caps.name("prob").unwrap().as_str(), prob);
                } else {
                    assert!(caps.name("prob").is_none());
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

        let prob_error = "AQo:1.1".parse::<Range>();
        assert!(prob_error.is_err());

        let dash_error_1 = "AQo-AQo".parse::<Range>();
        assert!(dash_error_1.is_err());

        let dash_error_2 = "AQo-86s".parse::<Range>();
        assert!(dash_error_2.is_err());

        let dash_error_3 = "AQo-KQo".parse::<Range>();
        assert!(dash_error_3.is_err());

        let dash_error_4 = "K2-K5".parse::<Range>();
        assert!(dash_error_4.is_err());

        let data = "85s:0.5".parse::<Range>();
        assert!(data.is_ok());

        let data = data.unwrap();
        assert_eq!(data.get_prob_suited(3, 6), 0.5);
        assert_eq!(data.get_prob_suited(6, 3), 0.5);
        assert_eq!(data.get_prob_offsuit(3, 6), 0.0);
        assert_eq!(data.get_prob_offsuit(6, 3), 0.0);
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
        ];

        for (input, expected) in tests {
            let range = input.parse::<Range>();
            assert!(range.is_ok());
            assert_eq!(range.unwrap().to_string(), expected);
        }

        let mut data = [
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
            [1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
            [1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.5, 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        ];

        data.reverse();
        for row in &mut data {
            row.reverse();
        }

        let range = Range { data };
        assert_eq!(
            range.to_string(),
            "22+,A2+,K2+,Q2s+,Q7o+,J3s+,J8o+,T4s+,T8o+,95s+,97o+,84s+,87o,74s+,76o,64s+,53s+,43s:0.5"
        );
    }
}
