use once_cell::sync::Lazy;
use regex::Regex;

const FLOAT_PAT: &str = r"(?P<float>(?:[1-9]\d*(?:\.\d*)?)|(?:0?\.\d+))";

static SIZE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(&format!(r"^({FLOAT_PAT}[%x]?)$")).unwrap());
static TRIM_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*(,)\s*").unwrap());

/// Bet size candidates for first bet and raise.
///
/// # Examples
/// ```
/// use postflop_solver::BetSizeCandidates;
/// use postflop_solver::BetSize::{PotRelative, LastBetRelative};
///
/// let bet_size = BetSizeCandidates::try_from(("0.5", "75%, 2.5x")).unwrap();
///
/// assert_eq!(bet_size.bet, vec![PotRelative(0.5)]);
/// assert_eq!(bet_size.raise, vec![PotRelative(0.75), LastBetRelative(2.5)]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BetSizeCandidates {
    /// Bet size candidates for first bet.
    pub bet: Vec<BetSize>,

    /// Bet size candidates for raise.
    pub raise: Vec<BetSize>,
}

/// Bet size specification.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum BetSize {
    /// Bet size is relative to the current pot size.
    PotRelative(f32),

    /// Bet size is relative to the last bet size. This is only valid for raise actions.
    LastBetRelative(f32),
}

impl TryFrom<(&str, &str)> for BetSizeCandidates {
    type Error = String;

    /// Attempts to convert comma-separated strings into bet sizes.
    ///
    /// # Examples
    /// ```
    /// use postflop_solver::BetSizeCandidates;
    /// use postflop_solver::BetSize::{PotRelative, LastBetRelative};
    ///
    /// let bet_size = BetSizeCandidates::try_from(("0.5", "75%, 2.5x")).unwrap();
    ///
    /// assert_eq!(bet_size.bet, vec![PotRelative(0.5)]);
    /// assert_eq!(bet_size.raise, vec![PotRelative(0.75), LastBetRelative(2.5)]);
    /// ```
    fn try_from((bet_str, raise_str): (&str, &str)) -> Result<Self, Self::Error> {
        let bet_string = TRIM_REGEX.replace_all(bet_str, "$1").trim().to_string();
        let mut bet_sizes = bet_string.split(',').collect::<Vec<_>>();

        let raise_string = TRIM_REGEX.replace_all(raise_str, "$1").trim().to_string();
        let mut raise_sizes = raise_string.split(',').collect::<Vec<_>>();

        if bet_sizes.last().unwrap().is_empty() {
            bet_sizes.pop();
        }

        if raise_sizes.last().unwrap().is_empty() {
            raise_sizes.pop();
        }

        let mut bet = Vec::new();
        let mut raise = Vec::new();

        for bet_size in bet_sizes {
            bet.push(bet_size_from_str(bet_size, false)?);
        }

        for raise_size in raise_sizes {
            raise.push(bet_size_from_str(raise_size, true)?);
        }

        bet.sort_unstable_by(|l, r| l.partial_cmp(r).unwrap());
        raise.sort_unstable_by(|l, r| l.partial_cmp(r).unwrap());

        Ok(BetSizeCandidates { bet, raise })
    }
}

fn bet_size_from_str(s: &str, allow_last_bet_rel: bool) -> Result<BetSize, String> {
    let caps = SIZE_REGEX
        .captures(s)
        .ok_or_else(|| format!("failed to parse bet size: {s}"))?;
    let float = caps.name("float").unwrap().as_str().parse::<f32>().unwrap();
    match caps.get(1).unwrap().as_str().chars().last().unwrap() {
        '%' => {
            if float <= 1000.0 {
                Ok(BetSize::PotRelative(float / 100.0))
            } else {
                Err(format!("bet size too large: {s}"))
            }
        }
        'x' => {
            if !allow_last_bet_rel {
                Err(format!("last bet relative not allowed: {s}"))
            } else if float < 2.0 {
                Err(format!("bet size too small: {s}"))
            } else if float > 10.0 {
                Err(format!("bet size too large: {s}"))
            } else {
                Ok(BetSize::LastBetRelative(float))
            }
        }
        _ => {
            if float <= 10.0 {
                Ok(BetSize::PotRelative(float))
            } else {
                Err(format!("bet size too large: {s}"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bet_size_from_str() {
        let tests = [
            (".75", BetSize::PotRelative(0.75)),
            ("100%", BetSize::PotRelative(1.0)),
            ("112.5%", BetSize::PotRelative(1.125)),
            ("3.5x", BetSize::LastBetRelative(3.5)),
        ];

        for (s, expected) in tests {
            assert_eq!(bet_size_from_str(s, true), Ok(expected));
        }

        let error_tests = ["", "0", "10.1", "1001%", "10.1x", "-30%"];

        for s in error_tests {
            assert!(bet_size_from_str(s, true).is_err());
        }
    }

    #[test]
    fn test_bet_sizes_from_str() {
        let tests = [
            (
                "40%, 70%",
                "",
                BetSizeCandidates {
                    bet: vec![BetSize::PotRelative(0.4), BetSize::PotRelative(0.7)],
                    raise: Vec::new(),
                },
            ),
            (
                "0.4,",
                "2.5x, 70%, 40%",
                BetSizeCandidates {
                    bet: vec![BetSize::PotRelative(0.4)],
                    raise: vec![
                        BetSize::PotRelative(0.4),
                        BetSize::PotRelative(0.7),
                        BetSize::LastBetRelative(2.5),
                    ],
                },
            ),
        ];

        for (bet, raise, expected) in tests {
            assert_eq!((bet, raise).try_into(), Ok(expected));
        }

        let error_tests = [("2.5x", ""), (",", "")];

        for (bet, raise) in error_tests {
            assert!(BetSizeCandidates::try_from((bet, raise)).is_err());
        }
    }
}
