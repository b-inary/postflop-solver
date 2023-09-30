use crate::hand::*;
use crate::range::*;
use std::mem;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// A type representing a card, defined as an alias of `u8`.
///
/// The correspondence between the card and its ID is defined as follows:
/// - `card_id = 4 * rank + suit` (where `0 <= card_id < 52`)
///   - `rank`: 2 => `0`, 3 => `1`, 4 => `2`, ..., A => `12`
///   - `suit`: club => `0`, diamond => `1`, heart => `2`, spade => `3`
pub type Card = u8;

/// Constant representing that the card is not yet dealt.
pub const NOT_DEALT: Card = Card::MAX;

/// A struct containing the card configuration.
///
/// # Examples
/// ```
/// use postflop_solver::*;
///
/// let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
/// let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";
///
/// let card_config = CardConfig {
///     range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
///     flop: flop_from_str("Td9d6h").unwrap(),
///     turn: card_from_str("Qc").unwrap(),
///     river: NOT_DEALT,
/// };
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct CardConfig {
    /// Initial range of each player.
    pub range: [Range; 2],

    /// Flop cards: each card must be unique.
    pub flop: [Card; 3],

    /// Turn card: must be in range [`0`, `52`) or `NOT_DEALT`.
    pub turn: Card,

    /// River card: must be in range [`0`, `52`) or `NOT_DEALT`.
    pub river: Card,
}

impl Default for CardConfig {
    #[inline]
    fn default() -> Self {
        Self {
            range: Default::default(),
            flop: [NOT_DEALT; 3],
            turn: NOT_DEALT,
            river: NOT_DEALT,
        }
    }
}

type PrivateCards = [Vec<(Card, Card)>; 2];

type Indices = [Vec<u16>; 2];

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StrengthItem {
    pub(crate) strength: u16,
    pub(crate) index: u16,
}

pub(crate) type SwapList = [Vec<(u16, u16)>; 2];

type IsomorphismData = (
    Vec<u8>,
    Vec<Card>,
    [SwapList; 4],
    Vec<Vec<u8>>,
    [Vec<Card>; 4],
    [[SwapList; 4]; 4],
);

/// Returns an index of the given card pair.
///
/// Examples: 2d2c => `0`, 2h2c => `1`, 2s2c => `2`, ..., AsAh => `1325`
#[inline]
pub(crate) fn card_pair_to_index(mut card1: Card, mut card2: Card) -> usize {
    if card1 > card2 {
        mem::swap(&mut card1, &mut card2);
    }
    card1 as usize * (101 - card1 as usize) / 2 + card2 as usize - 1
}

/// Returns a card pair from the given index.
///
/// Examples: `0` => 2d2c, `1` => 2h2c , `2` => 2s2c, ..., `1325` => AsAh
#[inline]
pub(crate) fn index_to_card_pair(index: usize) -> (Card, Card) {
    let card1 = (103 - (103.0 * 103.0 - 8.0 * index as f64).sqrt().ceil() as u16) / 2;
    let card2 = index as u16 - card1 * (101 - card1) / 2 + 1;
    (card1 as Card, card2 as Card)
}

impl CardConfig {
    pub(crate) fn valid_indices(
        &self,
        private_cards: &PrivateCards,
    ) -> (Indices, Vec<Indices>, Vec<Indices>) {
        let ret_flop = if self.turn == NOT_DEALT {
            [
                (0..private_cards[0].len() as u16).collect(),
                (0..private_cards[1].len() as u16).collect(),
            ]
        } else {
            Indices::default()
        };

        let mut ret_turn = vec![Indices::default(); 52];
        for board in 0..52 {
            if !self.flop.contains(&board)
                && (self.turn == NOT_DEALT || self.turn == board)
                && self.river == NOT_DEALT
            {
                ret_turn[board as usize] =
                    Self::valid_indices_internal(private_cards, board, NOT_DEALT);
            }
        }

        let mut ret_river = vec![Indices::default(); 52 * 51 / 2];
        for board1 in 0..52 {
            for board2 in board1 + 1..52 {
                if !self.flop.contains(&board1)
                    && !self.flop.contains(&board2)
                    && (self.turn == NOT_DEALT || board1 == self.turn || board2 == self.turn)
                    && (self.river == NOT_DEALT || board1 == self.river || board2 == self.river)
                {
                    let index = card_pair_to_index(board1, board2);
                    ret_river[index] = Self::valid_indices_internal(private_cards, board1, board2);
                }
            }
        }

        (ret_flop, ret_turn, ret_river)
    }

    fn valid_indices_internal(
        private_cards: &[Vec<(Card, Card)>; 2],
        board1: Card,
        board2: Card,
    ) -> [Vec<u16>; 2] {
        let mut ret = [
            Vec::with_capacity(private_cards[0].len()),
            Vec::with_capacity(private_cards[1].len()),
        ];

        let mut board_mask: u64 = 0;
        if board1 != NOT_DEALT {
            board_mask |= 1 << board1;
        }
        if board2 != NOT_DEALT {
            board_mask |= 1 << board2;
        }

        for player in 0..2 {
            ret[player].extend(private_cards[player].iter().enumerate().filter_map(
                |(index, &(c1, c2))| {
                    let hand_mask: u64 = (1 << c1) | (1 << c2);
                    if hand_mask & board_mask == 0 {
                        Some(index as u16)
                    } else {
                        None
                    }
                },
            ));

            ret[player].shrink_to_fit();
        }

        ret
    }

    pub(crate) fn hand_strength(
        &self,
        private_cards: &PrivateCards,
    ) -> Vec<[Vec<StrengthItem>; 2]> {
        let mut ret = vec![Default::default(); 52 * 51 / 2];

        let mut board = Hand::new();
        for &card in &self.flop {
            board = board.add_card(card as usize);
        }

        for board1 in 0..52 {
            for board2 in board1 + 1..52 {
                if !board.contains(board1 as usize)
                    && !board.contains(board2 as usize)
                    && (self.turn == NOT_DEALT || board1 == self.turn || board2 == self.turn)
                    && (self.river == NOT_DEALT || board1 == self.river || board2 == self.river)
                {
                    let board = board.add_card(board1 as usize).add_card(board2 as usize);
                    let mut strength = [
                        Vec::with_capacity(private_cards[0].len() + 2),
                        Vec::with_capacity(private_cards[1].len() + 2),
                    ];

                    for player in 0..2 {
                        // add the weakest and strongest sentinels
                        strength[player].push(StrengthItem {
                            strength: 0,
                            index: 0,
                        });
                        strength[player].push(StrengthItem {
                            strength: u16::MAX,
                            index: u16::MAX,
                        });

                        strength[player].extend(
                            private_cards[player].iter().enumerate().filter_map(
                                |(index, &(c1, c2))| {
                                    let (c1, c2) = (c1 as usize, c2 as usize);
                                    if board.contains(c1) || board.contains(c2) {
                                        None
                                    } else {
                                        let hand = board.add_card(c1).add_card(c2);
                                        Some(StrengthItem {
                                            strength: hand.evaluate() + 1, // +1 to avoid 0
                                            index: index as u16,
                                        })
                                    }
                                },
                            ),
                        );

                        strength[player].shrink_to_fit();
                        strength[player].sort_unstable();
                    }

                    ret[card_pair_to_index(board1, board2)] = strength;
                }
            }
        }

        ret
    }

    pub(crate) fn isomorphism(&self, private_cards: &[Vec<(Card, Card)>; 2]) -> IsomorphismData {
        let mut suit_isomorphism = [0; 4];
        let mut next_index = 1;
        'outer: for suit2 in 1..4 {
            for suit1 in 0..suit2 {
                if self.range[0].is_suit_isomorphic(suit1, suit2)
                    && self.range[1].is_suit_isomorphic(suit1, suit2)
                {
                    suit_isomorphism[suit2 as usize] = suit_isomorphism[suit1 as usize];
                    continue 'outer;
                }
            }
            suit_isomorphism[suit2 as usize] = next_index;
            next_index += 1;
        }

        let flop_mask: u64 = (1 << self.flop[0]) | (1 << self.flop[1]) | (1 << self.flop[2]);
        let mut flop_rankset = [0; 4];

        for &card in &self.flop {
            let rank = card >> 2;
            let suit = card & 3;
            flop_rankset[suit as usize] |= 1 << rank;
        }

        let mut isomorphic_suit = [None; 4];
        let mut reverse_table = vec![usize::MAX; 52 * 51 / 2];

        let mut isomorphism_ref_turn = Vec::new();
        let mut isomorphism_card_turn = Vec::new();
        let mut isomorphism_swap_turn = Default::default();

        // turn isomorphism
        if self.turn == NOT_DEALT {
            for suit1 in 1..4 {
                for suit2 in 0..suit1 {
                    if flop_rankset[suit1 as usize] == flop_rankset[suit2 as usize]
                        && suit_isomorphism[suit1 as usize] == suit_isomorphism[suit2 as usize]
                    {
                        isomorphic_suit[suit1 as usize] = Some(suit2);
                        Self::isomorphism_swap_internal(
                            &mut isomorphism_swap_turn,
                            &mut reverse_table,
                            suit1,
                            suit2,
                            private_cards,
                        );
                        break;
                    }
                }
            }

            Self::isomorphism_internal(
                &mut isomorphism_ref_turn,
                &mut isomorphism_card_turn,
                flop_mask,
                &isomorphic_suit,
            );
        }

        let mut isomorphism_ref_river = vec![Vec::new(); 52];
        let mut isomorphism_card_river: [Vec<Card>; 4] = Default::default();
        let mut isomorphism_swap_river: [[SwapList; 4]; 4] = Default::default();

        // river isomorphism
        if self.river == NOT_DEALT {
            for turn in 0..52 {
                if (1 << turn) & flop_mask != 0 || (self.turn != NOT_DEALT && self.turn != turn) {
                    continue;
                }

                let turn_mask = flop_mask | (1 << turn);
                let mut turn_rankset = flop_rankset;
                turn_rankset[turn as usize & 3] |= 1 << (turn >> 2);

                isomorphic_suit.fill(None);

                for suit1 in 1..4 {
                    for suit2 in 0..suit1 {
                        if (flop_rankset[suit1 as usize] == flop_rankset[suit2 as usize]
                            || self.turn != NOT_DEALT)
                            && turn_rankset[suit1 as usize] == turn_rankset[suit2 as usize]
                            && suit_isomorphism[suit1 as usize] == suit_isomorphism[suit2 as usize]
                        {
                            isomorphic_suit[suit1 as usize] = Some(suit2);
                            Self::isomorphism_swap_internal(
                                &mut isomorphism_swap_river[turn as usize & 3],
                                &mut reverse_table,
                                suit1,
                                suit2,
                                private_cards,
                            );
                            break;
                        }
                    }
                }

                Self::isomorphism_internal(
                    &mut isomorphism_ref_river[turn as usize],
                    &mut isomorphism_card_river[turn as usize & 3],
                    turn_mask,
                    &isomorphic_suit,
                );
            }
        }

        (
            isomorphism_ref_turn,
            isomorphism_card_turn,
            isomorphism_swap_turn,
            isomorphism_ref_river,
            isomorphism_card_river,
            isomorphism_swap_river,
        )
    }

    fn isomorphism_swap_internal(
        swap_list: &mut [SwapList; 4],
        reverse_table: &mut [usize],
        suit1: u8,
        suit2: u8,
        private_cards: &PrivateCards,
    ) {
        let swap_list = &mut swap_list[suit1 as usize];
        let replacer = |card: Card| {
            if card & 3 == suit1 {
                card - suit1 + suit2
            } else if card & 3 == suit2 {
                card + suit1 - suit2
            } else {
                card
            }
        };

        for player in 0..2 {
            if !swap_list[player].is_empty() {
                continue;
            }

            reverse_table.fill(usize::MAX);
            let cards = &private_cards[player];

            for i in 0..cards.len() {
                reverse_table[card_pair_to_index(cards[i].0, cards[i].1)] = i;
            }

            for (i, &(c1, c2)) in cards.iter().enumerate() {
                let c1 = replacer(c1);
                let c2 = replacer(c2);
                let index = reverse_table[card_pair_to_index(c1, c2)];
                if i < index {
                    swap_list[player].push((i as u16, index as u16));
                }
            }
        }
    }

    fn isomorphism_internal(
        isomorphism_ref: &mut Vec<u8>,
        isomorphism_card: &mut Vec<Card>,
        mask: u64,
        isomorphic_suit: &[Option<u8>; 4],
    ) {
        let push_card = isomorphism_card.is_empty();
        let mut counter = 0;
        let mut indices = [0; 52];

        for card in 0..52 {
            if (1 << card) & mask != 0 {
                continue;
            }

            let suit = card & 3;

            if let Some(replace_suit) = isomorphic_suit[suit as usize] {
                let replace_card = card - suit + replace_suit;
                isomorphism_ref.push(indices[replace_card as usize]);
                if push_card {
                    isomorphism_card.push(card);
                }
            } else {
                indices[card as usize] = counter;
                counter += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_pair_index() {
        let mut k = 0;
        for i in 0..52 {
            for j in (i + 1)..52 {
                assert_eq!(card_pair_to_index(i, j), k);
                assert_eq!(card_pair_to_index(j, i), k);
                assert_eq!(index_to_card_pair(k), (i, j));
                k += 1;
            }
        }
    }
}
