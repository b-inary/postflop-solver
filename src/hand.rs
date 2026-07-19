use crate::card::{Card, NOT_DEALT};
use crate::hand_table::*;

#[derive(Clone, Copy, Default)]
pub(crate) struct Hand {
    cards: [usize; 7],
    num_cards: usize,
}

#[inline]
fn keep_n_msb(mut x: i32, n: i32) -> i32 {
    let mut ret = 0;
    for _ in 0..n {
        let bit = 1 << (x.leading_zeros() ^ 31);
        x ^= bit;
        ret |= bit;
    }
    ret
}

#[inline]
fn find_straight(rankset: i32) -> i32 {
    const WHEEL: i32 = 0b1_0000_0000_1111;
    let is_straight = rankset & (rankset << 1) & (rankset << 2) & (rankset << 3) & (rankset << 4);
    if is_straight != 0 {
        keep_n_msb(is_straight, 1)
    } else if (rankset & WHEEL) == WHEEL {
        1 << 3
    } else {
        0
    }
}

impl Hand {
    #[inline]
    pub fn new() -> Hand {
        Hand::default()
    }

    #[inline]
    pub fn add_card(&self, card: usize) -> Hand {
        let mut hand = *self;
        hand.cards[hand.num_cards] = card;
        hand.num_cards += 1;
        hand
    }

    #[inline]
    pub fn contains(&self, card: usize) -> bool {
        self.cards[0..self.num_cards].contains(&card)
    }

    #[inline]
    pub fn evaluate(&self) -> u16 {
        HAND_TABLE.binary_search(&self.evaluate_internal()).unwrap() as u16
    }

    fn evaluate_internal(&self) -> i32 {
        let mut rankset = 0i32;
        let mut rankset_suit = [0i32; 4];
        let mut rankset_of_count = [0i32; 5];
        let mut rank_count = [0i32; 13];

        for &card in &self.cards {
            let rank = card / 4;
            let suit = card % 4;
            rankset |= 1 << rank;
            rankset_suit[suit] |= 1 << rank;
            rank_count[rank] += 1;
        }

        for rank in 0..13 {
            rankset_of_count[rank_count[rank] as usize] |= 1 << rank;
        }

        let mut flush_suit: i32 = -1;
        for suit in 0..4 {
            if rankset_suit[suit as usize].count_ones() >= 5 {
                flush_suit = suit;
            }
        }

        let is_straight = find_straight(rankset);

        if flush_suit >= 0 {
            let is_straight_flush = find_straight(rankset_suit[flush_suit as usize]);
            if is_straight_flush != 0 {
                // straight flush
                (8 << 26) | is_straight_flush
            } else {
                // flush
                (5 << 26) | keep_n_msb(rankset_suit[flush_suit as usize], 5)
            }
        } else if rankset_of_count[4] != 0 {
            // four of a kind
            let remaining = keep_n_msb(rankset ^ rankset_of_count[4], 1);
            (7 << 26) | (rankset_of_count[4] << 13) | remaining
        } else if rankset_of_count[3].count_ones() == 2 {
            // full house
            let trips = keep_n_msb(rankset_of_count[3], 1);
            let pair = rankset_of_count[3] ^ trips;
            (6 << 26) | (trips << 13) | pair
        } else if rankset_of_count[3] != 0 && rankset_of_count[2] != 0 {
            // full house
            let pair = keep_n_msb(rankset_of_count[2], 1);
            (6 << 26) | (rankset_of_count[3] << 13) | pair
        } else if is_straight != 0 {
            // straight
            (4 << 26) | is_straight
        } else if rankset_of_count[3] != 0 {
            // three of a kind
            let remaining = keep_n_msb(rankset_of_count[1], 2);
            (3 << 26) | (rankset_of_count[3] << 13) | remaining
        } else if rankset_of_count[2].count_ones() >= 2 {
            // two pair
            let pairs = keep_n_msb(rankset_of_count[2], 2);
            let remaining = keep_n_msb(rankset ^ pairs, 1);
            (2 << 26) | (pairs << 13) | remaining
        } else if rankset_of_count[2] != 0 {
            // one pair
            let remaining = keep_n_msb(rankset_of_count[1], 3);
            (1 << 26) | (rankset_of_count[2] << 13) | remaining
        } else {
            // high card
            keep_n_msb(rankset, 5)
        }
    }
}

/// 成牌轴：手牌的当前成牌强度分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HandCategory {
    /// 两对及以上（两对、三条、顺子、同花、葫芦、金刚、同花顺）
    Monster = 0,
    /// 超对（口袋对比子 > 所有公共牌 rank）
    Overpair = 1,
    /// 顶对大踢脚（kicker rank >= 公共牌中位 rank）
    TopPairGoodKicker = 2,
    /// 顶对弱踢脚
    TopPairBadKicker = 3,
    /// 中等对子（rank 在公共牌最高和最低之间）
    MediumPair = 4,
    /// 底对（rank == 公共牌最低 rank）
    BottomPair = 5,
    /// 两高张（两张底牌 rank > 公共牌最高 rank，无对子）
    TwoOvercards = 6,
    /// 空气牌（其他所有情况）
    Air = 7,
}

/// 听牌轴：当前听牌类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DrawType {
    /// 无听牌
    None = 0,
    /// 同花听牌（4 张同花）
    FlushDraw = 1,
    /// 两头顺听牌
    OESD = 2,
    /// 卡顺听牌
    Gutshot = 3,
    /// 花顺双抽（同时有同花听和两头顺/卡顺）
    FDPlusSD = 4,
}

/// 手牌分类结果：成牌类别 + 听牌类型。
pub type HandClassification = (HandCategory, DrawType);

/// 对已知牌进行手牌分类。
///
/// - `hero`: 两张底牌
/// - `flop`: 三张翻牌
/// - `turn`: 转牌（`NOT_DEALT` 表示尚未发出）
///
/// 返回 `(HandCategory, DrawType)` 元组。
pub fn classify_hand(hero: (Card, Card), flop: [Card; 3], turn: Card) -> HandClassification {
    // 收集已知牌
    let mut cards: Vec<usize> = vec![
        hero.0 as usize,
        hero.1 as usize,
        flop[0] as usize,
        flop[1] as usize,
        flop[2] as usize,
    ];
    if turn != NOT_DEALT {
        cards.push(turn as usize);
    }

    // 统计 rank 出现次数
    let mut rank_count = [0u8; 14]; // 0-12 实际 rank，13 用于 A-low
    let mut suit_count = [0u8; 4];
    for &card in &cards {
        rank_count[card / 4] += 1;
        suit_count[card % 4] += 1;
    }

    let h0_rank = (hero.0 / 4) as usize;
    let h1_rank = (hero.1 / 4) as usize;

    // 公共牌 rank 信息
    let board_cards: Vec<usize> = if turn != NOT_DEALT {
        vec![
            flop[0] as usize,
            flop[1] as usize,
            flop[2] as usize,
            turn as usize,
        ]
    } else {
        vec![flop[0] as usize, flop[1] as usize, flop[2] as usize]
    };
    let mut board_ranks: Vec<usize> = board_cards.iter().map(|c| c / 4).collect();
    board_ranks.sort_unstable();
    board_ranks.dedup();
    let max_board_rank = *board_ranks.iter().max().unwrap_or(&0);
    let min_board_rank = *board_ranks.iter().min().unwrap_or(&0);

    // 中位 board rank（排序后取中间值）
    let mut sorted_board_ranks: Vec<usize> = board_cards.iter().map(|c| c / 4).collect();
    sorted_board_ranks.sort_unstable();
    let median_board_rank = sorted_board_ranks[sorted_board_ranks.len() / 2];

    // 整体 rankset
    let mut rankset = 0u32;
    for &card in &cards {
        rankset |= 1 << (card / 4);
    }

    // 检查顺子（5 个连续 rank）
    let has_straight = (0..9).any(|start| (start..start + 5).all(|r| (rankset >> r) & 1 != 0));

    // 检查同花（5 张同花）
    let has_flush = suit_count.iter().any(|&c| c >= 5);

    // 统计 hero 对各 rank 的贡献
    let mut hero_contrib = [0u8; 13];
    hero_contrib[h0_rank] += 1;
    hero_contrib[h1_rank] += 1;

    // 检查听牌
    let has_fd = suit_count.iter().any(|&c| c == 4);
    let straight_draw = detect_straight_draw(rankset);

    let draw_type = match (has_fd, straight_draw) {
        (true, Some(_)) => DrawType::FDPlusSD,
        (true, None) => DrawType::FlushDraw,
        (false, Some(dt)) => dt,
        (false, None) => DrawType::None,
    };

    // 基于 hero 贡献的成牌分类
    // 收集 hero 参与的高阶牌型（trips+）
    let mut hero_trips_plus = false;
    // 收集 hero 参与的 pairs
    let mut hero_pair_rank: Option<usize> = None;
    let mut hero_pair_count = 0u8;

    for r in 0..13 {
        let total = rank_count[r];
        let hc = hero_contrib[r];
        if total >= 3 && hc >= 1 {
            // Hero 参与的 trips/quads → Monster
            hero_trips_plus = true;
        }
        if total == 2 && hc >= 1 {
            // Hero 参与的 pair
            hero_pair_rank = Some(r);
            hero_pair_count += 1;
        }
    }

    let category = if has_flush || has_straight || hero_trips_plus || hero_pair_count >= 2 {
        HandCategory::Monster
    } else if let Some(pr) = hero_pair_rank {
        // 确定踢脚：hero 的非对子牌
        let kicker_rank = if h0_rank != pr {
            h0_rank
        } else if h1_rank != pr {
            h1_rank
        } else {
            // 口袋对且 rank 与公共牌匹配：踢脚取公共牌最高非对 rank
            let mut board_kicker = 0;
            for &bc in &board_cards {
                let r = bc / 4;
                if r != pr && r > board_kicker {
                    board_kicker = r;
                }
            }
            board_kicker
        };
        classify_pair(
            pr,
            max_board_rank,
            min_board_rank,
            median_board_rank,
            kicker_rank,
        )
    } else if h0_rank > max_board_rank && h1_rank > max_board_rank {
        HandCategory::TwoOvercards
    } else {
        HandCategory::Air
    };

    (category, draw_type)
}

/// 一对的子分类。
#[inline]
fn classify_pair(
    pair_rank: usize,
    max_board_rank: usize,
    min_board_rank: usize,
    median_board_rank: usize,
    kicker_rank: usize,
) -> HandCategory {
    if pair_rank > max_board_rank {
        return HandCategory::Overpair;
    }

    if pair_rank == max_board_rank {
        if kicker_rank >= median_board_rank {
            HandCategory::TopPairGoodKicker
        } else {
            HandCategory::TopPairBadKicker
        }
    } else if pair_rank > min_board_rank {
        HandCategory::MediumPair
    } else {
        HandCategory::BottomPair
    }
}

/// 检测顺子听牌（OESD 或 Gutshot）。
///
/// 使用滑窗法检查所有 5 连续 rank 窗口中是否恰好有 4 个。
/// rank 0=2, 1=3, ..., 12=A。窗口 [0..4]=2-6, [8..12]=T-A。
/// 轮子（A-2-3-4-5）由窗口 [0..4] 自然处理（A=12 不在窗口中，需 4 张小牌）。
#[inline]
fn detect_straight_draw(rankset: u32) -> Option<DrawType> {
    let mut has_oesd = false;
    let mut has_gutshot = false;

    for start in 0..9 {
        // 窗口覆盖 ranks [start, start+4]，即 [start..start+4]
        let window_mask = 0b11111u32 << start;
        let hits = (rankset & window_mask).count_ones();
        if hits == 4 {
            let missing_in_window = window_mask & !rankset;
            // 缺失位是 bit start 或 bit (start+4) → 端点 → OESD
            if missing_in_window == (1 << start) || missing_in_window == (1 << (start + 4)) {
                has_oesd = true;
            } else {
                has_gutshot = true;
            }
        }
    }

    if has_oesd {
        Some(DrawType::OESD)
    } else if has_gutshot {
        Some(DrawType::Gutshot)
    } else {
        None
    }
}

/// 手牌桶分类（7 桶系统）
///
/// 基于 `(HandCategory, DrawType)` 组合，加上后门花检测和超强超对判断，
/// 将手牌分入 7 个互斥桶中，用于 DL EV 修正。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HandBucket {
    /// 桶1: 超强成牌 — Set、两对、超强超对（AA/KK 在 J 高及以下牌面）
    MonsterMade = 0,
    /// 桶2: 中等顶对 — 顶对好踢脚
    GoodTopPair = 1,
    /// 桶3: 中等成牌/抓 bluff — 顶对弱踢脚、中对、底对、弱超对
    MediumMade = 2,
    /// 桶4: 超强强听牌 — FD+SD 双抽、带对+前门花
    MonsterDraw = 3,
    /// 桶5: 常规强听牌 — 纯 FD、纯 OESD、带对+OESD
    StrongDraw = 4,
    /// 桶6: 弱听牌/后门牌 — Gutshot、后门花
    WeakDraw = 5,
    /// 桶7: 纯空气 — 无对子无听牌
    Trash = 6,
}

impl HandBucket {
    /// 对手牌进行桶分类。
    ///
    /// - `category`, `draw`: 来自 `classify_hand()` 的结果
    /// - `hero`: 两张底牌
    /// - `flop`: 三张翻牌
    pub fn classify(
        category: HandCategory,
        draw: DrawType,
        hero: (Card, Card),
        flop: [Card; 3],
    ) -> Self {
        let is_pair = matches!(
            category,
            HandCategory::Overpair
                | HandCategory::TopPairGoodKicker
                | HandCategory::TopPairBadKicker
                | HandCategory::MediumPair
                | HandCategory::BottomPair
        );

        // 检测后门花听牌（hero + flop 中某花色恰好 3 张，且 hero 至少贡献 1 张）
        let has_backdoor_fd = {
            let h0_suit = (hero.0 % 4) as usize;
            let h1_suit = (hero.1 % 4) as usize;
            let mut suit_count = [0u8; 4];
            suit_count[h0_suit] += 1;
            suit_count[h1_suit] += 1;
            for &fc in &flop {
                suit_count[(fc % 4) as usize] += 1;
            }
            (0..4).any(|s| suit_count[s] == 3 && (s == h0_suit || s == h1_suit))
        };

        let is_front_draw = matches!(
            draw,
            DrawType::FlushDraw | DrawType::OESD | DrawType::FDPlusSD
        );

        // === 覆盖规则（优先于纯成牌分类）===

        // 带对 + 前门花 → 桶4（超强强听牌）
        if is_pair && draw == DrawType::FlushDraw {
            return HandBucket::MonsterDraw;
        }
        // FD+SD 双抽（无论有无对子）→ 桶4
        if draw == DrawType::FDPlusSD {
            return HandBucket::MonsterDraw;
        }
        // 带对 + OESD → 桶5（常规强听牌）
        if is_pair && draw == DrawType::OESD {
            return HandBucket::StrongDraw;
        }

        // === 成牌分类 ===
        match category {
            HandCategory::Monster => HandBucket::MonsterMade,
            HandCategory::Overpair => {
                let pair_rank = hero.0.max(hero.1) / 4;
                if pair_rank >= 11 {
                    HandBucket::MonsterMade // AA/KK: 超强超对
                } else {
                    HandBucket::MediumMade // QQ/JJ 等: 弱超对
                }
            }
            HandCategory::TopPairGoodKicker => HandBucket::GoodTopPair,
            HandCategory::TopPairBadKicker
            | HandCategory::MediumPair
            | HandCategory::BottomPair => HandBucket::MediumMade,
            HandCategory::TwoOvercards | HandCategory::Air => {
                if is_front_draw {
                    HandBucket::StrongDraw
                } else if draw == DrawType::Gutshot || has_backdoor_fd {
                    HandBucket::WeakDraw
                } else {
                    HandBucket::Trash
                }
            }
        }
    }
}

/// 翻牌牌面纹理类型（8 种同构分类）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoardTexture {
    /// 三条面（3 张 rank 相同）
    Trips,
    /// 对子彩虹面（2 张 rank 相同 + 3 种花色）
    PairedRainbow,
    /// 对子两同花面（2 张 rank 相同 + 含 2 张同花色）
    PairedTwoTone,
    /// 纯彩虹面（3 张 rank 不同 + 3 种花色）
    Rainbow,
    /// 两同花面 A（2 张同花色包含最大 rank 牌）
    TwoToneA,
    /// 两同花面 B（2 张同花色不包含最大 rank 牌）
    TwoToneB,
    /// 单调面 A（3 张同花色，最大 rank ≥ J）
    MonotoneA,
    /// 单调面 B（3 张同花色，最大 rank < J）
    MonotoneB,
}

/// 自动检测翻牌的纹理类型。
pub fn detect_board_texture(flop: [Card; 3]) -> BoardTexture {
    let r0 = flop[0] / 4;
    let r1 = flop[1] / 4;
    let r2 = flop[2] / 4;
    let s0 = flop[0] % 4;
    let s1 = flop[1] % 4;
    let s2 = flop[2] % 4;

    let is_paired = r0 == r1 || r1 == r2 || r0 == r2;
    let is_trips = r0 == r1 && r1 == r2;

    if is_trips {
        return BoardTexture::Trips;
    }

    // 花色统计
    let mut suit_count = [0u8; 4];
    suit_count[s0 as usize] += 1;
    suit_count[s1 as usize] += 1;
    suit_count[s2 as usize] += 1;
    let max_suit_count = *suit_count.iter().max().unwrap();
    let all_same_suit = max_suit_count == 3;
    let two_tone = max_suit_count == 2;
    let rainbow = max_suit_count == 1;

    if is_paired {
        if rainbow {
            BoardTexture::PairedRainbow
        } else {
            BoardTexture::PairedTwoTone
        }
    } else if all_same_suit {
        let max_rank = r0.max(r1).max(r2);
        if max_rank >= 9 {
            BoardTexture::MonotoneA // J=9, Q=10, K=11, A=12
        } else {
            BoardTexture::MonotoneB
        }
    } else if rainbow {
        BoardTexture::Rainbow
    } else {
        // two_tone: 找到出现 2 次的花色，检查该花色是否包含最大 rank
        let two_suit = suit_count.iter().position(|&c| c == 2).unwrap() as u8;
        let max_rank = r0.max(r1).max(r2);
        let two_suit_has_max = flop.iter().any(|&c| c % 4 == two_suit && c / 4 == max_rank);
        if two_suit_has_max {
            BoardTexture::TwoToneA
        } else {
            BoardTexture::TwoToneB
        }
    }
}

/// 获取公牌两同花的花色（出现 2 次的花色），无则返回 None。
pub fn board_two_tone_suit(flop: [Card; 3]) -> Option<u8> {
    let mut suit_count = [0u8; 4];
    for &c in &flop {
        suit_count[(c % 4) as usize] += 1;
    }
    suit_count.iter().position(|&c| c == 2).map(|i| i as u8)
}

/// 获取单调面的花色（3 张同花），否则返回 None。
pub fn board_monotone_suit(flop: [Card; 3]) -> Option<u8> {
    let s0 = flop[0] % 4;
    let s1 = flop[1] % 4;
    let s2 = flop[2] % 4;
    if s0 == s1 && s1 == s2 {
        Some(s0)
    } else {
        None
    }
}

/// 检查 hero 是否持有指定花色的牌。
#[inline]
pub fn hero_has_suit(hero: (Card, Card), suit: u8) -> bool {
    hero.0 % 4 == suit || hero.1 % 4 == suit
}

/// 检查 hero 是否持有同花色的 A 或 K（nut blocker）。
#[inline]
pub fn has_nut_blocker(hero: (Card, Card), flush_suit: u8) -> bool {
    let cards = [hero.0, hero.1];
    cards.iter().any(|&c| {
        c % 4 == flush_suit && (c / 4 == 12 || c / 4 == 11) // A=12, K=11
    })
}

/// 检查 hero 是否持有同花色的 A（nut flush draw）。
#[inline]
pub fn has_nut_flush_draw(hero: (Card, Card), flush_suit: u8) -> bool {
    let cards = [hero.0, hero.1];
    cards.iter().any(|&c| c % 4 == flush_suit && c / 4 == 12) // A=12
}

/// 求解后手牌桶固定修正系数 α（v4 方案）。
///
/// 修正公式：`修正后 DLEv = DLEv × α(牌面, 位置, 桶, 街)`
///
/// 翻牌圈和转牌圈的 α 系数各自独立回归。
pub struct BoardCorrectionContext {
    pub texture: BoardTexture,
    pub is_turn: bool,
}

impl BoardCorrectionContext {
    /// 创建翻牌圈修正上下文。
    pub fn new(texture: BoardTexture) -> Self {
        Self { texture, is_turn: false }
    }
    /// 创建转牌圈修正上下文。
    pub fn new_turn(texture: BoardTexture) -> Self {
        Self { texture, is_turn: true }
    }

    /// IP 位置进攻因子：赋予 IP 低 EV 手牌额外的 bluff 溢价。
    #[inline]
    fn ip_aggression_factor(bucket: HandBucket) -> f64 {
        match bucket {
            HandBucket::Trash => 1.10,
            HandBucket::WeakDraw => 1.08,
            HandBucket::MediumMade => 1.05,
            HandBucket::GoodTopPair => 1.03,
            HandBucket::StrongDraw => 1.05,
            HandBucket::MonsterDraw => 1.03,
            HandBucket::MonsterMade => 1.00,
        }
    }

    /// 查询修正系数 α。
    /// `position`: 0=OOP, 1=IP
    #[inline]
    pub fn alpha(&self, position: usize, bucket: HandBucket) -> f64 {
        let base = if self.is_turn {
            self.turn_alpha(position, bucket)
        } else {
            self.flop_alpha(position, bucket)
        };
        if !self.is_turn && position == 1 {
            base * Self::ip_aggression_factor(bucket)
        } else {
            base
        }
    }

    /// 翻牌圈 α 系数（基于纯 equity DLEv 回归）。
    fn flop_alpha(&self, position: usize, bucket: HandBucket) -> f64 {
        let is_oop = position == 0;
        match self.texture {
            BoardTexture::Trips => match (is_oop, bucket) {
                (true, HandBucket::MonsterMade) => 1.49,
                (false, HandBucket::MonsterMade) => 1.38,
                (true, HandBucket::MediumMade) => 1.15,
                (false, HandBucket::MediumMade) => 0.94,
                (true, HandBucket::WeakDraw) => 0.92,
                (true, HandBucket::Trash) => 0.83,
                _ => 1.0,
            },
            BoardTexture::PairedRainbow => match (is_oop, bucket) {
                (true, HandBucket::MonsterMade) => 1.22,
                (false, HandBucket::MonsterMade) => 1.18,
                (true, HandBucket::MediumMade) => 0.81,
                (false, HandBucket::MediumMade) => 0.88,
                (true, HandBucket::WeakDraw) => 0.85,
                (false, HandBucket::WeakDraw) => 1.14,
                (true, HandBucket::Trash) => 0.67,
                (false, HandBucket::Trash) => 0.82,
                _ => 1.0,
            },
            BoardTexture::PairedTwoTone => match (is_oop, bucket) {
                (true, HandBucket::MonsterMade) => 1.14,
                (false, HandBucket::MonsterMade) => 1.13,
                (true, HandBucket::MediumMade) => 0.82,
                (false, HandBucket::MediumMade) => 0.96,
                (true, HandBucket::StrongDraw) => 1.03,
                (false, HandBucket::StrongDraw) => 1.17,
                (true, HandBucket::WeakDraw) => 0.79,
                (false, HandBucket::WeakDraw) => 1.00,
                (true, HandBucket::Trash) => 0.73,
                (false, HandBucket::Trash) => 0.87,
                _ => 1.0,
            },
            BoardTexture::Rainbow => match (is_oop, bucket) {
                (true, HandBucket::MonsterMade) => 1.26,
                (false, HandBucket::MonsterMade) => 1.44,
                (true, HandBucket::GoodTopPair) => 1.00,
                (false, HandBucket::GoodTopPair) => 1.07,
                (true, HandBucket::MediumMade) => 0.74,
                (false, HandBucket::MediumMade) => 0.79,
                (true, HandBucket::StrongDraw) => 1.00,
                (false, HandBucket::StrongDraw) => 1.41,
                (true, HandBucket::WeakDraw) => 1.01,
                (true, HandBucket::Trash) => 0.68,
                (false, HandBucket::Trash) => 0.53,
                _ => 1.0,
            },
            BoardTexture::TwoToneA => match (is_oop, bucket) {
                (true, HandBucket::MonsterMade) => 1.17,
                (false, HandBucket::MonsterMade) => 1.28,
                (true, HandBucket::GoodTopPair) => 0.92,
                (false, HandBucket::GoodTopPair) => 0.99,
                (true, HandBucket::MediumMade) => 0.74,
                (false, HandBucket::MediumMade) => 0.83,
                (true, HandBucket::MonsterDraw) => 1.22,
                (false, HandBucket::MonsterDraw) => 1.38,
                (true, HandBucket::StrongDraw) => 1.00,
                (false, HandBucket::StrongDraw) => 1.36,
                (true, HandBucket::WeakDraw) => 0.95,
                (true, HandBucket::Trash) => 0.65,
                (false, HandBucket::Trash) => 0.67,
                _ => 1.0,
            },
            BoardTexture::TwoToneB => match (is_oop, bucket) {
                (true, HandBucket::MonsterMade) => 1.19,
                (false, HandBucket::MonsterMade) => 1.33,
                (true, HandBucket::GoodTopPair) => 0.93,
                (false, HandBucket::GoodTopPair) => 1.00,
                (true, HandBucket::MediumMade) => 0.74,
                (false, HandBucket::MediumMade) => 0.82,
                (true, HandBucket::MonsterDraw) => 1.21,
                (false, HandBucket::MonsterDraw) => 1.22,
                (true, HandBucket::StrongDraw) => 0.99,
                (false, HandBucket::StrongDraw) => 1.33,
                (true, HandBucket::WeakDraw) => 0.94,
                (true, HandBucket::Trash) => 0.63,
                (false, HandBucket::Trash) => 0.65,
                _ => 1.0,
            },
            BoardTexture::MonotoneA => match (is_oop, bucket) {
                (true, HandBucket::MonsterMade) => 1.16,
                (false, HandBucket::MonsterMade) => 1.19,
                (true, HandBucket::GoodTopPair) => 0.76,
                (false, HandBucket::GoodTopPair) => 0.79,
                (true, HandBucket::MediumMade) => 0.71,
                (false, HandBucket::MediumMade) => 0.77,
                (true, HandBucket::MonsterDraw) => 1.09,
                (false, HandBucket::MonsterDraw) => 1.14,
                (true, HandBucket::StrongDraw) => 0.86,
                (false, HandBucket::StrongDraw) => 1.10,
                (true, HandBucket::WeakDraw) => 0.90,
                (true, HandBucket::Trash) => 0.70,
                (false, HandBucket::Trash) => 0.54,
                _ => 1.0,
            },
            BoardTexture::MonotoneB => match (is_oop, bucket) {
                (true, HandBucket::MonsterMade) => 1.16,
                (false, HandBucket::MonsterMade) => 1.18,
                (true, HandBucket::GoodTopPair) => 0.74,
                (false, HandBucket::GoodTopPair) => 0.76,
                (true, HandBucket::MediumMade) => 0.76,
                (false, HandBucket::MediumMade) => 0.79,
                (true, HandBucket::MonsterDraw) => 0.95,
                (false, HandBucket::MonsterDraw) => 0.86,
                (true, HandBucket::StrongDraw) => 1.08,
                (false, HandBucket::StrongDraw) => 1.17,
                (true, HandBucket::Trash) => 0.70,
                (false, HandBucket::Trash) => 0.79,
                _ => 1.0,
            },
        }
    }

    /// 转牌圈 α 系数（基于纯 equity DLEv 回归）。
    fn turn_alpha(&self, _position: usize, _bucket: HandBucket) -> f64 {
        1.0
    }
    pub fn apply(
        &self,
        dlev: f32,
        position: usize,
        bucket: HandBucket,
        _draw: DrawType,
        _hero: (Card, Card),
    ) -> f32 {
        let a = self.alpha(position, bucket);
        (dlev as f64 * a) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_hands() {
        let mut appeared = vec![false; HAND_TABLE.len()];
        let mut counter = [0; 9];

        for i in 0..52 {
            let hand = Hand::new().add_card(i);
            for j in (i + 1)..52 {
                let hand = hand.add_card(j);
                for k in (j + 1)..52 {
                    let hand = hand.add_card(k);
                    for m in (k + 1)..52 {
                        let hand = hand.add_card(m);
                        for n in (m + 1)..52 {
                            let hand = hand.add_card(n);
                            for p in (n + 1)..52 {
                                let hand = hand.add_card(p);
                                for q in (p + 1)..52 {
                                    let hand = hand.add_card(q);
                                    let raw_value = hand.evaluate_internal();
                                    let index_result = HAND_TABLE.binary_search(&raw_value);
                                    assert!(index_result.is_ok());
                                    appeared[index_result.unwrap()] = true;
                                    counter[(raw_value >> 26) as usize] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        assert!(appeared.iter().all(|&x| x));
        assert_eq!(counter[8], 41584); // straight flush
        assert_eq!(counter[7], 224848); // four of a kind
        assert_eq!(counter[6], 3473184); // full house
        assert_eq!(counter[5], 4047644); // flush
        assert_eq!(counter[4], 6180020); // straight
        assert_eq!(counter[3], 6461620); // three of a kind
        assert_eq!(counter[2], 31433400); // two pair
        assert_eq!(counter[1], 58627800); // one pair
        assert_eq!(counter[0], 23294460); // high card
    }
}
