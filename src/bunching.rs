use crate::atomic_float::*;
use crate::card::*;
use crate::range::*;
use crate::utility::*;
use std::io::{self, Write};
use std::mem;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

const COMB_49_1: usize = 49;
const COMB_49_2: usize = 1176;
const COMB_49_3: usize = 18424;
const COMB_49_4: usize = 211876;
const COMB_49_5: usize = 1906884;
const COMB_49_6: usize = 13983816;
const COMB_49_8: usize = 450978066;

const COMB_TABLE: [[usize; 49]; 8] = [
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48,
    ],
    [
        0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210,
        231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528, 561, 595, 630, 666, 703, 741,
        780, 820, 861, 903, 946, 990, 1035, 1081, 1128,
    ],
    [
        0, 0, 0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680, 816, 969, 1140,
        1330, 1540, 1771, 2024, 2300, 2600, 2925, 3276, 3654, 4060, 4495, 4960, 5456, 5984, 6545,
        7140, 7770, 8436, 9139, 9880, 10660, 11480, 12341, 13244, 14190, 15180, 16215, 17296,
    ],
    [
        0, 0, 0, 0, 1, 5, 15, 35, 70, 126, 210, 330, 495, 715, 1001, 1365, 1820, 2380, 3060, 3876,
        4845, 5985, 7315, 8855, 10626, 12650, 14950, 17550, 20475, 23751, 27405, 31465, 35960,
        40920, 46376, 52360, 58905, 66045, 73815, 82251, 91390, 101270, 111930, 123410, 135751,
        148995, 163185, 178365, 194580,
    ],
    [
        0, 0, 0, 0, 0, 1, 6, 21, 56, 126, 252, 462, 792, 1287, 2002, 3003, 4368, 6188, 8568, 11628,
        15504, 20349, 26334, 33649, 42504, 53130, 65780, 80730, 98280, 118755, 142506, 169911,
        201376, 237336, 278256, 324632, 376992, 435897, 501942, 575757, 658008, 749398, 850668,
        962598, 1086008, 1221759, 1370754, 1533939, 1712304,
    ],
    [
        0, 0, 0, 0, 0, 0, 1, 7, 28, 84, 210, 462, 924, 1716, 3003, 5005, 8008, 12376, 18564, 27132,
        38760, 54264, 74613, 100947, 134596, 177100, 230230, 296010, 376740, 475020, 593775,
        736281, 906192, 1107568, 1344904, 1623160, 1947792, 2324784, 2760681, 3262623, 3838380,
        4496388, 5245786, 6096454, 7059052, 8145060, 9366819, 10737573, 12271512,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 1, 8, 36, 120, 330, 792, 1716, 3432, 6435, 11440, 19448, 31824, 50388,
        77520, 116280, 170544, 245157, 346104, 480700, 657800, 888030, 1184040, 1560780, 2035800,
        2629575, 3365856, 4272048, 5379616, 6724520, 8347680, 10295472, 12620256, 15380937,
        18643560, 22481940, 26978328, 32224114, 38320568, 45379620, 53524680, 62891499, 73629072,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 9, 45, 165, 495, 1287, 3003, 6435, 12870, 24310, 43758, 75582,
        125970, 203490, 319770, 490314, 735471, 1081575, 1562275, 2220075, 3108105, 4292145,
        5852925, 7888725, 10518300, 13884156, 18156204, 23535820, 30260340, 38608020, 48903492,
        61523748, 76904685, 95548245, 118030185, 145008513, 177232627, 215553195, 260932815,
        314457495, 377348994,
    ],
];

/// A configuration for computing the bunching effect.
///
/// This struct calculates the number of possible hand combinations of folded players when some
/// cards are known to be dead, fully utilizing the inclusion-exclusion principle.
/// For example, if the hero's and opponent's hands are fixed, with specific turn and river cards,
/// we want to know the "weight" of the possible hand combinations of the folded players to
/// accurately account for the bunching effect.
/// In this case, the query is defined by 6 cards (2 for the hero, 2 for the opponent, 2 for the
/// board), and this struct can answer the query instantly after precomputation.
///
/// # Examples
///
/// ```no_run
/// use postflop_solver::*;
///
/// let utg_range_str = "...";
/// let mp_range_str = "...";
/// let co_range_str = "...";
/// let sb_range_str = "...";
///
/// let mut data = BunchingData::new(
///     // support up to 4 fold players
///     &[
///         utg_range_str.parse().unwrap(),
///         mp_range_str.parse().unwrap(),
///         co_range_str.parse().unwrap(),
///         sb_range_str.parse().unwrap(),
///     ],
///     flop_from_str("QsJh2h").unwrap()
/// )
/// .unwrap();
///
/// // precompute the combination table
/// data.process(true);
///
/// assert!(data.is_ready());
/// ```
///
/// # Memory Usage
///
/// The `BunchingData` struct requires 62MB of memory to store the final result.
///
/// The additional memory usage required for the computation is as follows:
///
/// | #(fold players) | Aditional memory usage |
/// |:---:|:---:|
/// | 1 | 18.4KB |
/// | 2 | 1.77MB |
/// | 3 | 123MB |
/// | 4 | 3.42GB |
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct BunchingData {
    // input
    fold_ranges: Vec<Range>,
    flop: [Card; 3],

    // current status
    phase: u8,
    progress_percent: u8,

    // combination table (computed in phase 1)
    temp_table1: Vec<f64>,
    temp_table2: Vec<f64>,
    temp_table3: Vec<AtomicF64>,

    // sums of each subset (computed in phase 2)
    sum: [Vec<AtomicF64>; 7],

    // inclusion-exclusion sums (computed in phase 3)
    result4: Vec<AtomicF32>,
    result5: Vec<AtomicF32>,
    result6: Vec<AtomicF32>,
}

#[inline]
fn mask_to_index(mut mask: u64, k: usize) -> usize {
    let mut index = 0;
    for i in 0..k {
        assert!(mask != 0);
        let tz = mask.trailing_zeros();
        index += COMB_TABLE[i][tz as usize];
        mask &= mask - 1;
    }
    index
}

#[inline]
fn index_to_mask(mut index: usize, k: usize) -> u64 {
    let mut mask = 0;
    for i in (0..k).rev() {
        let n = COMB_TABLE[i].partition_point(|&x| x <= index) - 1;
        index -= COMB_TABLE[i][n];
        mask |= 1 << n;
    }
    mask
}

#[inline]
fn next_combination(mask: u64) -> u64 {
    assert!(mask != 0);
    let t = mask | (mask - 1);
    (t + 1) | (((!t & (t + 1)) - 1) >> (mask.trailing_zeros() + 1))
}

#[inline]
fn compress_mask(mut mask: u64, flop: [Card; 3]) -> u64 {
    assert!(flop[0] < flop[1] && flop[1] < flop[2]);
    for i in 0..3 {
        let m = (1 << (flop[i] as usize - i)) - 1;
        mask = (mask & m) | ((mask >> 1) & !m);
    }
    mask
}

impl BunchingData {
    /// Creates a new `BunchingConfig` instance.
    ///
    /// `fold_ranges` can contain at most 4 ranges (6-max).
    #[inline]
    pub fn new(fold_ranges: &[Range], mut flop: [Card; 3]) -> Result<Self, String> {
        let mut fold_ranges_vec = Vec::new();

        for range in fold_ranges {
            if !range.is_empty() {
                if !range.is_suit_symmetric() {
                    return Err("Fold ranges must be suit-symmetric".to_string());
                }
                fold_ranges_vec.push(*range);
            }
        }

        if fold_ranges_vec.is_empty() {
            return Err("Fold ranges is empty".to_string());
        }

        if fold_ranges_vec.len() > 4 {
            return Err("The number of folded players must be at most 4".to_string());
        }

        flop.sort_unstable();

        if flop[0] == flop[1] || flop[1] == flop[2] || flop[2] >= 52 {
            return Err("Invalid flop".to_string());
        }

        Ok(Self {
            fold_ranges: fold_ranges_vec,
            flop,
            phase: 0,
            progress_percent: 0,
            temp_table1: Vec::new(),
            temp_table2: Vec::new(),
            temp_table3: Vec::new(),
            sum: Default::default(),
            result4: Vec::new(),
            result5: Vec::new(),
            result6: Vec::new(),
        })
    }

    /// Returns a reference to the fold ranges.
    #[inline]
    pub fn fold_ranges(&self) -> &[Range] {
        &self.fold_ranges
    }

    /// Returns the flop.
    #[inline]
    pub fn flop(&self) -> [Card; 3] {
        self.flop
    }

    /// Returns whether the instance is ready to use.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.phase == 3 && self.progress_percent == 100
    }

    /// Returns the current phase (0-3).
    #[inline]
    pub fn phase(&self) -> u8 {
        self.phase
    }

    /// Returns the current progress in percent (0-100).
    #[inline]
    pub fn progress_percent(&self) -> u8 {
        self.progress_percent
    }

    /// Returns the memory usage in bytes.
    #[inline]
    pub fn memory_usage(&self) -> u64 {
        let mut sum = 0;

        sum += mem::size_of::<Self>() as u64;

        sum += vec_memory_usage(&self.fold_ranges);
        sum += vec_memory_usage(&self.temp_table1);
        sum += vec_memory_usage(&self.temp_table2);
        sum += vec_memory_usage(&self.temp_table3);

        for vec in &self.sum {
            sum += vec_memory_usage(vec);
        }

        sum += vec_memory_usage(&self.result4);
        sum += vec_memory_usage(&self.result5);
        sum += vec_memory_usage(&self.result6);

        sum
    }

    /// Processes all phases.
    #[inline]
    pub fn process(&mut self, print_progress: bool) {
        self.phase1(print_progress);
        self.phase2(print_progress);
        self.phase3(print_progress);
    }

    /// Processes the phase 1.
    #[inline]
    pub fn phase1(&mut self, print_progress: bool) {
        if print_progress {
            print!("Phase 1/3: Preparing...");
            io::stdout().flush().unwrap();
        }

        self.phase1_prepare();

        while self.progress_percent < 100 {
            if print_progress {
                print!("\rPhase 1/3: {}% completed...", self.progress_percent);
                io::stdout().flush().unwrap();
            }

            self.phase1_proceed_by_percent();
        }

        if print_progress {
            println!("\rPhase 1/3: Done.           ");
        }
    }

    /// Processes the phase 2.
    #[inline]
    pub fn phase2(&mut self, print_progress: bool) {
        if print_progress {
            print!("Phase 2/3: Preparing...");
            io::stdout().flush().unwrap();
        }

        self.phase2_prepare();

        while self.progress_percent < 100 {
            if print_progress {
                print!("\rPhase 2/3: {}% completed...", self.progress_percent);
                io::stdout().flush().unwrap();
            }

            self.phase2_proceed_by_percent();
        }

        if print_progress {
            println!("\rPhase 2/3: Done.           ");
        }
    }

    /// Processes the phase 3.
    #[inline]
    pub fn phase3(&mut self, print_progress: bool) {
        if print_progress {
            print!("Phase 3/3: Preparing...");
            io::stdout().flush().unwrap();
        }

        self.phase3_prepare();

        while self.progress_percent < 100 {
            if print_progress {
                print!("\rPhase 3/3: {}% completed...", self.progress_percent);
                io::stdout().flush().unwrap();
            }

            self.phase3_proceed_by_percent();
        }

        if print_progress {
            println!("\rPhase 3/3: Done.           ");
        }
    }

    /// Manually prepares the phase 1.
    #[inline]
    pub fn phase1_prepare(&mut self) {
        if self.phase != 0 {
            panic!("Invalid state");
        }

        match self.fold_ranges.len() {
            1 => self.phase1_prepare1(),
            2 => self.phase1_prepare2(),
            3 => self.phase1_prepare3(),
            _ => self.phase1_prepare4(),
        }

        self.phase = 1;
        self.progress_percent = 0;
    }

    /// Manually prepares the phase 2.
    #[inline]
    pub fn phase2_prepare(&mut self) {
        if self.phase != 1 || self.progress_percent != 100 {
            panic!("Invalid state");
        }

        self.sum[0] = vec![AtomicF64::new(0.0)];
        self.sum[1] = (0..COMB_49_1).map(|_| AtomicF64::new(0.0)).collect();

        if self.fold_ranges.len() >= 2 {
            self.sum[2] = (0..COMB_49_2).map(|_| AtomicF64::new(0.0)).collect();
            self.sum[3] = (0..COMB_49_3).map(|_| AtomicF64::new(0.0)).collect();
        }

        if self.fold_ranges.len() >= 3 {
            self.sum[4] = (0..COMB_49_4).map(|_| AtomicF64::new(0.0)).collect();
            self.sum[5] = (0..COMB_49_5).map(|_| AtomicF64::new(0.0)).collect();
        }

        if self.fold_ranges.len() == 4 {
            self.sum[6] = (0..COMB_49_6).map(|_| AtomicF64::new(0.0)).collect();
        }

        self.phase = 2;
        self.progress_percent = 0;
    }

    /// Manually prepares the phase 3.
    #[inline]
    pub fn phase3_prepare(&mut self) {
        if self.phase != 2 || self.progress_percent != 100 {
            panic!("Invalid state");
        }

        self.result4 = (0..COMB_49_4).map(|_| AtomicF32::new(0.0)).collect();
        self.result5 = (0..COMB_49_5).map(|_| AtomicF32::new(0.0)).collect();
        self.result6 = (0..COMB_49_6).map(|_| AtomicF32::new(0.0)).collect();

        self.phase = 3;
        self.progress_percent = 0;
    }

    /// Manually proceeds the phase 1 by one percent.
    #[inline]
    pub fn phase1_proceed_by_percent(&mut self) {
        if self.phase != 1 || self.progress_percent == 100 {
            panic!("Invalid state");
        }

        match self.fold_ranges.len() {
            1 => self.phase1_process1(),
            2 => self.phase1_process::<4>(),
            3 => self.phase1_process::<6>(),
            _ => self.phase1_process::<8>(),
        }

        self.progress_percent += 1;

        if self.progress_percent == 100 {
            self.temp_table1 = Vec::new();
            self.temp_table2 = Vec::new();
        }
    }

    /// Manually proceeds the phase 2 by one percent.
    #[inline]
    pub fn phase2_proceed_by_percent(&mut self) {
        if self.phase != 2 || self.progress_percent == 100 {
            panic!("Invalid state");
        }

        match self.fold_ranges.len() {
            1 => self.phase2_process::<2>(),
            2 => self.phase2_process::<4>(),
            3 => self.phase2_process::<6>(),
            _ => self.phase2_process::<8>(),
        }

        self.progress_percent += 1;

        if self.progress_percent == 100 && self.fold_ranges.len() == 4 {
            self.temp_table3 = Vec::new();
        }
    }

    /// Manually proceeds the phase 3 by one percent.
    #[inline]
    pub fn phase3_proceed_by_percent(&mut self) {
        if self.phase != 3 || self.progress_percent == 100 {
            panic!("Invalid state");
        }

        if self.progress_percent == 0 {
            self.phase3_process::<4>(0, COMB_49_4);
        } else if self.progress_percent < 7 {
            let start = (COMB_49_5 as f64 * (self.progress_percent - 1) as f64 / 6.0) as usize;
            let end = (COMB_49_5 as f64 * self.progress_percent as f64 / 6.0) as usize;
            self.phase3_process::<5>(start, end);
        } else {
            let start = (COMB_49_6 as f64 * (self.progress_percent - 7) as f64 / 93.0) as usize;
            let end = (COMB_49_6 as f64 * (self.progress_percent - 6) as f64 / 93.0) as usize;
            self.phase3_process::<6>(start, end);
        }

        self.progress_percent += 1;

        if self.progress_percent == 100 {
            self.sum.iter_mut().for_each(|s| {
                *s = Vec::new();
            });
        }
    }

    pub(crate) fn result_4cards(&self, mask: u64) -> f32 {
        let index = mask_to_index(compress_mask(mask, self.flop), 4);
        self.result4[index].load()
    }

    pub(crate) fn result_5cards(&self, mask: u64) -> f32 {
        let index = mask_to_index(compress_mask(mask, self.flop), 5);
        self.result5[index].load()
    }

    pub(crate) fn result_6cards(&self, mask: u64) -> f32 {
        let index = mask_to_index(compress_mask(mask, self.flop), 6);
        self.result6[index].load()
    }

    /* Phase 1: Preparation */

    fn phase1_prepare1(&mut self) {
        self.temp_table1 = vec![0.0; COMB_49_2];
        Self::phase1_compress(&mut self.temp_table1, &self.fold_ranges[0], self.flop);
        self.sum[2] = (0..COMB_49_2).map(|_| AtomicF64::new(0.0)).collect();
    }

    fn phase1_prepare2(&mut self) {
        self.temp_table1 = vec![0.0; COMB_49_2];
        self.temp_table2 = vec![0.0; COMB_49_2];

        Self::phase1_compress(&mut self.temp_table1, &self.fold_ranges[0], self.flop);
        Self::phase1_compress(&mut self.temp_table2, &self.fold_ranges[1], self.flop);

        self.sum[4] = (0..COMB_49_4).map(|_| AtomicF64::new(0.0)).collect();
    }

    fn phase1_prepare3(&mut self) {
        self.temp_table1 = vec![0.0; COMB_49_4];
        self.temp_table2 = vec![0.0; COMB_49_2];

        Self::phase1_combine(
            &mut self.temp_table1,
            &self.fold_ranges[0],
            &self.fold_ranges[1],
            self.flop,
        );

        Self::phase1_compress(&mut self.temp_table2, &self.fold_ranges[2], self.flop);

        self.sum[6] = (0..COMB_49_6).map(|_| AtomicF64::new(0.0)).collect();
    }

    fn phase1_prepare4(&mut self) {
        self.temp_table1 = vec![0.0; COMB_49_4];
        self.temp_table2 = vec![0.0; COMB_49_4];

        Self::phase1_combine(
            &mut self.temp_table1,
            &self.fold_ranges[0],
            &self.fold_ranges[1],
            self.flop,
        );

        Self::phase1_combine(
            &mut self.temp_table2,
            &self.fold_ranges[2],
            &self.fold_ranges[3],
            self.flop,
        );

        self.temp_table3 = (0..COMB_49_8).map(|_| AtomicF64::new(0.0)).collect();
    }

    fn phase1_compress(table: &mut [f64], range: &Range, flop: [Card; 3]) {
        let range = range.raw_data();
        let flop_mask: u64 = flop.iter().map(|&c| 1 << c).sum();

        let mut src_index = 0;

        for card1 in 0..52 {
            let mask1 = 1 << card1;
            if flop_mask & mask1 != 0 {
                src_index += 51 - card1;
                continue;
            }

            for card2 in card1 + 1..52 {
                let freq = range[src_index] as f64;
                src_index += 1;

                let mask2 = 1 << card2;
                if flop_mask & mask2 != 0 || freq == 0.0 {
                    continue;
                }

                let index = mask_to_index(compress_mask(mask1 | mask2, flop), 2);
                table[index] = freq;
            }
        }
    }

    fn phase1_combine(table: &mut [f64], range1: &Range, range2: &Range, flop: [Card; 3]) {
        let range1 = range1.raw_data();
        let range2 = range2.raw_data();
        let flop_mask: u64 = flop.iter().map(|&c| 1 << c).sum();

        let mut src_index1 = 0;

        for card11 in 0..52 {
            let mask11 = 1 << card11;
            if flop_mask & mask11 != 0 {
                src_index1 += 51 - card11;
                continue;
            }

            for card12 in card11 + 1..52 {
                let freq1 = range1[src_index1] as f64;
                src_index1 += 1;

                let mask12 = 1 << card12;
                if flop_mask & mask12 != 0 || freq1 == 0.0 {
                    continue;
                }

                let mask1 = mask11 | mask12;
                let mut src_index2 = 0;

                for card21 in 0..52 {
                    let mask21 = 1 << card21;
                    if (flop_mask | mask1) & mask21 != 0 {
                        src_index2 += 51 - card21;
                        continue;
                    }

                    let mask2 = mask1 | mask21;
                    for card22 in card21 + 1..52 {
                        let freq2 = range2[src_index2] as f64;
                        src_index2 += 1;

                        let mask22 = 1 << card22;
                        if (flop_mask | mask1) & mask22 != 0 || freq2 == 0.0 {
                            continue;
                        }

                        let mask = mask2 | mask22;
                        let index = mask_to_index(compress_mask(mask, flop), 4);
                        table[index] += freq1 * freq2;
                    }
                }
            }
        }
    }

    /* Phase 1: Main process */

    fn phase1_process1(&mut self) {
        if self.progress_percent != 0 {
            return;
        }

        self.sum[2]
            .iter_mut()
            .zip(self.temp_table1.iter())
            .for_each(|(dst, &src)| {
                dst.store(src);
            });
    }

    fn phase1_process<const K: usize>(&mut self) {
        let (k1, k2, src_len1, src_len2, dst_table) = match K {
            4 => (2, 2, COMB_49_2, COMB_49_2, &mut self.sum[4]),
            6 => (4, 2, COMB_49_4, COMB_49_2, &mut self.sum[6]),
            8 => (4, 4, COMB_49_4, COMB_49_4, &mut self.temp_table3),
            _ => unreachable!(),
        };

        let start_index = (src_len1 as f64 * self.progress_percent as f64 / 100.0) as usize;
        let end_index = (src_len1 as f64 * (self.progress_percent + 1) as f64 / 100.0) as usize;

        into_par_iter(start_index..end_index).for_each(|src_index1| {
            let freq1 = self.temp_table1[src_index1];
            if freq1 == 0.0 {
                return;
            }

            let mask1 = index_to_mask(src_index1, k1);
            let mut mask2 = (1 << k2) - 1;

            for src_index2 in 0..src_len2 {
                if mask1 & mask2 == 0 {
                    let freq2 = self.temp_table2[src_index2];
                    if freq2 > 0.0 {
                        let mask = mask1 | mask2;
                        let dst_index = mask_to_index(mask, K);
                        dst_table[dst_index].add(freq1 * freq2);
                    }
                }
                mask2 = next_combination(mask2);
            }
        });
    }

    /* Phase 2: Main process */

    fn phase2_process<const K: usize>(&mut self) {
        let (src_len, src_table) = match K {
            2 => (COMB_49_2, &self.sum[2]),
            4 => (COMB_49_4, &self.sum[4]),
            6 => (COMB_49_6, &self.sum[6]),
            8 => (COMB_49_8, &self.temp_table3),
            _ => unreachable!(),
        };

        let start_index = (src_len as f64 * self.progress_percent as f64 / 100.0) as usize;
        let end_index = (src_len as f64 * (self.progress_percent + 1) as f64 / 100.0) as usize;

        let num_ones = (0u32..(1 << K) - 1)
            .map(|i| i.count_ones() as u8)
            .collect::<Vec<_>>();

        into_par_iter(start_index..end_index)
            .step_by(100)
            .for_each(|chunk_start_index| {
                let chunk_end_index = usize::min(chunk_start_index + 100, end_index);
                let mut src_mask = index_to_mask(chunk_start_index, K);

                for src_index in chunk_start_index..chunk_end_index {
                    let mut src_mask_copy = src_mask;
                    src_mask = next_combination(src_mask);

                    let freq = src_table[src_index].load();
                    if freq == 0.0 {
                        continue;
                    }

                    let mut src_mask_bit = [0; K];
                    for i in 0..K {
                        let lsb = src_mask_copy & src_mask_copy.wrapping_neg();
                        src_mask_copy ^= lsb;
                        src_mask_bit[i] = lsb;
                    }

                    for i in 0..(1 << K) - 1 {
                        if num_ones[i] > 6 {
                            continue;
                        }

                        let mut dst_mask = 0;
                        for j in 0..K {
                            if i & (1 << j) != 0 {
                                dst_mask |= src_mask_bit[j];
                            }
                        }

                        let dst_index = mask_to_index(dst_mask, num_ones[i] as usize);
                        self.sum[num_ones[i] as usize][dst_index].add(freq);
                    }
                }
            });
    }

    /* Phase 3: Main process */

    fn phase3_process<const N: usize>(&mut self, start_index: usize, end_index: usize) {
        let dst_table = match N {
            4 => &mut self.result4,
            5 => &mut self.result5,
            6 => &mut self.result6,
            _ => unreachable!(),
        };

        let mut indices = (0u8..(1 << N))
            .map(|i| (i, i.count_ones() as u8))
            .collect::<Vec<_>>();
        indices.retain(|&(_, num_ones)| num_ones <= 2 * self.fold_ranges.len() as u8);
        indices.sort_by_key(|&(_, num_ones)| std::cmp::Reverse(num_ones));

        into_par_iter(start_index..end_index)
            .step_by(100)
            .for_each(|dst_start_index| {
                let dst_end_index = usize::min(dst_start_index + 100, end_index);
                let mut mask = index_to_mask(dst_start_index, N);

                for dst_index in dst_start_index..dst_end_index {
                    let mut mask_copy = mask;
                    mask = next_combination(mask);

                    let mut mask_bit = [0; N];
                    for i in 0..N {
                        let lsb = mask_copy & mask_copy.wrapping_neg();
                        mask_copy ^= lsb;
                        mask_bit[i] = lsb;
                    }

                    let mut result = 0.0;

                    for &(i, k) in &indices {
                        let mut src_mask = 0;
                        for j in 0..N {
                            if i & (1 << j) != 0 {
                                src_mask |= mask_bit[j];
                            }
                        }

                        let src_index = mask_to_index(src_mask, k as usize);
                        if k & 1 == 0 {
                            result += self.sum[k as usize][src_index].load();
                        } else {
                            result -= self.sum[k as usize][src_index].load();
                        }
                    }

                    dst_table[dst_index].store(f32::max(result as f32, 0.0));
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_combination() {
        let seq = [
            0b001111, 0b010111, 0b011011, 0b011101, 0b011110, 0b100111, 0b101011, 0b101101,
            0b101110, 0b110011, 0b110101, 0b110110, 0b111001, 0b111010, 0b111100,
        ];

        let mut mask = 0b001111;
        for i in 0..15 {
            assert_eq!(mask, seq[i]);
            mask = next_combination(mask);
        }
    }

    #[test]
    fn test_compress_mask() {
        let x = (1 << 14) | (1 << 16) | (1 << 24) | (1 << 26);
        let y = compress_mask(x, [5, 15, 25]);
        assert_eq!(y, (1 << 13) | (1 << 14) | (1 << 22) | (1 << 23));
    }

    #[test]
    fn test_comb_4() {
        let mut mask = 0b1111;
        for i in 0..COMB_49_4 {
            assert_eq!(i, mask_to_index(mask, 4));
            assert_eq!(mask, index_to_mask(i, 4));
            mask = next_combination(mask);
        }
    }

    #[test]
    fn test_comb_5() {
        let mut mask = 0b11111;
        for i in 0..COMB_49_5 {
            assert_eq!(i, mask_to_index(mask, 5));
            assert_eq!(mask, index_to_mask(i, 5));
            mask = next_combination(mask);
        }
    }

    #[test]
    fn test_comb_6() {
        let mut mask = 0b111111;
        for i in 0..COMB_49_6 {
            assert_eq!(i, mask_to_index(mask, 6));
            assert_eq!(mask, index_to_mask(i, 6));
            mask = next_combination(mask);
        }
    }

    #[test]
    fn test_comb_8() {
        let mut mask = 0b11111111;
        for i in 0..10000 {
            assert_eq!(i, mask_to_index(mask, 8));
            assert_eq!(mask, index_to_mask(i, 8));
            mask = next_combination(mask);
        }
    }

    #[test]
    fn test_bunching_independent_1() {
        let range1 = "33+,A3+,K3+,Q3+,J3+,T3+,93+,83+,73+,63+,53+,43+,33";
        let flop = flop_from_str("2s2h2d").unwrap();
        let mut bunching = BunchingData::new(&[range1.parse().unwrap()], flop).unwrap();

        bunching.phase1(false);
        bunching.phase2(false);

        // equality is exact because the result is an integer (< 2^53)
        assert_eq!(bunching.sum[0][0].load(), 48.0 * 47.0 / 2.0);

        for (i, a) in bunching.sum[1].iter().enumerate() {
            if i == 0 {
                assert_eq!(a.load(), 0.0); // 2c
            } else {
                assert_eq!(a.load(), 47.0);
            }
        }

        bunching.phase3(false);

        let mut mask4 = 0b1111;
        for i in 0..COMB_49_4 {
            let ans = if mask4 & 1 == 0 {
                44.0 * 43.0 / 2.0
            } else {
                45.0 * 44.0 / 2.0
            };
            assert_eq!(bunching.result4[i].load(), ans);
            mask4 = next_combination(mask4);
        }

        let mut mask5 = 0b11111;
        for i in 0..COMB_49_5 {
            let ans = if mask5 & 1 == 0 {
                43.0 * 42.0 / 2.0
            } else {
                44.0 * 43.0 / 2.0
            };
            assert_eq!(bunching.result5[i].load(), ans);
            mask5 = next_combination(mask5);
        }

        let mut mask6 = 0b111111;
        for i in 0..COMB_49_6 {
            let ans = if mask6 & 1 == 0 {
                42.0 * 41.0 / 2.0
            } else {
                43.0 * 42.0 / 2.0
            };
            assert_eq!(bunching.result6[i].load(), ans);
            mask6 = next_combination(mask6);
        }
    }

    #[test]
    fn test_bunching_independent_2() {
        let range1 = "77,76,75,74,73,72,66,65,64,63,62,55,54,53,52,44,43,42,33,32,22";
        let range2 = "AA,AK,AQ,AJ,AT,A9,KK,KQ,KJ,KT,K9,QQ,QJ,QT,Q9,JJ,JT,J9,TT,T9,99";

        let mut bunching = BunchingData::new(
            &[range1.parse().unwrap(), range2.parse().unwrap()],
            flop_from_str("8s8h8d").unwrap(),
        )
        .unwrap();

        bunching.phase1(false);
        bunching.phase2(false);

        assert_eq!(bunching.sum[0][0].load(), f64::powi(24.0 * 23.0 / 2.0, 2));

        for (i, a) in bunching.sum[1].iter().enumerate() {
            if i == 24 {
                assert_eq!(a.load(), 0.0); // 8c
            } else {
                assert_eq!(a.load(), 24.0 * 23.0 / 2.0 * 23.0);
            }
        }

        bunching.phase3(false);

        assert_eq!(
            bunching.result4[0].load(),
            (20.0 * 19.0 / 2.0) * 24.0 * 23.0 / 2.0
        );
        assert_eq!(
            bunching.result5[0].load(),
            (19.0 * 18.0 / 2.0) * 24.0 * 23.0 / 2.0
        );
        assert_eq!(
            bunching.result6[0].load(),
            (18.0 * 17.0 / 2.0) * 24.0 * 23.0 / 2.0
        );
    }

    #[test]
    fn test_bunching_independent_3() {
        let range1 = "55,54,53,52,44,43,42,33,32,22";
        let range2 = "99,98,97,96,88,87,86,77,76,66";
        let range3 = "KK,KQ,KJ,KT,QQ,QJ,QT,JJ,JT,TT";

        let mut bunching = BunchingData::new(
            &[
                range1.parse().unwrap(),
                range2.parse().unwrap(),
                range3.parse().unwrap(),
            ],
            flop_from_str("AsAhAd").unwrap(),
        )
        .unwrap();

        bunching.phase1(false);
        bunching.phase2(false);

        assert_eq!(bunching.sum[0][0].load(), f64::powi(16.0 * 15.0 / 2.0, 3));

        for (i, a) in bunching.sum[1].iter().enumerate() {
            if i == 48 {
                assert_eq!(a.load(), 0.0); // Ac
            } else {
                assert_eq!(a.load(), f64::powi(16.0 * 15.0 / 2.0, 2) * 15.0);
            }
        }

        bunching.phase3(false);

        assert_eq!(
            bunching.result4[0].load(),
            (12.0 * 11.0 / 2.0) * f32::powi(16.0 * 15.0 / 2.0, 2)
        );
        assert_eq!(
            bunching.result5[0].load(),
            (11.0 * 10.0 / 2.0) * f32::powi(16.0 * 15.0 / 2.0, 2)
        );
        assert_eq!(
            bunching.result6[0].load(),
            (10.0 * 9.0 / 2.0) * f32::powi(16.0 * 15.0 / 2.0, 2)
        );
    }

    #[test]
    #[ignore]
    fn test_bunching_independent_4() {
        let range1 = "44,43,42,33,32,22";
        let range2 = "77,76,75,66,65,55";
        let range3 = "JJ,JT,J9,TT,T9,99";
        let range4 = "AA,AK,AQ,KK,KQ,QQ";

        let mut bunching = BunchingData::new(
            &[
                range1.parse().unwrap(),
                range2.parse().unwrap(),
                range3.parse().unwrap(),
                range4.parse().unwrap(),
            ],
            flop_from_str("8s8h8d").unwrap(),
        )
        .unwrap();

        bunching.phase1(true);
        bunching.phase2(true);

        assert_eq!(bunching.sum[0][0].load(), f64::powi(12.0 * 11.0 / 2.0, 4));

        for (i, a) in bunching.sum[1].iter().enumerate() {
            if i == 24 {
                assert_eq!(a.load(), 0.0); // 8c
            } else {
                assert_eq!(a.load(), f64::powi(12.0 * 11.0 / 2.0, 3) * 11.0);
            }
        }

        bunching.phase3(true);

        assert_eq!(
            bunching.result4[0].load(),
            (8.0 * 7.0 / 2.0) * f32::powi(12.0 * 11.0 / 2.0, 3)
        );
        assert_eq!(
            bunching.result5[0].load(),
            (7.0 * 6.0 / 2.0) * f32::powi(12.0 * 11.0 / 2.0, 3)
        );
        assert_eq!(
            bunching.result6[0].load(),
            (6.0 * 5.0 / 2.0) * f32::powi(12.0 * 11.0 / 2.0, 3)
        );
    }

    #[test]
    #[ignore]
    fn test_bunching_wizard() {
        // from GTO Wizard (6max/NL50/General/100bb/Small) (2023-03-06)
        let utg_range = "55:0.38,44:0.78,33:0.8,22:0.775,A2s:0.34,A9o-A2o,K8s:0.015,K6s:0.245,K5s:0.455,K4s-K2s,KJo:0.26,KTo:0.93,K9o-K2o,Q8s:0.59,Q7s-Q2s,QJo:0.48,QTo:0.875,Q9o-Q2o,J8s-J2s,J2o+,T8s:0.59,T7s-T2s,T2o+,98s:0.485,97s:0.985,96s-92s,92o+,87s:0.72,86s:0.995,85s-82s,82o+,76s:0.715,75s-72s,72o+,65s:0.64,64s-62s,62o+,54s:0.77,53s-52s,52o+,42+,32";
        let mp_range = "44:0.435,33:0.725,22:0.72,A9o:0.605,A8o-A2o,K4s:0.73,K3s-K2s,KTo:0.43,K9o-K2o,Q7s:0.935,Q6s:0.555,Q5s-Q2s,QTo:0.51,Q9o-Q2o,J8s:0.42,J7s-J2s,JTo:0.65,J9o-J2o,T7s-T2s,T2o+,97s:0.375,96s-92s,92o+,87s:0.175,86s:0.945,85s-82s,82o+,76s:0.575,75s-72s,72o+,65s:0.445,64s-62s,62o+,54s:0.7,53s-52s,52o+,42+,32";
        let co_range = "33:0.59,22:0.635,A8o:0.265,A7o-A6o,A5o:0.445,A4o-A2o,K2s,K9o:0.905,K8o-K2o,Q4s-Q2s,Q9o-Q2o,J6s-J2s,J9o:0.88,J8o-J2o,T7s:0.405,T6s-T2s,T9o:0.96,T8o-T2o,96s-92s,92o+,86s:0.57,85s-82s,82o+,76s:0.37,75s-72s,72o+,65s:0.475,64s-62s,62o+,54s:0.68,53s-52s,52o+,42+,32";
        let sb_range = "66:0.46,55:0.821,44:0.92,33:0.93,22:0.925,A6s:0.73,A3s:0.47,A2s,ATo:0.105,A9o-A2o,K8s:0.795,K7s,K6s:0.85,K5s:0.965,K4s-K2s,KJo:0.085,KTo:0.645,K9o-K2o,Q8s-Q2s,QJo:0.765,QTo-Q2o,J8s-J2s,J2o+,T8s:0.69,T7s-T2s,T2o+,98s:0.905,97s-92s,92o+,87s:0.78,86s-82s,82o+,76s:0.77,75s-72s,72o+,65s:0.845,64s-62s,62o+,54s:0.735,53s-52s,52o+,42+,32";

        let flop = flop_from_str("QsJh2h").unwrap();
        let mut bunching = BunchingData::new(
            &[
                utg_range.parse().unwrap(),
                mp_range.parse().unwrap(),
                co_range.parse().unwrap(),
                sb_range.parse().unwrap(),
            ],
            flop,
        )
        .unwrap();

        bunching.process(true);
    }
}
