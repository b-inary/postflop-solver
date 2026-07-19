use postflop_solver::*;
use std::collections::HashMap;
use std::time::Instant;

/// 手牌桶分类（用户定义的 7 桶系统）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HandBucket {
    /// 桶1: 超强成牌 — Set、两对、超强超对（AA/KK 在 J 高及以下牌面）
    MonsterMade,
    /// 桶2: 中等顶对 — 顶对好踢脚
    GoodTopPair,
    /// 桶3: 中等成牌/抓 bluff — 顶对弱踢脚、中对、底对、弱超对
    MediumMade,
    /// 桶4: 超强强听牌 — FD+SD 双抽、带对+前门花
    MonsterDraw,
    /// 桶5: 常规强听牌 — 纯 FD、纯 OESD、带对+OESD
    StrongDraw,
    /// 桶6: 弱听牌/后门牌 — Gutshot、后门花
    WeakDraw,
    /// 桶7: 纯空气 — 无对子无听牌
    Trash,
}

impl HandBucket {
    fn classify(
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

        let is_front_draw = matches!(draw, DrawType::FlushDraw | DrawType::OESD | DrawType::FDPlusSD);

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
            HandCategory::Monster => HandBucket::MonsterMade, // Set、两对、顺子、同花等
            HandCategory::Overpair => {
                // 超强超对 (AA/KK) vs 弱超对 (QQ/JJ/TT...)
                let pair_rank = hero.0.max(hero.1) / 4;
                if pair_rank >= 11 {
                    // K=11, A=12: 超强超对
                    HandBucket::MonsterMade
                } else {
                    HandBucket::MediumMade
                }
            }
            HandCategory::TopPairGoodKicker => HandBucket::GoodTopPair,
            HandCategory::TopPairBadKicker
            | HandCategory::MediumPair
            | HandCategory::BottomPair => HandBucket::MediumMade,
            HandCategory::TwoOvercards | HandCategory::Air => {
                // 无对子：按听牌类型分桶
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

    fn label(&self) -> &'static str {
        match self {
            HandBucket::MonsterMade => "桶1:超强成牌",
            HandBucket::GoodTopPair => "桶2:中等顶对",
            HandBucket::MediumMade => "桶3:中等成牌",
            HandBucket::MonsterDraw => "桶4:超强听牌",
            HandBucket::StrongDraw => "桶5:常规强听牌",
            HandBucket::WeakDraw => "桶6:弱听牌",
            HandBucket::Trash => "桶7:纯空气",
        }
    }

    fn order(&self) -> u8 {
        match self {
            HandBucket::MonsterMade => 0,
            HandBucket::GoodTopPair => 1,
            HandBucket::MediumMade => 2,
            HandBucket::MonsterDraw => 3,
            HandBucket::StrongDraw => 4,
            HandBucket::WeakDraw => 5,
            HandBucket::Trash => 6,
        }
    }
}

#[derive(Debug, Default)]
struct BucketData {
    count: usize,
    sum_full: f64,
    sum_dl: f64,
    sum_sq_diff: f64,
    sum_abs_diff: f64,
    max_abs_diff: f64,
    // 加权 EV（按权重）
    sum_full_weighted: f64,
    sum_dl_weighted: f64,
    sum_weight: f64,
}

impl BucketData {
    fn add(&mut self, full_ev: f64, dl_ev: f64, weight: f64) {
        let diff = dl_ev - full_ev;
        self.count += 1;
        self.sum_full += full_ev;
        self.sum_dl += dl_ev;
        self.sum_sq_diff += diff * diff;
        self.sum_abs_diff += diff.abs();
        self.max_abs_diff = self.max_abs_diff.max(diff.abs());
        self.sum_full_weighted += full_ev * weight;
        self.sum_dl_weighted += dl_ev * weight;
        self.sum_weight += weight;
    }

    fn mean_full(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_full / self.count as f64 }
    }
    fn mean_dl(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_dl / self.count as f64 }
    }
    fn mean_bias(&self) -> f64 {
        self.mean_dl() - self.mean_full()
    }
    fn mae(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_abs_diff / self.count as f64 }
    }
    fn mult_correction(&self) -> f64 {
        let md = self.mean_dl();
        if md.abs() < 0.01 { f64::NAN } else { self.mean_full() / md }
    }
    fn add_correction(&self) -> f64 {
        self.mean_full() - self.mean_dl()
    }
    fn weighted_mean_full(&self) -> f64 {
        if self.sum_weight == 0.0 { 0.0 } else { self.sum_full_weighted / self.sum_weight }
    }
    fn weighted_mean_dl(&self) -> f64 {
        if self.sum_weight == 0.0 { 0.0 } else { self.sum_dl_weighted / self.sum_weight }
    }
}

fn main() {
    let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
    let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("60%, e, a", "2.5x")).unwrap();
    let base_tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 200,
        effective_stack: 900,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };
    let target_exp = base_tree_config.starting_pot as f32 * 0.005;

    println!("=== Full Solve ===");
    let t = Instant::now();
    let mut game_full = PostFlopGame::with_config(
        card_config.clone(),
        ActionTree::new(base_tree_config.clone()).unwrap(),
    ).unwrap();
    game_full.allocate_memory(false);
    let exp_full = solve(&mut game_full, 2000, target_exp, true);
    println!("时间: {:?}, 可利用度: {:.4}\n", t.elapsed(), exp_full);

    println!("=== DL Solve (δ=0.05, 启用 BucketCorrection) ===");
    let t = Instant::now();
    let mut dl_config = base_tree_config.clone();
    dl_config.depth_limit = Some(BoardState::Flop);
    dl_config.equity_pos_correction = 0.05;
    dl_config.bucket_correction = Some(BucketCorrection::default());
    let mut game_dl = PostFlopGame::with_config(
        card_config.clone(),
        ActionTree::new(dl_config).unwrap(),
    ).unwrap();
    game_dl.allocate_memory(false);
    let exp_dl = solve(&mut game_dl, 2000, target_exp, true);
    println!("时间: {:?}, 可利用度: {:.4}\n", t.elapsed(), exp_dl);

    let flop = card_config.flop;

    // === 分桶统计 ===
    // 细粒度：(position, category, draw_type)
    let mut fine_buckets: HashMap<(u8, HandCategory, DrawType), BucketData> = HashMap::new();
    // 粗粒度：(position, HandBucket)
    let mut macro_buckets: HashMap<(u8, HandBucket), BucketData> = HashMap::new();

    for player in 0..2u8 {
        game_dl.back_to_root();
        game_full.back_to_root();
        game_dl.cache_normalized_weights();
        game_full.cache_normalized_weights();

        let cards = game_dl.private_cards(player as usize);
        let ev_dl = game_dl.expected_values(player as usize);
        let ev_full = game_full.expected_values(player as usize);
        let weights = game_full.normalized_weights(player as usize);

        for (i, &(c1, c2)) in cards.iter().enumerate() {
            if weights[i] == 0.0 {
                continue;
            }
            let (cat, draw) = classify_hand((c1, c2), flop, NOT_DEALT);
            let w = weights[i] as f64;

            fine_buckets
                .entry((player, cat, draw))
                .or_default()
                .add(ev_full[i] as f64, ev_dl[i] as f64, w);

            let macro_b = HandBucket::classify(cat, draw, (c1, c2), flop);
            macro_buckets
                .entry((player, macro_b))
                .or_default()
                .add(ev_full[i] as f64, ev_dl[i] as f64, w);
        }
    }

    // === 逐手牌明细 ===
    println!("\n{}", "=".repeat(110));
    println!("逐手牌明细");
    println!("公共牌面: Td9d6h");
    println!("{}", "=".repeat(110));

    println!(
        "\n{:<5} {:>8} {:>14} {:>10} {:>9} {:>9} {:>9} {:>14}",
        "位置", "手牌", "大桶", "Draw", "FullEV", "DLEv", "Diff", "权重%"
    );
    println!("{}", "-".repeat(110));

    // 收集所有手牌数据用于排序
    struct HandRow {
        pos: u8,
        name: String,
        #[allow(dead_code)]
        category: HandCategory,
        #[allow(dead_code)]
        draw: DrawType,
        macro_bucket: HandBucket,
        equity: f32,
        full_ev: f32,
        dl_ev: f32,
        dl_ev_corrected: f32,
        weight_pct: f32,
    }

    let mut all_hands: Vec<HandRow> = Vec::new();

    for player in 0..2u8 {
        game_dl.back_to_root();
        game_full.back_to_root();
        game_dl.cache_normalized_weights();
        game_full.cache_normalized_weights();

        let cards = game_dl.private_cards(player as usize);
        let ev_dl = game_dl.expected_values(player as usize);
        let ev_dl_corrected = game_dl.expected_values_with_bucket_correction(player as usize);
        let ev_full = game_full.expected_values(player as usize);
        let equity = game_full.equity(player as usize);
        let weights = game_full.normalized_weights(player as usize);
        let total_w: f32 = weights.iter().sum();
        let names = holes_to_strings(cards).unwrap();

        for (i, name) in names.iter().enumerate() {
            if weights[i] == 0.0 {
                continue;
            }
            let (c1, c2) = cards[i];
            let (cat, draw) = classify_hand((c1, c2), flop, NOT_DEALT);
            let macro_b = HandBucket::classify(cat, draw, (c1, c2), flop);

            all_hands.push(HandRow {
                pos: player,
                name: name.clone(),
                category: cat,
                draw,
                macro_bucket: macro_b,
                equity: equity[i],
                full_ev: ev_full[i],
                dl_ev: ev_dl[i],
                dl_ev_corrected: ev_dl_corrected[i],
                weight_pct: weights[i] / total_w * 100.0,
            });
        }
    }

    // 先输出 OOP，再输出 IP；各自按 Full EV 从低到高排序
    for target_pos in [0u8, 1u8] {
        let pos_label = if target_pos == 0 { "OOP" } else { "IP" };
        println!("\n--- {} ---", pos_label);

        let mut pos_hands: Vec<&HandRow> = all_hands.iter().filter(|h| h.pos == target_pos).collect();
        pos_hands.sort_by(|a, b| a.full_ev.partial_cmp(&b.full_ev).unwrap());

        for h in &pos_hands {
            let diff = h.dl_ev - h.full_ev;
            println!(
                "{:<5} {:>8} {:>14} {:>10} {:>9.1} {:>9.1} {:>+9.1} {:>10.2}%",
                pos_label,
                h.name,
                h.macro_bucket.label(),
                format!("{:?}", h.draw),
                h.full_ev,
                h.dl_ev,
                diff,
                h.weight_pct,
            );
        }
    }

    // === 输出细粒度表 ===
    println!("\n{}", "=".repeat(130));
    println!("细粒度分桶分析：(HandCategory × DrawType)");
    println!("{}", "=".repeat(130));

    let mut fine_keys: Vec<_> = fine_buckets.keys().cloned().collect();
    fine_keys.sort_by(|a, b| (a.0, a.1 as u8, a.2 as u8).cmp(&(b.0, b.1 as u8, b.2 as u8)));

    println!(
        "\n{:<5} {:>18} {:>10} {:>5} {:>9} {:>9} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "位置", "类型", "Draw", "手数", "FullEV", "DLEv", "Bias", "MAE", "MultCor", "AddCor", "加权Bias"
    );
    println!("{}", "-".repeat(130));

    let mut overall_mae = 0.0f64;
    let mut overall_count = 0usize;

    for &(pos, cat, draw) in &fine_keys {
        let d = &fine_buckets[&(pos, cat, draw)];
        if d.count == 0 { continue; }

        let pos_str = if pos == 0 { "OOP" } else { "IP" };
        let mult = d.mult_correction();
        let mult_str = if mult.is_nan() { "N/A".to_string() } else { format!("{:.3}", mult) };

        let weighted_bias = d.weighted_mean_dl() - d.weighted_mean_full();

        println!(
            "{:<5} {:>18} {:>10} {:>5} {:>9.1} {:>9.1} {:>+8.1} {:>8.1} {:>9} {:>+9.1} {:>+9.1}",
            pos_str,
            format!("{:?}", cat),
            format!("{:?}", draw),
            d.count,
            d.mean_full(),
            d.mean_dl(),
            d.mean_bias(),
            d.mae(),
            mult_str,
            d.add_correction(),
            weighted_bias,
        );

        overall_mae += d.sum_abs_diff;
        overall_count += d.count;
    }

    println!("\n总 MAE: {:.1} ({} 手牌)", overall_mae / overall_count as f64, overall_count);

    // === 输出粗粒度大桶表 ===
    println!("\n\n{}", "=".repeat(120));
    println!("粗粒度大桶分析");
    println!("{}", "=".repeat(120));

    let mut macro_keys: Vec<_> = macro_buckets.keys().cloned().collect();
    macro_keys.sort_by(|a, b| (a.0, a.1.order()).cmp(&(b.0, b.1.order())));

    println!(
        "\n{:<5} {:>14} {:>5} {:>9} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9}",
        "位置", "大桶", "手数", "FullEV", "DLEv", "Bias", "MAE", "MultCor", "加权Full", "加权DL"
    );
    println!("{}", "-".repeat(120));

    for &(pos, bucket) in &macro_keys {
        let d = &macro_buckets[&(pos, bucket)];
        if d.count == 0 { continue; }

        let pos_str = if pos == 0 { "OOP" } else { "IP" };
        let mult = d.mult_correction();
        let mult_str = if mult.is_nan() { "N/A".to_string() } else { format!("{:.3}", mult) };

        println!(
            "{:<5} {:>14} {:>5} {:>9.1} {:>9.1} {:>+8.1} {:>8.1} {:>9} {:>9.1} {:>9.1}",
            pos_str,
            bucket.label(),
            d.count,
            d.mean_full(),
            d.mean_dl(),
            d.mean_bias(),
            d.mae(),
            mult_str,
            d.weighted_mean_full(),
            d.weighted_mean_dl(),
        );
    }

    // === 跨位置汇总大桶 ===
    println!("\n\n{}", "=".repeat(100));
    println!("跨位置汇总大桶（OOP+IP 合并）");
    println!("{}", "=".repeat(100));

    let mut cross_macro: HashMap<HandBucket, BucketData> = HashMap::new();
    for h in &all_hands {
        let entry = cross_macro.entry(h.macro_bucket).or_default();
        entry.add(h.full_ev as f64, h.dl_ev as f64, h.weight_pct as f64);
    }

    let mut cross_keys: Vec<_> = cross_macro.keys().cloned().collect();
    cross_keys.sort_by_key(|k| k.order());

    println!(
        "\n{:<14} {:>5} {:>9} {:>9} {:>8} {:>8} {:>9} {:>8} {:>9} {:>9}",
        "大桶", "手数", "FullEV", "DLEv", "Bias", "MAE", "MultCor", "占比%", "加权Full", "加权DL"
    );
    println!("{}", "-".repeat(100));

    for bucket in &cross_keys {
        let d = &cross_macro[bucket];
        if d.count == 0 { continue; }
        let mult = d.mult_correction();
        let mult_str = if mult.is_nan() { "N/A".to_string() } else { format!("{:.3}", mult) };
        let pct = d.count as f64 / overall_count as f64 * 100.0;

        println!(
            "{:<14} {:>5} {:>9.1} {:>9.1} {:>+8.1} {:>8.1} {:>9} {:>7.1}% {:>9.1} {:>9.1}",
            bucket.label(),
            d.count,
            d.mean_full(),
            d.mean_dl(),
            d.mean_bias(),
            d.mae(),
            mult_str,
            pct,
            d.weighted_mean_full(),
            d.weighted_mean_dl(),
        );
    }

    // === 建议的修正系数表 ===
    println!("\n\n{}", "=".repeat(100));
    println!("建议的 per-bucket 修正系数（基于基准 DL 数据）");
    println!("{}", "=".repeat(100));
    println!("注意：这些系数是\"静态\"修正，未考虑 solver 策略反馈。");
    println!("实际应用时可能需要缩小修正幅度（如 50-70%）以避免过矫正。\n");

    println!("{:<14} {:>9} {:>9} {:>9} {:>9}", "大桶", "MultCor", "50%步长", "70%步长", "AddCor");
    println!("{}", "-".repeat(60));

    for bucket in &cross_keys {
        let d = &cross_macro[bucket];
        if d.count == 0 { continue; }
        let mult = d.mult_correction();
        if mult.is_nan() { continue; }

        // 从 1.0 出发，走 mult_correction 的 50% 和 70%
        let step_50 = 1.0 + (mult - 1.0) * 0.5;
        let step_70 = 1.0 + (mult - 1.0) * 0.7;

        println!(
            "{:<14} {:>9.3} {:>9.3} {:>9.3} {:>+9.1}",
            bucket.label(),
            mult,
            step_50,
            step_70,
            d.add_correction(),
        );
    }

    // === 修正前后 MAE 对比 ===
    println!("\n\n{}", "=".repeat(80));
    println!("BucketCorrection 修正效果对比");
    println!("{}", "=".repeat(80));
    println!(
        "\nSPR = {:.2} (effective_stack=900 / starting_pot=200)",
        900.0 / 200.0
    );
    println!("β(SPR=4.5) = 1.0 (标准底池)");
    println!();

    for target_pos in [0u8, 1u8] {
        let pos_label = if target_pos == 0 { "OOP" } else { "IP" };
        let pos_hands: Vec<&HandRow> = all_hands.iter().filter(|h| h.pos == target_pos).collect();

        let n = pos_hands.len() as f64;
        let mae_before: f64 = pos_hands.iter().map(|h| (h.dl_ev - h.full_ev).abs() as f64).sum::<f64>() / n;
        let mae_after: f64 = pos_hands.iter().map(|h| (h.dl_ev_corrected - h.full_ev).abs() as f64).sum::<f64>() / n;
        let bias_before: f64 = pos_hands.iter().map(|h| (h.dl_ev - h.full_ev) as f64).sum::<f64>() / n;
        let bias_after: f64 = pos_hands.iter().map(|h| (h.dl_ev_corrected - h.full_ev) as f64).sum::<f64>() / n;

        println!(
            "{}: MAE {:.2} → {:.2} ({:+.1}%), Bias {:+.2} → {:+.2}",
            pos_label,
            mae_before,
            mae_after,
            (mae_after - mae_before) / mae_before * 100.0,
            bias_before,
            bias_after,
        );
    }

    // === 输出 CSV 文件 ===
    let csv_path = "dl_bucket_analysis.csv";
    let mut csv = String::new();
    csv.push_str("手牌,手牌桶,玩家位置,Equity,深度限制解算EV,修正后DL_EV,完整GTO解算EV\n");

    // 按位置分组，各自按 Full EV 排序
    for target_pos in [0u8, 1u8] {
        let pos_label = if target_pos == 0 { "OOP" } else { "IP" };
        let mut pos_hands: Vec<&HandRow> = all_hands.iter().filter(|h| h.pos == target_pos).collect();
        pos_hands.sort_by(|a, b| a.full_ev.partial_cmp(&b.full_ev).unwrap());

        for h in &pos_hands {
            csv.push_str(&format!(
                "{},{},{},{:.4},{:.2},{:.2},{:.2}\n",
                h.name,
                h.macro_bucket.label(),
                pos_label,
                h.equity,
                h.dl_ev,
                h.dl_ev_corrected,
                h.full_ev,
            ));
        }
    }

    std::fs::write(csv_path, &csv).expect("Failed to write CSV");
    println!("\nCSV 已保存到: {}", csv_path);
    println!("=== 分析完成 ===");
}
