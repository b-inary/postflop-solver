use postflop_solver::*;
use std::time::Instant;

struct BoardDef {
    id: usize,
    name: &'static str,
    flop_str: &'static str,
    description: &'static str,
}

const BOARDS: [BoardDef; 8] = [
    BoardDef { id: 1, name: "三条面", flop_str: "8h8s8d", description: "8♥8♠8♦ 散色三条" },
    BoardDef { id: 2, name: "对子彩虹", flop_str: "KhKd5c", description: "K♥K♦5♣ 对子+彩虹" },
    BoardDef { id: 3, name: "对子两同花", flop_str: "KhKc5c", description: "K♥K♣5♣ 对子+两同花" },
    BoardDef { id: 4, name: "纯彩虹", flop_str: "QsJh4d", description: "Q♠J♥4♦ 彩虹无对" },
    BoardDef { id: 5, name: "两同花A", flop_str: "QsJs4h", description: "Q♠J♠4♥ 大牌同花" },
    BoardDef { id: 6, name: "两同花B", flop_str: "QhJs4s", description: "Q♥J♠4♠ 小牌同花" },
    BoardDef { id: 7, name: "单调面A", flop_str: "AsJs9s", description: "A♠J♠9♠ 高牌单色" },
    BoardDef { id: 8, name: "单调面B", flop_str: "8s6s3s", description: "8♠6♠3♠ 低牌单色" },
];

#[derive(Clone)]
struct HandDetail {
    position: &'static str,
    name: String,
    bucket_label: &'static str,
    draw: String,
    full_ev: f32,
    dl_ev: f32,
}

struct BoardResult {
    board: &'static BoardDef,
    oop_hands: usize,
    ip_hands: usize,
    full_exploitability: f32,
    dl_exploitability: f32,
    full_time_secs: f64,
    dl_time_secs: f64,
    // Per-position stats
    oop_mae_before: f64,
    oop_mae_v1: f64,
    oop_mae_v2: f64,
    oop_bias_before: f64,
    oop_bias_v1: f64,
    oop_bias_v2: f64,
    ip_mae_before: f64,
    ip_mae_v1: f64,
    ip_mae_v2: f64,
    ip_bias_before: f64,
    ip_bias_v1: f64,
    ip_bias_v2: f64,
    // Per-bucket stats (7 buckets × 2 positions)
    bucket_stats: Vec<BucketStat>,
    hand_details: Vec<HandDetail>,
}

struct BucketStat {
    position: &'static str,
    bucket_label: &'static str,
    count: usize,
    mean_full: f64,
    mean_dl: f64,
    mean_dl_v2: f64,
    bias_before: f64,
    bias_v2: f64,
    mae_before: f64,
    mae_v2: f64,
}

fn run_board(board: &'static BoardDef) -> BoardResult {
    let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
    let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";

    let flop = flop_from_str(board.flop_str).unwrap();

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("60%, e, a", "2.5x")).unwrap();

    // pot = 5BB, stack = 100BB. Use BB=10 for integer pot/stack.
    let starting_pot = 50; // 5BB × 10
    let effective_stack = 1000; // 100BB × 10
    let target_exp = starting_pot as f32 * 0.005;

    let base_tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot,
        effective_stack,
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

    // === Full Solve ===
    let t = Instant::now();
    let mut game_full = PostFlopGame::with_config(
        card_config.clone(),
        ActionTree::new(base_tree_config.clone()).unwrap(),
    ).unwrap();
    game_full.allocate_memory(false);
    let exp_full = solve(&mut game_full, 2000, target_exp, false);
    let full_time = t.elapsed().as_secs_f64();

    // === DL Solve ===
    let t = Instant::now();
    let mut dl_config = base_tree_config.clone();
    dl_config.depth_limit = Some(BoardState::Flop);
    dl_config.equity_pos_correction = 0.05;
    // No BucketCorrection — raw DL EV for training
    let mut game_dl = PostFlopGame::with_config(
        card_config.clone(),
        ActionTree::new(dl_config).unwrap(),
    ).unwrap();
    game_dl.allocate_memory(false);
    let exp_dl = solve(&mut game_dl, 2000, target_exp, false);
    let dl_time = t.elapsed().as_secs_f64();

    // === Collect data ===
    game_full.back_to_root();
    game_dl.back_to_root();
    game_full.cache_normalized_weights();
    game_dl.cache_normalized_weights();

    let mut bucket_stats = Vec::new();
    let mut oop_mae_b = 0.0f64; let mut oop_mae_a = 0.0f64; let mut oop_mae_v2 = 0.0f64;
    let mut oop_bias_b = 0.0f64; let mut oop_bias_a = 0.0f64; let mut oop_bias_v2 = 0.0f64;
    let mut oop_count = 0usize;
    let mut ip_mae_b = 0.0f64; let mut ip_mae_a = 0.0f64; let mut ip_mae_v2 = 0.0f64;
    let mut ip_bias_b = 0.0f64; let mut ip_bias_a = 0.0f64; let mut ip_bias_v2 = 0.0f64;
    let mut ip_count = 0usize;

    // Accumulate per-bucket data: (sum_full, sum_dl, sum_dl_v1, sum_dl_v2, sum_abs_b, sum_abs_v2, count)
    let mut bucket_accum: Vec<[(f64, f64, f64, f64, f64, f64, usize); 7]> = vec![
        [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0); 7]; 2
    ];

    let positions = ["OOP", "IP"];
    let mut hand_details = Vec::new();

    let bucket_labels = [
        "桶1:超强成牌", "桶2:中等顶对", "桶3:中等成牌",
        "桶4:超强听牌", "桶5:常规强听牌", "桶6:弱听牌", "桶7:纯空气",
    ];

    for player in 0..2 {
        let cards = game_dl.private_cards(player);
        let ev_dl = game_dl.expected_values(player);
        let ev_dl_v1 = game_dl.expected_values_with_bucket_correction(player);
        let ev_dl_v2 = game_dl.expected_values_with_board_correction(player);
        let ev_full = game_full.expected_values(player);
        let weights = game_full.normalized_weights(player);
        let names = holes_to_strings(cards).unwrap();

        for (i, &(c1, c2)) in cards.iter().enumerate() {
            if weights[i] == 0.0 { continue; }
            let (cat, draw) = classify_hand((c1, c2), flop, NOT_DEALT);
            let bucket = HandBucket::classify(cat, draw, (c1, c2), flop);
            let bi = bucket as usize;

            let diff_b = (ev_dl[i] - ev_full[i]) as f64;
            let diff_v1 = (ev_dl_v1[i] - ev_full[i]) as f64;
            let diff_v2 = (ev_dl_v2[i] - ev_full[i]) as f64;

            bucket_accum[player][bi].0 += ev_full[i] as f64;
            bucket_accum[player][bi].1 += ev_dl[i] as f64;
            bucket_accum[player][bi].2 += ev_dl_v1[i] as f64;
            bucket_accum[player][bi].3 += ev_dl_v2[i] as f64;
            bucket_accum[player][bi].4 += diff_b.abs();
            bucket_accum[player][bi].5 += diff_v2.abs();
            bucket_accum[player][bi].6 += 1;

            hand_details.push(HandDetail {
                position: positions[player],
                name: names[i].clone(),
                bucket_label: bucket_labels[bi],
                draw: format!("{:?}", draw),
                full_ev: ev_full[i],
                dl_ev: ev_dl[i],
            });

            if player == 0 {
                oop_mae_b += diff_b.abs();
                oop_mae_a += diff_v1.abs();
                oop_mae_v2 += diff_v2.abs();
                oop_bias_b += diff_b;
                oop_bias_a += diff_v1;
                oop_bias_v2 += diff_v2;
                oop_count += 1;
            } else {
                ip_mae_b += diff_b.abs();
                ip_mae_a += diff_v1.abs();
                ip_mae_v2 += diff_v2.abs();
                ip_bias_b += diff_b;
                ip_bias_a += diff_v1;
                ip_bias_v2 += diff_v2;
                ip_count += 1;
            }
        }
    }

    let bucket_labels = [
        "桶1:超强成牌", "桶2:中等顶对", "桶3:中等成牌",
        "桶4:超强听牌", "桶5:常规强听牌", "桶6:弱听牌", "桶7:纯空气",
    ];
    let positions = ["OOP", "IP"];

    for player in 0..2 {
        for bi in 0..7 {
            let (_, _, _, _, _, _, count) = bucket_accum[player][bi];
            if count == 0 { continue; }
            let (sf, sd, _sv1, sv2, sab, sav2, cnt) = bucket_accum[player][bi];
            let n = cnt as f64;
            bucket_stats.push(BucketStat {
                position: positions[player],
                bucket_label: bucket_labels[bi],
                count: cnt,
                mean_full: sf / n,
                mean_dl: sd / n,
                mean_dl_v2: sv2 / n,
                bias_before: (sd - sf) / n,
                bias_v2: (sv2 - sf) / n,
                mae_before: sab / n,
                mae_v2: sav2 / n,
            });
        }
    }

    BoardResult {
        board,
        oop_hands: oop_count,
        ip_hands: ip_count,
        full_exploitability: exp_full,
        dl_exploitability: exp_dl,
        full_time_secs: full_time,
        dl_time_secs: dl_time,
        oop_mae_before: oop_mae_b / oop_count as f64,
        oop_mae_v1: oop_mae_a / oop_count as f64,
        oop_mae_v2: oop_mae_v2 / oop_count as f64,
        oop_bias_before: oop_bias_b / oop_count as f64,
        oop_bias_v1: oop_bias_a / oop_count as f64,
        oop_bias_v2: oop_bias_v2 / oop_count as f64,
        ip_mae_before: ip_mae_b / ip_count as f64,
        ip_mae_v1: ip_mae_a / ip_count as f64,
        ip_mae_v2: ip_mae_v2 / ip_count as f64,
        ip_bias_before: ip_bias_b / ip_count as f64,
        ip_bias_v1: ip_bias_a / ip_count as f64,
        ip_bias_v2: ip_bias_v2 / ip_count as f64,
        bucket_stats,
        hand_details,
    }
}

fn main() {
    println!("多牌面泛化验证：8 种同构翻牌 DL EV vs Full EV");
    println!("配置: pot=5BB, stack=100BB, SPR=20, δ=0.05, BucketCorrection=enabled\n");

    let mut results = Vec::new();
    for board in &BOARDS {
        println!("正在求解 #{} {} ({})...", board.id, board.name, board.description);
        let result = run_board(board);
        println!(
            "  Full: {:.1}s (exploit={:.4}), DL: {:.1}s (exploit={:.4}), OOP hands={}, IP hands={}",
            result.full_time_secs, result.full_exploitability,
            result.dl_time_secs, result.dl_exploitability,
            result.oop_hands, result.ip_hands,
        );
        results.push(result);
    }

    // === Write markdown report ===
    let mut md = String::new();

    md.push_str("# 多牌面泛化验证报告\n\n");
    md.push_str("## 配置\n\n");
    md.push_str("| 参数 | 值 |\n|---|---|\n");
    md.push_str("| 底池 | 5BB |\n");
    md.push_str("| 有效筹码 | 100BB |\n");
    md.push_str("| SPR | 20 (深筹码, β=1.2) |\n");
    md.push_str("| 抽水 | 0% |\n");
    md.push_str("| δ (位置修正) | 0.05 |\n");
    md.push_str("| BucketCorrection | 启用 (默认系数) |\n");
    md.push_str("| 迭代次数 | 2000 |\n");
    md.push_str("| OOP 范围 | 66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s |\n");
    md.push_str("| IP 范围 | QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+ |\n");

    // Summary table
    md.push_str("\n## 总览\n\n");
    md.push_str("| # | 牌面 | 牌型 | OOP原始MAE | OOP v1 | OOP v2 | IP原始MAE | IP v1 | IP v2 |\n");
    md.push_str("|---|------|------|----------|--------|--------|---------|-------|-------|\n");

    for r in &results {
        md.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            r.board.id, r.board.flop_str, r.board.name,
            r.oop_mae_before, r.oop_mae_v1, r.oop_mae_v2,
            r.ip_mae_before, r.ip_mae_v1, r.ip_mae_v2,
        ));
    }

    // Per-board detail
    for r in &results {
        md.push_str(&format!("\n## 牌面 #{}: {} — {}\n\n", r.board.id, r.board.flop_str, r.board.name));
        md.push_str(&format!("{}\n\n", r.board.description));
        md.push_str(&format!("- OOP 手牌数: {}\n", r.oop_hands));
        md.push_str(&format!("- IP 手牌数: {}\n", r.ip_hands));
        md.push_str(&format!("- Full 可利用度: {:.4}\n", r.full_exploitability));
        md.push_str(&format!("- DL 可利用度: {:.4}\n\n", r.dl_exploitability));

        // Position summary
        md.push_str("### 位置汇总\n\n");
        md.push_str("| 位置 | MAE原始 | MAE v1 | MAE v2 | Bias v1 | Bias v2 |\n");
        md.push_str("|------|--------|------|------|--------|--------|\n");

        md.push_str(&format!(
            "| OOP | {:.2} | {:.2} | {:.2} | {:+.2} | {:+.2} |\n",
            r.oop_mae_before, r.oop_mae_v1, r.oop_mae_v2, r.oop_bias_v1, r.oop_bias_v2,
        ));
        md.push_str(&format!(
            "| IP | {:.2} | {:.2} | {:.2} | {:+.2} | {:+.2} |\n",
            r.ip_mae_before, r.ip_mae_v1, r.ip_mae_v2, r.ip_bias_v1, r.ip_bias_v2,
        ));

        // Per-bucket detail
        md.push_str("\n### 分桶明细\n\n");
        md.push_str("| 位置 | 桶 | 手数 | Mean Full | Mean DL | Mean DL v2修正 | Bias前 | Bias v2 | MAE前 | MAE v2 |\n");
        md.push_str("|------|---|------|----------|--------|--------------|-------|--------|------|-------|\n");

        for bs in &r.bucket_stats {
            md.push_str(&format!(
                "| {} | {} | {} | {:.1} | {:.1} | {:.1} | {:+.1} | {:+.1} | {:.1} | {:.1} |\n",
                bs.position, bs.bucket_label, bs.count,
                bs.mean_full, bs.mean_dl, bs.mean_dl_v2,
                bs.bias_before, bs.bias_v2,
                bs.mae_before, bs.mae_v2,
            ));
        }
    }

    let report_path = "doc/多牌面泛化验证报告.md";
    std::fs::write(report_path, &md).expect("Failed to write report");
    println!("\n报告已保存到: {}", report_path);

    // === Output per-hand CSVs for ALL 8 boards ===
    for r in &results {
        let csv_path = format!("doc/board{}_{}_hands.csv", r.board.id, r.board.name);

        let mut csv = String::new();
        csv.push_str(&format!("牌面: {}\n", r.board.flop_str));
        csv.push_str("位置,手牌,大桶,Draw,FullEV,DLEv\n");

        let mut sorted_hands = r.hand_details.clone();
        sorted_hands.sort_by(|a, b| {
            a.position.cmp(b.position)
                .then_with(|| a.full_ev.partial_cmp(&b.full_ev).unwrap())
        });

        for h in &sorted_hands {
            csv.push_str(&format!(
                "{},{},{},{},{:.2},{:.2}\n",
                h.position, h.name, h.bucket_label, h.draw,
                h.full_ev, h.dl_ev,
            ));
        }

        std::fs::write(&csv_path, &csv).expect("Failed to write CSV");
        println!("CSV 已保存: {}", csv_path);
    }
}
