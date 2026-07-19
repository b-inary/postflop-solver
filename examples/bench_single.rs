use clap::Parser;
use postflop_solver::*;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "bench_single", about = "单牌面 Full EV vs DL EV 计算工具")]
struct Args {
    /// 翻牌，如 "8h8s8d"
    #[arg(long)]
    flop: String,

    /// 转牌（可选），如 "Ac"。设置后改为转牌圈模式
    #[arg(long)]
    turn: Option<String>,

    /// 输出 CSV 文件路径
    #[arg(long)]
    output: String,

    /// 底池大小（BB）
    #[arg(long, default_value = "5")]
    pot: i32,

    /// 有效筹码（BB）
    #[arg(long, default_value = "100")]
    stack: i32,

    /// 抽水比例（0.0 ~ 1.0）
    #[arg(long, default_value = "0.0")]
    rake: f64,

    /// OOP 范围（默认使用标准范围）
    #[arg(long, default_value = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s")]
    oop_range: String,

    /// IP 范围（默认使用标准范围）
    #[arg(long, default_value = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+")]
    ip_range: String,

    /// 下注尺寸（格式同 BetSizeOptions）
    #[arg(long, default_value = "60%, e, a")]
    bet_sizes: String,

    /// DL截断阶段: flop(翻牌截断) 或 turn(转牌截断)
    #[arg(long, default_value = "flop")]
    dl_stage: String,

    /// 迭代次数
    #[arg(long, default_value = "2000")]
    iterations: u32,
}

fn board_texture_label(texture: BoardTexture) -> &'static str {
    match texture {
        BoardTexture::Trips => "三条面",
        BoardTexture::PairedRainbow => "对子彩虹",
        BoardTexture::PairedTwoTone => "对子两同花",
        BoardTexture::Rainbow => "纯彩虹",
        BoardTexture::TwoToneA => "两同花A",
        BoardTexture::TwoToneB => "两同花B",
        BoardTexture::MonotoneA => "单调面A",
        BoardTexture::MonotoneB => "单调面B",
    }
}

/// 输出当前节点的策略到 JSON
fn dump_strategy(game: &PostFlopGame, prefix: &str) -> String {
    if game.is_terminal_node() || game.is_chance_node() {
        return String::new();
    }
    let actions = game.available_actions();
    let action_labels: Vec<String> = actions.iter().map(|a| format!("{:?}", a)).collect();
    let strategy = game.strategy();
    let player = game.current_player();
    let cards = game.private_cards(player);
    let names = holes_to_strings(cards).unwrap();
    let num_actions = actions.len();
    let num_hands = names.len();

    let mut out = String::new();
    for (h, name) in names.iter().enumerate() {
        let mut probs = String::new();
        for a in 0..num_actions {
            if a > 0 { probs.push(','); }
            probs.push_str(&format!("{:.4}", strategy[h + a * num_hands]));
        }
        if h > 0 { out.push('\n'); }
        out.push_str(&format!("{}{},{},{}", prefix, name, action_labels.join("|"), probs));
    }
    out
}

fn main() {
    let args = Args::parse();

    let flop = flop_from_str(&args.flop).expect("Invalid flop format. Use e.g. '8h8s8d'");
    let texture = detect_board_texture(flop);
    let texture_label = board_texture_label(texture);

    // 解析转牌
    let turn_card = args.turn.as_ref().map(|s| card_from_str(s).expect("Invalid turn card format"));
    let is_turn_mode = turn_card.is_some();
    let turn = turn_card.unwrap_or(NOT_DEALT);

    // BB unit: multiply by 10 for integer pot/stack
    let starting_pot = args.pot * 10;
    let effective_stack = args.stack * 10;
    let spr = args.stack as f64 / args.pot as f64;

    let card_config = CardConfig {
        range: [args.oop_range.parse().unwrap(), args.ip_range.parse().unwrap()],
        flop,
        turn,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from((args.bet_sizes.as_str(), "2.5x")).unwrap();

    let initial_state = if is_turn_mode { BoardState::Turn } else { BoardState::Flop };
    let dl_limit = if is_turn_mode { BoardState::Turn } else { BoardState::Flop };
    let dl_stage: BoardState = match args.dl_stage.as_str() {
        "turn" => BoardState::Turn,
        _ => BoardState::Flop,
    };

    let base_tree_config = TreeConfig {
        initial_state,
        starting_pot,
        effective_stack,
        rake_rate: args.rake,
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

    let target_exp = starting_pot as f32 * 0.005;

    let mode_str = if is_turn_mode { "转牌圈" } else { "翻牌圈" };
    let turn_str = if is_turn_mode { format!(", turn={}", args.turn.as_ref().unwrap()) } else { String::new() };
    println!("牌面: {}{} ({})", args.flop, turn_str, texture_label);
    println!("模式: {}", mode_str);
    println!("SPR: {:.1} (pot={}BB, stack={}BB)", spr, args.pot, args.stack);

    // === Full Solve ===
    print!("Full solve... ");
    let t = Instant::now();
    let mut game_full = PostFlopGame::with_config(
        card_config.clone(),
        ActionTree::new(base_tree_config.clone()).unwrap(),
    )
    .unwrap();
    game_full.allocate_memory(false);
    let exp_full = solve(&mut game_full, args.iterations, target_exp, false);
    println!("{:.1}s (exploit={:.4})", t.elapsed().as_secs_f64(), exp_full);

    // === DL / 跨街求解 ===
    let t = Instant::now();
    let mut game_dl;

    if is_turn_mode {
        // 阶段1: 翻牌圈 DL 求解 → 提取范围权重
        print!("Flop DL... ");
        let mut dl_config = base_tree_config.clone();
        dl_config.depth_limit = Some(dl_stage);
        dl_config.equity_pos_correction = 0.05;
        let mut game_flop = PostFlopGame::with_config(
            card_config.clone(),
            ActionTree::new(dl_config).unwrap(),
        ).unwrap();
        game_flop.allocate_memory(false);
        let exp_flop = solve(&mut game_flop, args.iterations, target_exp, false);
        print!("{:.1}s → ", t.elapsed().as_secs_f64());

        // 提取翻牌后的归一化权重
        game_flop.back_to_root();
        game_flop.cache_normalized_weights();
        let flop_weights = [
            game_flop.normalized_weights(0).to_vec(),
            game_flop.normalized_weights(1).to_vec(),
        ];
        drop(game_flop);

        // 阶段2: 转牌圈全树展开（继承翻牌权重）
        print!("Turn Full... ");
        let t2 = Instant::now();
        let turn_config = base_tree_config.clone();
        let mut game_turn = PostFlopGame::with_config(
            card_config.clone(),
            ActionTree::new(turn_config).unwrap(),
        ).unwrap();
        game_turn.allocate_memory(false);
        game_turn.set_custom_initial_weights(flop_weights);
        let exp_turn = solve(&mut game_turn, args.iterations, target_exp, false);
        println!("{:.1}s (flop={:.4}, turn={:.4})", t2.elapsed().as_secs_f64(), exp_flop, exp_turn);
        game_dl = game_turn;
    } else {
        print!("DL solve... ");
        let mut dl_config = base_tree_config.clone();
        dl_config.depth_limit = Some(dl_stage);
        dl_config.equity_pos_correction = 0.05;
        game_dl = PostFlopGame::with_config(
            card_config.clone(),
            ActionTree::new(dl_config).unwrap(),
        ).unwrap();
        game_dl.allocate_memory(false);
        let exp_dl = solve(&mut game_dl, args.iterations, target_exp, false);
        println!("{:.1}s (exploit={:.4})", t.elapsed().as_secs_f64(), exp_dl);
    }
    let dl_time = t.elapsed().as_secs_f64();

    // === Collect data ===
    game_full.back_to_root();
    game_dl.back_to_root();
    game_full.cache_normalized_weights();
    game_dl.cache_normalized_weights();

    let mut csv_lines = Vec::new();
    csv_lines.push(format!("牌面: {}", args.flop));
    csv_lines.push(format!("牌面类型: {}", texture_label));
    csv_lines.push(format!("SPR: {:.1}", spr));
    csv_lines.push(
        "位置,手牌,大桶,Draw,FullEV,DLEv,DLEv_corrected,DLEv偏差%,修正后偏差%,修正结果".to_string(),
    );

    let bucket_labels = [
        "桶1:超强成牌", "桶2:中等顶对", "桶3:中等成牌",
        "桶4:超强听牌", "桶5:常规强听牌", "桶6:弱听牌", "桶7:纯空气",
    ];
    let positions = ["OOP", "IP"];

    for player in 0..2 {
        let cards = game_dl.private_cards(player);
        let ev_dl = game_dl.expected_values(player);
        let ev_corrected = game_dl.expected_values_with_board_correction(player);
        let ev_full = game_full.expected_values(player);
        let weights = game_full.normalized_weights(player);
        let names = holes_to_strings(cards).unwrap();

        // Collect and sort by full EV
        let mut rows: Vec<(String, String, String, f32, f32, f32)> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            if weights[i] == 0.0 { continue; }
            let (c1, c2) = cards[i];
            let (cat, draw) = classify_hand((c1, c2), flop, NOT_DEALT);
            let bucket = HandBucket::classify(cat, draw, (c1, c2), flop);
            rows.push((
                name.clone(),
                bucket_labels[bucket as usize].to_string(),
                format!("{:?}", draw),
                ev_full[i],
                ev_dl[i],
                ev_corrected[i],
            ));
        }
        rows.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

        for (name, bucket, draw, full, dl, corrected) in &rows {
            let dl_bias = if full.abs() > 0.01 {
                (dl - full) / full * 100.0
            } else {
                0.0
            };
            let corr_bias = if full.abs() > 0.01 {
                (corrected - full) / full * 100.0
            } else {
                0.0
            };
            let result = if corr_bias.abs() < dl_bias.abs() - 0.01 {
                "改善"
            } else if corr_bias.abs() > dl_bias.abs() + 0.01 {
                "恶化"
            } else {
                "没改变"
            };

            csv_lines.push(format!(
                "{},{},{},{},{:.2},{:.2},{:.2},{:+.1}%,{:+.1}%,{}",
                positions[player], name, bucket, draw, full, dl, corrected, dl_bias, corr_bias, result,
            ));
        }
    }

    let csv_content = csv_lines.join("\n") + "\n";
    std::fs::write(&args.output, &csv_content).expect("Failed to write CSV");
    println!("CSV 已保存: {}", args.output);

    // Dump OOP + IP strategy
    let strat_file = args.output.replace(".csv", "_strategy.txt");
    let mut strat = String::new();

    // Full EV: OOP
    strat.push_str("=== Full EV 根节点策略 (OOP) ===\n");
    strat.push_str(&dump_strategy(&game_full, ""));
    strat.push('\n');

    // Full EV: IP (after OOP check)
    game_full.back_to_root();
    if !game_full.is_terminal_node() && !game_full.is_chance_node() && !game_full.available_actions().is_empty() {
        game_full.play(0); // OOP CHECK
        strat.push_str("=== Full EV IP根节点策略 (OOP Check后) ===\n");
        strat.push_str(&dump_strategy(&game_full, ""));
        strat.push('\n');
    }

    // DL: OOP
    game_dl.back_to_root();
    strat.push_str("=== DLEv_corrected 根节点策略 (OOP) ===\n");
    strat.push_str(&dump_strategy(&game_dl, ""));
    strat.push('\n');

    // DL: IP
    game_dl.back_to_root();
    if !game_dl.is_terminal_node() && !game_dl.is_chance_node() && !game_dl.available_actions().is_empty() {
        game_dl.play(0);
        strat.push_str("=== DLEv_corrected IP根节点策略 (OOP Check后) ===\n");
        strat.push_str(&dump_strategy(&game_dl, ""));
        strat.push('\n');
    }

    std::fs::write(&strat_file, &strat).expect("Failed to write strategy");
    println!("策略已保存: {}", strat_file);
}
