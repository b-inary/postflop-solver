/// 测试转牌圈全树展开在不同配置下的耗时
use postflop_solver::*;
use std::time::Instant;

fn test(label: &str, oop_range: &str, ip_range: &str, bet_sizes_str: &str, pot: i32, stack: i32) {
    let flop = flop_from_str("QsJh4d").unwrap();
    let turn = card_from_str("9c").unwrap();
    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop,
        turn,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from((bet_sizes_str, "2.5x")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: pot * 10,
        effective_stack: stack * 10,
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

    println!("--- {} ---", label);
    println!("  范围: OOP={} 手牌, IP={} 手牌", oop_range.len(), ip_range.len());
    println!("  下注: {}", bet_sizes_str);

    let t = Instant::now();
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    println!("  建树: {:.1}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game.allocate_memory(false);
    let (mem_usage, _) = game.memory_usage();
    println!("  内存: {:.1}s ({}MB)", t.elapsed().as_secs_f64(), mem_usage as f64 / 1024.0 / 1024.0);

    let target = (pot * 10) as f32 * 0.005;
    let t = Instant::now();
    let exp = solve(&mut game, 2000, target, false);
    println!("  求解: {:.1}s (exploit={:.4})", t.elapsed().as_secs_f64(), exp);
    println!();
}

fn main() {
    let wide_range = "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s+,A2o+,K2o+,Q2o+,J2o+,T2o+,92o+,82o+,72o+,62o+,52o+,42o+,32o+";
    let default_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
    let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";
    // 庄位BTN ~50% 范围
    let btn_50 = "22+,A2s+,K2s+,Q2s+,J2s+,T6s+,95s+,85s+,74s+,63s+,52s+,42s+,A2o+,K2o+,Q7o+,J7o+,T7o+,97o+,86o+,75o+,64o+,53o+";
    // 大盲BB ~70% 范围（比 BTN 宽）
    let bb_70 = "22+,A2s+,K2s+,Q2s+,J2s+,T2s+,92s+,82s+,72s+,62s+,52s+,42s+,32s+,A2o+,K2o+,Q2o+,J2o+,T2o+,92o+,82o+,72o+,62o+,52o+,42o+";

    println!("=== 转牌圈全树展开耗时测试 ===");
    println!("牌面: QsJh4d + 9c\n");

    // 测试1: 默认范围 + 默认尺寸
    test("测试1: 默认范围 + 60%,e,a", default_range, ip_range, "60%, e, a", 5, 100);

    // 测试2: 宽范围 + 默认尺寸
    test("测试2: 宽范围 + 60%,e,a", wide_range, wide_range, "60%, e, a", 5, 100);

    // 测试3: 默认范围 + 多尺寸
    test("测试3: 默认范围 + 30%,50%,75%,125%,a", default_range, ip_range, "30%, 50%, 75%, 125%, a", 5, 100);

    // 测试4: 宽范围 + 多尺寸
    test("测试4: 宽范围 + 30%,50%,75%,125%,a", wide_range, wide_range, "30%, 50%, 75%, 125%, a", 5, 100);

    // 测试5: 宽范围+多尺寸+短码
    test("测试5: 宽范围+多尺寸+短码 pot=5,stack=10", wide_range, wide_range, "30%, 50%, 75%, 125%, a", 5, 10);

    // 测试6: BTN~50% vs BB~70% + 多尺寸
    test("测试6: BTN50% vs BB70% + 多尺寸", bb_70, btn_50, "30%, 50%, 75%, 125%, a", 5, 100);

    // 测试5: 宽范围 + 多尺寸 + 短码(SPR=2)
    test("测试5: 宽范围+多尺寸+短码 pot=5,stack=10", wide_range, wide_range, "30%, 50%, 75%, 125%, a", 5, 10);

    println!("=== 全部完成 ===");
}
