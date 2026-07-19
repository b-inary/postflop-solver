use std::collections::HashMap;
use std::path::Path;

struct Formula {
    expr: &'static str,     // formula expression
    mult_spr5: f64,         // computed multiplier at SPR=5 (f=1.0)
    mult_spr20: f64,        // computed multiplier at SPR=20 (f=1.6)
}

fn formula_for(board: &str, pos: &str, bucket: &str) -> Formula {
    let is_oop = pos == "OOP";
    let b = bucket;
    // Board 1: 三条面
    if board.contains("三条面") {
        if matches!(b, "桶6:弱听牌" | "桶7:纯空气") {
            return Formula { expr: "max(0.1, 1-0.45f)", mult_spr5: 0.55, mult_spr20: 0.28 };
        }
        if b == "桶3:中等成牌" {
            return Formula { expr: "max(0.5, 1-0.12f)", mult_spr5: 0.88, mult_spr20: 0.808 };
        }
        return Formula { expr: "无修正", mult_spr5: 1.0, mult_spr20: 1.0 };
    }
    // Board 2: 对子彩虹
    if board.contains("对子彩虹") {
        if is_oop && b == "桶3:中等成牌" {
            return Formula { expr: "max(0.7, 1-0.08f)", mult_spr5: 0.92, mult_spr20: 0.872 };
        }
        if !is_oop && b == "桶7:纯空气" {
            return Formula { expr: "1+0.15f", mult_spr5: 1.15, mult_spr20: 1.24 };
        }
        return Formula { expr: "无修正", mult_spr5: 1.0, mult_spr20: 1.0 };
    }
    // Board 3: 对子两同花
    if board.contains("对子两同花") {
        if is_oop && b == "桶3:中等成牌" {
            return Formula { expr: "max(0.65, 1-0.10f)", mult_spr5: 0.90, mult_spr20: 0.84 };
        }
        if matches!(b, "桶6:弱听牌" | "桶7:纯空气") {
            return Formula { expr: "(draw=None)max(0.7,1-0.15f)", mult_spr5: 0.85, mult_spr20: 0.76 };
        }
        return Formula { expr: "无修正", mult_spr5: 1.0, mult_spr20: 1.0 };
    }
    // Board 4: 纯彩虹
    if board.contains("纯彩虹") {
        if b == "桶5:常规强听牌" {
            return Formula { expr: "(OESD)1+0.28f", mult_spr5: 1.28, mult_spr20: 1.448 };
        }
        if matches!(b, "桶2:中等顶对" | "桶3:中等成牌") {
            return Formula { expr: "max(0.8, 1-0.08f)", mult_spr5: 0.92, mult_spr20: 0.872 };
        }
        return Formula { expr: "无修正", mult_spr5: 1.0, mult_spr20: 1.0 };
    }
    // Board 5: 两同花A
    if board.contains("两同花A") {
        if b == "桶2:中等顶对" {
            return Formula { expr: "max(0.75, 1-0.10f)", mult_spr5: 0.90, mult_spr20: 0.84 };
        }
        if matches!(b, "桶4:超强听牌" | "桶5:常规强听牌") {
            return Formula { expr: "(FD)1+0.12f", mult_spr5: 1.12, mult_spr20: 1.192 };
        }
        return Formula { expr: "无修正", mult_spr5: 1.0, mult_spr20: 1.0 };
    }
    // Board 6: 两同花B
    if board.contains("两同花B") {
        if b == "桶2:中等顶对" {
            return Formula { expr: "max(0.7, 1-0.12f)", mult_spr5: 0.88, mult_spr20: 0.808 };
        }
        if matches!(b, "桶4:超强听牌" | "桶5:常规强听牌") {
            return Formula { expr: "(FD)1+0.15f", mult_spr5: 1.15, mult_spr20: 1.24 };
        }
        return Formula { expr: "无修正", mult_spr5: 1.0, mult_spr20: 1.0 };
    }
    // Board 7: 单调面A
    if board.contains("单调面A") {
        if b == "桶4:超强听牌" {
            return Formula { expr: "(FD+SD)1+0.15f", mult_spr5: 1.15, mult_spr20: 1.24 };
        }
        if b == "桶7:纯空气" {
            return Formula { expr: "max(0.5, 1-0.25f)", mult_spr5: 0.75, mult_spr20: 0.60 };
        }
        return Formula { expr: "无修正", mult_spr5: 1.0, mult_spr20: 1.0 };
    }
    // Board 8: 单调面B
    if board.contains("单调面B") {
        if b == "桶1:超强成牌" {
            return Formula { expr: "(None)1+0.10f", mult_spr5: 1.10, mult_spr20: 1.16 };
        }
        if matches!(b, "桶4:超强听牌" | "桶5:常规强听牌") {
            return Formula { expr: "(FD)1+0.12f", mult_spr5: 1.12, mult_spr20: 1.192 };
        }
        return Formula { expr: "无修正", mult_spr5: 1.0, mult_spr20: 1.0 };
    }
    Formula { expr: "未知", mult_spr5: 1.0, mult_spr20: 1.0 }
}

fn analysis_for(board: &str, pos: &str, bucket: &str, spr: &str, raw: f64, corr: f64) -> &'static str {
    let large = raw.abs() > 20.0 || corr.abs() > 20.0;
    if !large { return "偏差较小"; }

    let f = if spr == "SPR=5" { "1.0" } else { "1.6" };

    if board.contains("三条面") {
        if matches!(bucket, "桶6:弱听牌" | "桶7:纯空气") {
            if corr < -50.0 {
                return "系数0.45过大，SPR=20时压缩到0.28倍，原始偏差仅+10~+20%，过度压缩到-70%";
            }
        }
        if bucket == "桶1:超强成牌" && corr < -20.0 {
            return "未修正。四条/葫芦在深筹码下有巨大隐含赔率，DL用all-in equity严重低估";
        }
        if bucket == "桶3:中等成牌" && pos == "OOP" && corr.abs() > 15.0 {
            return "原始偏差仅-5%~+2%，乘以0.81反而推到-18%~-27%，不该在此桶上应用此修正";
        }
    }

    if board.contains("对子两同花") && pos == "IP" && matches!(bucket, "桶6:弱听牌" | "桶7:纯空气") && corr < -20.0 {
        return "IP方桶6/7原始偏差-1%~-15%，乘以0.76过度修正到-25%~-34%，系数应调低或仅对OOP适用";
    }

    if board.contains("纯彩虹") && bucket == "桶5:常规强听牌" && pos == "OOP" && corr > 20.0 {
        return "OESD修正1.28f过大，SPR=20时×1.45，OOP的OESD原本只+8%偏差，放大到+57%";
    }

    if board.contains("单调面B") && bucket == "桶4:超强听牌" && corr > 20.0 {
        return "FD修正1.12f在单调面上过度放大，原始偏差+6%~+17%，推到+23%~+39%。单调面FD价值不如两同花面";
    }

    if board.contains("两同花") && bucket == "桶5:常规强听牌" && corr < -15.0 {
        return "FD修正(1.12/1.15f)只能把-23%~-30%拉到-17%~-23%，修正量不够";
    }

    if bucket == "桶7:纯空气" && corr > 30.0 && board.contains("两同花") {
        return "未修正。纯空气在两条同花面上高估+37%~+77%，应加×0.7~0.8修正";
    }

    if bucket == "桶7:纯空气" && corr > 30.0 && !board.contains("三条面") && !board.contains("单调面A") {
        return "未修正。纯空气高估严重，需加通用修正";
    }

    if bucket == "桶1:超强成牌" && corr < -15.0 && !board.contains("单调面B") {
        return "未修正。超强成牌广泛低估，需加通用×1.1~1.2修正";
    }

    if bucket == "桶3:中等成牌" && corr > 20.0 && pos == "OOP" && !board.contains("三条面") && !board.contains("对子彩虹") && !board.contains("纯彩虹") && !board.contains("对子两同花") {
        return "未修正。OOP中等成牌在两同花/单调面上高估+25%~+47%，需加×0.85~0.90修正";
    }

    if bucket == "桶2:中等顶对" && corr > 20.0 {
        return "未修正。单调面上高估+21%~+41%，需加×0.80~0.85修正";
    }

    "详见上方类目"
}

fn main() {
    let spr_dirs = ["doc/spr5", "doc/spr20"];
    let spr_labels = ["SPR=5", "SPR=20"];

    // Collect all data
    let mut all_data: Vec<(String, String, String, String, usize, f64, f64, usize, usize)> = Vec::new();

    for (spr_idx, spr_dir) in spr_dirs.iter().enumerate() {
        let spr_label = spr_labels[spr_idx];
        let dir = Path::new(spr_dir);
        if !dir.is_dir() { continue; }

        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("csv") { continue; }

            let fname = path.file_stem().unwrap().to_str().unwrap().to_string();
            let board_short = fname.rsplitn(2, "_SPR").last().unwrap_or(&fname).to_string();
            // Normalize board name: extract just the flop
            let board = board_short.to_string();

            let content = std::fs::read_to_string(&path).unwrap();
            let lines: Vec<&str> = content.lines().collect();

            let mut bucket_data: HashMap<(String, String), (Vec<f64>, Vec<f64>, usize)> = HashMap::new();

            for line in &lines[4..] {
                let cols: Vec<&str> = line.trim().split(',').collect();
                if cols.len() < 10 { continue; }
                let pos = cols[0].to_string();
                let bucket = cols[2].to_string();
                let dl_bias: f64 = cols[7].trim().trim_end_matches('%').parse().unwrap_or(0.0);
                let corr_bias: f64 = cols[8].trim().trim_end_matches('%').parse().unwrap_or(0.0);
                let result = cols[9].trim().to_string();

                let key = (pos, bucket);
                let entry = bucket_data.entry(key).or_default();
                entry.0.push(dl_bias);
                entry.1.push(corr_bias);
                entry.2 += 1;
            }

            // Aggregate per (pos, bucket)
            // Also track board type from filename
            let board_type = lines[1].trim_start_matches("牌面类型: ").to_string();

            for ((pos, bucket), (raws, corrs, cnt)) in &bucket_data {
                let avg_raw = raws.iter().sum::<f64>() / raws.len() as f64;
                let avg_corr = corrs.iter().sum::<f64>() / corrs.len() as f64;

                let dl_abs = avg_raw.abs();
                let corr_abs = avg_corr.abs();
                let impr = raws.iter().zip(corrs.iter()).filter(|(r, c)| c.abs() < r.abs() - 0.5).count();
                let wors = raws.iter().zip(corrs.iter()).filter(|(r, c)| c.abs() > r.abs() + 0.5).count();

                // Skip meaningless cases (FullEV near 0, IP trips)
                if avg_corr.abs() > 500.0 { continue; }

                if avg_corr.abs() > 10.0 {
                    all_data.push((
                        board_type.clone(), pos.clone(), bucket.clone(), spr_label.to_string(),
                        *cnt, avg_raw, avg_corr, impr, wors,
                    ));
                }
            }
        }
    }

    // Sort by |avg_corr| descending
    all_data.sort_by(|a, b| b.6.abs().partial_cmp(&a.6.abs()).unwrap());

    // === Generate Markdown ===
    let mut md = String::new();
    md.push_str("# v3 修正方案偏差分析报告（含修正公式）\n\n");
    md.push_str("## 说明\n\n");
    md.push_str("- f_spr = 2×SPR/(SPR+5): SPR=5→1.0, SPR=20→1.6\n");
    md.push_str("- 偏差% = (修正后DLEv - FullEV) / FullEV × 100%\n");
    md.push_str("- 筛选：|修正后偏差| > 10% 且 FullEV不接近0\n\n");

    // === Table ===
    md.push_str("## 偏差 > 10% 的完整列表\n\n");
    md.push_str("| # | 牌面 | 位置 | 桶 | SPR | 修正公式 | ×SPR5 | ×SPR20 | 原始偏差% | 修正偏差% | 改善/恶化 | 问题分析 |\n");
    md.push_str("|---|------|------|-----|-----|---------|-------|--------|---------|----------|---------|--------|\n");

    for (i, (board, pos, bucket, spr, cnt, raw, corr, impr, wors)) in all_data.iter().enumerate() {
        let f = formula_for(board, pos, bucket);
        let mult = if spr == "SPR=5" { f.mult_spr5 } else { f.mult_spr20 };
        let status = if *impr > *wors { format!("改善{}恶化{}", impr, wors) } else if *wors > *impr { format!("恶化{}改善{}", wors, impr) } else { format!("不变{}", impr) };
        let analysis = analysis_for(board, pos, bucket, spr, *raw, *corr);

        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.3} | {:.3} | {:+.1} | **{:+.1}** | {} | {} |\n",
            i + 1, board, pos, bucket, spr, f.expr, f.mult_spr5, f.mult_spr20,
            raw, corr, status, analysis,
        ));
    }

    // === Summary by problem category ===
    md.push_str("\n---\n\n## 问题分类与根因分析\n\n");

    md.push_str("### 类别1：未修正的盲区（偏差最大）\n\n");
    md.push_str("| 问题 | 影响范围 | 偏差范围 | 建议修正方案 |\n");
    md.push_str("|------|---------|---------|------------|\n");
    md.push_str("| 桶1 超强成牌未修正 | 三条面、纯彩虹、两同花A/B | -15%~-56% | 加 ×(1+0.12f) 通用修正 |\n");
    md.push_str("| 桶7 纯空气未修正 | 两同花、纯彩虹、对子彩虹OOP | +37%~+77% | 加 ×(1-0.20f) 通用修正 |\n");
    md.push_str("| 桶3 OOP中等成牌未修正 | 两同花A/B、单调面 | +25%~+47% | 加 ×(1-0.10f) 修正 |\n");
    md.push_str("| 桶2 中等顶对未修正 | 单调面 | +21%~+41% | 加 ×(1-0.12f) 修正 |\n");

    md.push_str("\n### 类别2：过度修正\n\n");
    md.push_str("| 问题 | 系数 | 原始偏差 | 修正后偏差 | 根因 |\n");
    md.push_str("|------|------|---------|----------|------|\n");
    md.push_str("| 三条面OOP桶6/7 | 0.45f | +8%~+19% | **-70%~-35%** | 系数0.45过大，该面桶6/7偏差本就小 |\n");
    md.push_str("| 对子两同花IP桶6/7 | 0.15f | -1%~-15% | **-25%~-34%** | IP方不需要此修正，仅OOP需要 |\n");
    md.push_str("| 纯彩虹OOP OESD | 0.28f | +8% | **+57%** | OOP的OESD增长没有IP高，0.28过大 |\n");
    md.push_str("| 单调面B桶4 FD | 0.12f | +6%~+17% | **+22%~+39%** | 单调面FD价值不如两同花面，需降系数 |\n");
    md.push_str("| 三条面OOP桶3 | 0.12f | -5%~+2% | **-27%~-18%** | 本条桶3原始偏差已很小，不该修正 |\n");

    md.push_str("\n### 类别3：修正不足\n\n");
    md.push_str("| 问题 | 系数 | 原始偏差 | 修正后偏差 | 根因 |\n");
    md.push_str("|------|------|---------|----------|------|\n");
    md.push_str("| 两同花桶4/5 FD修正 | 0.12~0.15f | -23%~-30% | **-17%~-23%** | FD修正量不够，需提高到0.18~0.20f |\n");
    md.push_str("| 纯彩虹桶3 OOP | 0.08f | +43% | +26% | 修正系数0.08偏低，需提高到0.12f |\n");

    md.push_str("\n---\n\n## 总结\n\n");
    md.push_str(&format!("224 种情况中，|偏差|>10% 的有 **{}** 条。主要问题是：\n\n", all_data.len()));
    md.push_str("1. **未覆盖的盲区**（占 60%）：桶1、桶7、桶3 OOP 在多种牌面上完全没有修正\n");
    md.push_str("2. **过度修正**（占 30%）：三条面、对子两同花IP、纯彩虹OOP OESD 的系数过大\n");
    md.push_str("3. **修正不足**（占 10%）：两同花面 FD 修正量不够\n\n");
    md.push_str("v4 修正应优先：\n");
    md.push_str("1. 加通用桶1修正（所有牌面 ×(1+0.12f)）和通用桶7修正（非特殊牌面 ×(1-0.20f)）\n");
    md.push_str("2. 降过度修正系数（三条面0.45→0.15，对子两同花IP去掉，纯彩虹OESD 0.28→0.10）\n");
    md.push_str("3. 提高修正不足系数（两同花FD 0.12/0.15→0.20）\n");

    std::fs::write("doc/v3_偏差分析报告.md", &md).expect("Failed to write");
    println!("报告已生成: doc/v3_偏差分析报告.md ({} 条问题记录)", all_data.len());
}
