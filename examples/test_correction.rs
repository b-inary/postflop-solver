use std::collections::HashMap;
use std::path::Path;

// α(牌面, 桶, 位置) — 从数据中提取的基础修正因子
fn alpha(board_flop: &str, pos: &str, bucket: &str) -> f64 {
    let is_oop = pos == "OOP";
    match board_flop {
        "8h8s8d" => match (pos, bucket) {
            ("OOP", "桶1:超强成牌") => 1.50,
            ("IP", "桶1:超强成牌") => 1.30,
            ("OOP", "桶3:中等成牌") => 1.18,
            ("IP", "桶3:中等成牌") => 0.91,
            ("OOP", "桶6:弱听牌") => 0.90,
            ("OOP", "桶7:纯空气") => 0.82,
            _ => 1.0,
        },
        "KhKd5c" => match (pos, bucket) {
            ("OOP", "桶1:超强成牌") => 1.23,
            ("IP", "桶1:超强成牌") => 1.10,
            ("OOP", "桶3:中等成牌") => 0.81,
            ("IP", "桶3:中等成牌") => 0.87,
            ("OOP", "桶6:弱听牌") => 0.84,
            ("IP", "桶6:弱听牌") => 1.49,
            ("OOP", "桶7:纯空气") => 0.65,
            ("IP", "桶7:纯空气") => 0.97,
            _ => 1.0,
        },
        "KhKc5c" => match (pos, bucket) {
            ("OOP", "桶1:超强成牌") => 1.16,
            ("IP", "桶1:超强成牌") => 1.07,
            ("OOP", "桶3:中等成牌") => 0.82,
            ("IP", "桶3:中等成牌") => 0.96,
            ("OOP", "桶5:常规强听牌") => 1.01,
            ("IP", "桶5:常规强听牌") => 1.18,
            ("OOP", "桶6:弱听牌") => 0.77,
            ("OOP", "桶7:纯空气") => 0.72,
            ("IP", "桶6:弱听牌") => 1.13,
            ("IP", "桶7:纯空气") => 1.03,
            _ => 1.0,
        },
        "QsJh4d" => match (pos, bucket) {
            ("OOP", "桶1:超强成牌") => 1.28,
            ("IP", "桶1:超强成牌") => 1.36,
            ("OOP", "桶2:中等顶对") => 1.02,
            ("IP", "桶2:中等顶对") => 1.01,
            ("OOP", "桶3:中等成牌") => 0.75,
            ("IP", "桶3:中等成牌") => 0.79,
            ("OOP", "桶5:常规强听牌") => 0.98,
            ("IP", "桶5:常规强听牌") => 1.42,
            ("OOP", "桶6:弱听牌") => 0.96,
            ("OOP", "桶7:纯空气") => 0.65,
            ("IP", "桶7:纯空气") => 0.66,
            _ => 1.0,
        },
        "QsJs4h" => match (pos, bucket) {
            ("OOP", "桶1:超强成牌") => 1.18,
            ("IP", "桶1:超强成牌") => 1.22,
            ("OOP", "桶2:中等顶对") => 0.93,
            ("IP", "桶2:中等顶对") => 0.95,
            ("OOP", "桶3:中等成牌") => 0.74,
            ("IP", "桶3:中等成牌") => 0.84,
            ("OOP", "桶4:超强听牌") => 1.22,
            ("IP", "桶4:超强听牌") => 1.36,
            ("OOP", "桶5:常规强听牌") => 0.98,
            ("IP", "桶5:常规强听牌") => 1.39,
            ("OOP", "桶6:弱听牌") => 0.90,
            ("OOP", "桶7:纯空气") => 0.62,
            ("IP", "桶7:纯空气") => 0.73,
            _ => 1.0,
        },
        "QhJs4s" => match (pos, bucket) {
            ("OOP", "桶1:超强成牌") => 1.21,
            ("IP", "桶1:超强成牌") => 1.26,
            ("OOP", "桶2:中等顶对") => 0.94,
            ("IP", "桶2:中等顶对") => 0.96,
            ("OOP", "桶3:中等成牌") => 0.74,
            ("IP", "桶3:中等成牌") => 0.83,
            ("OOP", "桶4:超强听牌") => 1.22,
            ("IP", "桶4:超强听牌") => 1.18,
            ("OOP", "桶5:常规强听牌") => 0.97,
            ("IP", "桶5:常规强听牌") => 1.37,
            ("OOP", "桶6:弱听牌") => 0.90,
            ("OOP", "桶7:纯空气") => 0.60,
            ("IP", "桶7:纯空气") => 0.72,
            _ => 1.0,
        },
        "AsJs9s" => match (pos, bucket) {
            ("OOP", "桶1:超强成牌") => 1.17,
            ("IP", "桶1:超强成牌") => 1.14,
            ("OOP", "桶2:中等顶对") => 0.77,
            ("IP", "桶2:中等顶对") => 0.78,
            ("OOP", "桶3:中等成牌") => 0.71,
            ("IP", "桶3:中等成牌") => 0.80,
            ("OOP", "桶4:超强听牌") => 1.10,
            ("IP", "桶4:超强听牌") => 1.14,
            ("OOP", "桶5:常规强听牌") => 0.85,
            ("OOP", "桶7:纯空气") => 0.62,
            ("IP", "桶7:纯空气") => 0.64,
            _ => 1.0,
        },
        "8s6s3s" => match (pos, bucket) {
            ("OOP", "桶1:超强成牌") => 1.17,
            ("IP", "桶1:超强成牌") => 1.12,
            ("OOP", "桶2:中等顶对") => 0.73,
            ("IP", "桶2:中等顶对") => 0.77,
            ("OOP", "桶3:中等成牌") => 0.76,
            ("IP", "桶3:中等成牌") => 0.81,
            ("OOP", "桶4:超强听牌") => 0.96,
            ("IP", "桶4:超强听牌") => 0.84,
            ("OOP", "桶5:常规强听牌") => 1.08,
            ("IP", "桶5:常规强听牌") => 1.18,
            ("OOP", "桶7:纯空气") => 0.68,
            ("IP", "桶7:纯空气") => 0.95,
            _ => 1.0,
        },
        _ => 1.0,
    }
}

fn main() {
    let spr_dirs = ["doc/spr5", "doc/spr20"];
    let spr_labels = ["SPR=5", "SPR=20"];

    let mut all_results: Vec<(String, String, String, String, usize, f64, f64, f64)> = Vec::new();
    let mut overall_raw_mae = 0.0f64;
    let mut overall_corr_mae = 0.0f64;
    let mut overall_count = 0usize;

    for (spr_idx, spr_dir) in spr_dirs.iter().enumerate() {
        let spr_label = spr_labels[spr_idx];
        let dir = Path::new(spr_dir);
        if !dir.is_dir() { continue; }

        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("csv") { continue; }

            let fname = path.file_stem().unwrap().to_str().unwrap().to_string();
            // Extract flop code from filename
            let flop = fname.split('_').next().unwrap_or(&fname).to_string();

            let content = std::fs::read_to_string(&path).unwrap();
            let lines: Vec<&str> = content.lines().collect();

            // Aggregate per (pos, bucket)
            let mut bucket_data: HashMap<(String, String), (Vec<f64>, Vec<f64>)> = HashMap::new();

            for line in &lines[4..] {
                let cols: Vec<&str> = line.trim().split(',').collect();
                if cols.len() < 10 { continue; }
                let pos = cols[0].to_string();
                let bucket = cols[2].to_string();
                let full: f64 = cols[4].parse().unwrap_or(0.0);
                let dl: f64 = cols[5].parse().unwrap_or(0.0);

                if full.abs() < 0.5 { continue; } // Skip near-zero FullEV

                let key = (pos.clone(), bucket.clone());
                let entry = bucket_data.entry(key).or_insert_with(|| (Vec::new(), Vec::new()));
                entry.0.push(full);
                entry.1.push(dl);
            }

            for ((pos, bucket), (fulls, dls)) in &bucket_data {
                if fulls.is_empty() { continue; }

                let n = fulls.len();
                let avg_full = fulls.iter().sum::<f64>() / n as f64;
                let avg_dl = dls.iter().sum::<f64>() / n as f64;

                let a = alpha(&flop, pos, bucket);
                let corrected = avg_dl * a;

                let raw_bias = (avg_dl - avg_full) / avg_full * 100.0;
                let corr_bias = (corrected - avg_full) / avg_full * 100.0;

                overall_raw_mae += raw_bias.abs();
                overall_corr_mae += corr_bias.abs();
                overall_count += 1;

                all_results.push((
                    flop.clone(), pos.clone(), bucket.clone(), spr_label.to_string(),
                    n, raw_bias, corr_bias, a,
                ));
            }
        }
    }

    // Sort by |corr_bias| descending
    all_results.sort_by(|a, b| b.6.abs().partial_cmp(&a.6.abs()).unwrap());

    // Generate report
    let mut md = String::new();
    md.push_str("# 固定 α 修正测试报告\n\n");
    md.push_str("**修正方法：**`修正后 DLEv = DLEv × α(牌面,桶,位置)`，α 从数据中取 SPR=5 时的修正因子。\n\n");

    md.push_str("## 总体结果\n\n");
    md.push_str(&format!(
        "| 指标 | 修正前 | 修正后 |\n|------|--------|--------|\n"
    ));
    let avg_raw = overall_raw_mae / overall_count as f64;
    let avg_corr = overall_corr_mae / overall_count as f64;
    md.push_str(&format!(
        "| 平均 |偏差| | {:.1}% | **{:.1}%** |\n", avg_raw, avg_corr
    ));
    md.push_str(&format!(
        "| 改善 | {:+.1}% | |\n", (avg_corr - avg_raw)
    ));

    // Count improved vs worsened
    let improved = all_results.iter().filter(|r| r.6.abs() < r.5.abs() - 0.5).count();
    let worsened = all_results.iter().filter(|r| r.6.abs() > r.5.abs() + 0.5).count();
    md.push_str(&format!("| 改善组合数 | {} / {} |\n", improved, all_results.len()));
    md.push_str(&format!("| 恶化组合数 | {} / {} |\n", worsened, all_results.len()));

    // Table of results
    md.push_str("\n## 各组合结果（按修正后 |偏差| 从大到小排序）\n\n");
    md.push_str("| # | 牌面 | 位置 | 桶 | SPR | 手数 | α | 原始偏差% | 修正偏差% | 效果 |\n");
    md.push_str("|---|------|------|-----|-----|------|----|---------|----------|------|\n");

    for (i, (flop, pos, bucket, spr, n, raw, corr, a)) in all_results.iter().enumerate() {
        let status = if corr.abs() < 10.0 { "✅" }
            else if corr.abs() < 20.0 { "⚠️" }
            else { "❌" };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.3} | {:+.1} | **{:+.1}** | {} |\n",
            i + 1, flop, pos, bucket, spr, n, a, raw, corr, status,
        ));
    }

    // Summary by SPR
    for spr_label in &spr_labels {
        let spr_results: Vec<_> = all_results.iter().filter(|r| r.3 == *spr_label).collect();
        let n = spr_results.len();
        let mae_raw: f64 = spr_results.iter().map(|r| r.5.abs()).sum::<f64>() / n as f64;
        let mae_corr: f64 = spr_results.iter().map(|r| r.6.abs()).sum::<f64>() / n as f64;
        md.push_str(&format!("\n### {}: MAE {:.1}% → {:.1}%\n", spr_label, mae_raw, mae_corr));
    }

    // Summary by bucket
    md.push_str("\n## 按桶汇总\n\n");
    let bucket_order = [
        "桶1:超强成牌", "桶2:中等顶对", "桶3:中等成牌",
        "桶4:超强听牌", "桶5:常规强听牌", "桶6:弱听牌", "桶7:纯空气",
    ];
    md.push_str("| 桶 | 组合数 | 平均原始偏差 | 平均修正偏差 | 改善 |\n");
    md.push_str("|---|------|-----------|-----------|------|\n");
    for bucket in &bucket_order {
        let bucket_results: Vec<_> = all_results.iter().filter(|r| r.2 == *bucket).collect();
        if bucket_results.is_empty() { continue; }
        let n = bucket_results.len();
        let avg_raw: f64 = bucket_results.iter().map(|r| r.5.abs()).sum::<f64>() / n as f64;
        let avg_corr: f64 = bucket_results.iter().map(|r| r.6.abs()).sum::<f64>() / n as f64;
        md.push_str(&format!(
            "| {} | {} | {:.1}% | **{:.1}%** | {:+.1}% |\n",
            bucket, n, avg_raw, avg_corr, avg_corr - avg_raw,
        ));
    }

    std::fs::write("doc/v4_固定α测试报告.md", &md).expect("Failed to write");
    println!("报告已生成: doc/v4_固定α测试报告.md");
}
