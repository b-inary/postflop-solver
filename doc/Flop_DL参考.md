# Flop DL 参考文档

## 分支与标签

| 项目 | 值 |
|------|---|
| **分支** | `flop-dl-alpha` |
| **标签** | `v1.0-flop-dl` |
| **最后提交** | `3001356` (2026-07-19) |
| **仓库** | `https://github.com/LY1Master/postflop-solver` |

## Flop DL 完成的工作

### 核心算法

1. **DL 树截断**：`depth_limit = Some(BoardState::Flop)`，翻牌动作后截断
2. **终端估值**：`evaluate_depth_limited`，公式 `DLEv = Σ_v cfreach[v] × equity(h,v) × factor`
3. **equity 矩阵预计算**：1081 种 turn+river 组合的 all-in 胜率

### 牌面与手牌分类

4. **8 种同构牌面自动检测**：`detect_board_texture(flop)` → `BoardTexture` 枚举
5. **7 种手牌桶自动分类**：`HandBucket::classify()`，支持后门花检测

### EV 修正

6. **α post-solve 修正**：`修正后DLev = DLEv × α(牌面,位置,桶)`，112 个系数查找表
7. **IP 攻击因子**：`ip_aggression_factor()`，给 IP 低 EV 桶加 bluff 溢价
8. **FE bonus (实验)**：在 `evaluate_depth_limited` 中加 fold equity 项

### 跨街求解

9. **转牌圈全树展开**：`depth_limit = None` 到摊牌
10. **跨街范围传递**：`set_custom_initial_weights()`

### 工具

11. **`bench_single`**：单牌面 Full vs DL EV + 策略 dump
12. **`bench_multi_boards`**：8牌面批量测试
13. **`rerun_dl` / `apply_alpha` / `test_correction`**：数据分析工具

### 验证结果

14. **EV 精度**: MAE ~10-14%（8牌面×2SPR验证）
15. **策略一致性**: OOP 99.4%、IP bet 频率差距 5pp（vs 商业求解器）
16. **速度**: DL ~1s vs Full 100-460s（×300加速）

### 关键文件

| 文件 | 内容 |
|------|------|
| `src/hand.rs` | `BoardTexture`、`HandBucket`、`BoardCorrectionContext` |
| `src/game/evaluation.rs` | `evaluate_depth_limited`、`compute_equity_matrix`、FE bonus |
| `src/game/interpreter.rs` | `expected_values_with_board_correction`、跨街权重 |
| `src/game/base.rs` | DL 树构建、`set_custom_initial_weights`、`air_ratio` |
| `src/action_tree.rs` | `TreeConfig.depth_limit`、`CategoryCoefficients` |

### 测试数据

| 目录 | 内容 |
|------|------|
| `doc/spr5/` `doc/spr20/` | 8牌面×2SPR Full EV vs DLEv原始数据 |
| `doc/flop_pure/` | 纯equity DLEv（无pos_coef/cat_coef） |
| `doc/flop_corrected/` | α修正后 DLEv |
| `doc/flop_fe_v3/` | FE bonus 策略测试数据 |

### 文档

| 文档 | 内容 |
|------|------|
| `doc/项目交接文档.md` | 完整项目交接文档 |
| `doc/翻牌圈DL优化方案.md` | 翻牌圈优化方案说明 |
| `doc/修正方案1/2/3.md` | α修正方案的三个迭代版本 |
| `doc/v3_修正公式与偏差分析.md` | v3修正公式详细分析 |
| `doc/v4_固定α测试报告.md` | 固定α方案验证 |
| `doc/多牌面泛化验证报告.md` | 8牌面泛化验证 |
| `doc/flop_fe_v3_report.md` | FE v3 策略验证报告 |
