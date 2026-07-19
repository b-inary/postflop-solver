# 翻牌圈深度限制解算（DL）优化方案

## 问题背景

完整 GTO 求解（Full Solve）遍历全部街的博弈树，单个牌面耗时 2-7 分钟，无法用于实时场景。因此采用深度限制解算（Depth-Limited CFR）：在翻牌圈截断博弈树，用 All-in Equity 近似终端节点 EV。

## 优化后的 DLEv 计算公式

### 终端节点估值（纯 equity 版本）

```
DLEv[h] = Σ_v cfreach[v] × equity_matrix[h][v] × factor
```

| 符号 | 含义 | 来源 |
|------|------|------|
| `cfreach[v]` | 对手手牌 v 的反事实到达概率 | CFR 迭代 |
| `equity_matrix[h][v]` | 手牌 h vs v 的 all-in 胜率差 [-1,+1] | 预计算 |
| `factor` | `(half_pot − 0.5×rake) / num_combinations` | 当前节点底池 |

**去除了以下系数（在 α 修正阶段统一处理）：**
- `pos_coef`（位置系数 δ=0.05）— 移除
- `cat_coef`（旧手牌分类系数）— 移除

### 求解后 α 修正

```
修正后 DLEv = DLEv × α(牌面纹理, 位置, 手牌桶)
```

α 系数为固定值查找表（8 牌面 × 2 位置 × 7 桶 = 112 个系数），通过 SPR=5 和 SPR=20 的数据回归得到。

## 计算流程

```
第一阶段：预计算（一次性的）
  ┌─────────────────────────────────────────┐
  │ compute_equity_matrix()                  │
  │   → 对每对 (OOP手牌, IP手牌),            │
  │     枚举所有 turn+river 组合计算胜率差    │
  └─────────────────────────────────────────┘

第二阶段：CFR 求解（2000 次迭代）
  ┌─────────────────────────────────────────┐
  │ solve()                                  │
  │   ├─ CFR 在翻牌圈动作树上迭代             │
  │   ├─ 到达 DL 终端节点时调 evaluate_depth_limited │
  │   │    └─ DLEv = cfv × factor（纯 equity）│
  │   └─ 2000 次迭代后收敛                    │
  └─────────────────────────────────────────┘

第三阶段：求解后修正
  ┌─────────────────────────────────────────┐
  │ expected_values_with_board_correction()  │
  │   ├─ detect_board_texture(flop)          │
  │   ├─ 对每手牌分类为 HandBucket            │
  │   └─ DLEv_corrected = DLEv × α           │
  └─────────────────────────────────────────┘
```

## 速度对比

| 模式 | 耗时（单个牌面） | 倍数 |
|------|---------------|------|
| Full Solve（完整解算） | 100~460 秒 | ×1 |
| DL Solve（深度限制） | **0.5~1.5 秒** | **×100~×300** |
| DL Solve + α 修正 | **0.5~1.5 秒**（修正计算 <1ms） | **×100~×300** |

## 关键文件

| 文件 | 功能 |
|------|------|
| `src/game/evaluation.rs` → `evaluate_depth_limited` | DL 终端节点估值 |
| `src/game/interpreter.rs` → `expected_values_with_board_correction` | 求解后 α 修正 |
| `src/hand.rs` → `BoardCorrectionContext::alpha()` | α 系数查找表 |
| `src/hand.rs` → `detect_board_texture()` | 8 种同构牌面自动分类 |
| `src/hand.rs` → `HandBucket::classify()` | 7 种手牌桶分类 |

## 验证结果

- 修正前平均 |DLEv 偏差|：~26%
- 修正后平均 |DLEv 偏差|：~10%
- SPR=5 和 SPR=20 均有效
