# Turn DL 测试要求

## 一、每次测试必须记录的数据

### 1.1 每手牌级别数据（.csv格式）

每个牌面 × SPR 组合输出一个 CSV 文件，文件名为 `{牌面}_{牌面类型}_SPR{值}.csv`。

| 列名 | 说明 |
|------|------|
| 位置 | OOP / IP |
| 手牌 | AsAc 等标准格式 |
| 大桶 | 7种手牌桶名称 |
| Draw | 听牌类型 |
| FullEV | 全树精确解算EV |
| DLEv | 深度限制（未修正）EV |
| DLEv_corrected | 修正后 EV（如有修正方案） |
| DLEv偏差% | (DLEv - FullEV) / FullEV × 100% |
| 修正后偏差% | (DLEv_corrected - FullEV) / FullEV × 100% |
| 修正结果 | 改善 / 恶化 / 没改变 |

### 1.2 策略数据

每牌面 × SPR 输出策略文件 `{牌面}_{牌面类型}_SPR{值}_strategy.txt`。

包含:
- OOP 根节点每手牌各动作概率
- IP 根节点（OOP check后）每手牌各动作概率
- 转牌层各节点策略（如有展开）

### 1.3 汇总数据

每轮测试后生成汇总报告 (.md格式)，包含:
- 各牌面 OOP/IP 的 MAE（修正前/修正后）
- 各牌面 OOP/IP 的平均 Bet%
- 各桶级别的 MAE 和策略统计
- 与 Full solve 的偏差分析

## 二、测试配置

### 2.1 牌面

| # | 翻牌 | 类型 | 转牌（Turn DL测试用） |
|---|------|------|-------------------|
| 1 | 8h8s8d | 三条面 | Ac |
| 2 | KhKd5c | 对子彩虹 | 7h |
| 3 | KhKc5c | 对子两同花 | 7h |
| 4 | QsJh4d | 纯彩虹 | 9c |
| 5 | QsJs4h | 两同花A | Ad |
| 6 | QhJs4s | 两同花B | Ad |
| 7 | AsJs9s | 单调面A | Kh |
| 8 | 8s6s3s | 单调面B | 5h |

### 2.2 参数

| 参数 | 值 |
|------|---|
| OOP 范围 | `66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s` |
| IP 范围 | `QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+` |
| 底池 | 5BB |
| 有效筹码 | 25BB (SPR=5) / 100BB (SPR=20) |
| 抽水 | 0% |
| 下注尺寸 | 60%, e, a (3个尺寸) |
| 迭代次数 | 2000 |

### 2.3 测试矩阵

| 轮次 | SPR值 | 牌面数 | 说明 |
|------|:---:|:---:|------|
| 第1轮 | 5 | 8 | 小SPR验证基本收敛 |
| 第2轮 | 20 | 8 | 标准SPR验证 |
| 第3轮 | 随机5-20 | 8 | 泛化验证 |

## 三、验收标准

### 3.1 收敛性

- Turn DL exploitability < 1.0% pot
- DL solve 耗时 < Full solve 的 50%

### 3.2 EV 精度

- 修正前 Turn DL MAE < 30%
- 修正后（如有α）MAE < 15%
- 与 Flop DL + α 精度对比

### 3.3 策略一致性

- OOP Bet% ≤ Full solve × 1.5
- IP Bet% 与 Full solve 差距 ≤ 15pp

### 3.4 速度

- Turn DL solve < 30s (SPR=5) / < 120s (SPR=20)
- α修正 overhead < 0.1s

## 四、与 Flop DL 的对比基准

Flop DL (flop-dl-alpha分支) 已达到的指标:
- EV MAE: 10-14%
- 耗时: ~1s
- OOP 策略一致性: 99.4%
- IP bet 频率: 49% vs Full 54%

Turn DL 应达到更优的策略质量（因为有转牌层），但耗时会更高。

## 五、测试工具

主要使用 `bench_single` 工具:
```bash
cargo run --release --example bench_single -- \
  --flop "QsJh4d" --pot 5 --stack 25 \
  --dl-stage turn --iterations 2000 \
  --output "doc/turn_dl_test.csv"
```
