# Web 演示（Web Demo）

`web_demo/app.py`（Flask，纯 HTTP/音频管道）+ `web_demo/retrieval_core.py`（**打分核心，可自检**）。

## 打分协议（与训练/评测逐字节一致）

```
a_t = normalize(g_audio(AuT(x)_t))              # [T, 512] 帧嵌入
k_h = normalize(g_text(mean_tokens(E_tok(h))))  # [512]
cos(h) = max_t <a_t, k_h>                       # ← 唯一排序分数
```

其余列**不参与排序**，仅用于解释/定位：

| 字段 | 含义 |
|---|---|
| seg | 滑窗平均余弦峰值 → 定位片段 [start, end]（可播放） |
| contrast | seg - 曲线均值（峰度） |
| z | (峰值-均值)/std —— **实测无用**，保留仅展示 |
| pool_max | 同一音频上 256 个随机池词的最高分（"随机词底线"） |
| prob / prob_display | 相对 softmax 置信（展示用 scale=10；训练温度 ≈67 会饱和） |
| verdict | 绝对标定带（见下） |

## 阈值标定带（VERDICT_BANDS）

来源：valid 中文查询 vs 真实实体 + 999 随机同语言池词（两个子域实测）：

| 域 | 真实体 cos | 干扰词 p95 | R@1 (cos) |
|---|---|---|---|
| 会议 TTS（≈14.1s, n=120） | 0.697 ± 0.052 | 0.699 | 0.808 |
| 短朗读（≈4.6s, n=118） | 0.722 ± 0.040 | 0.670 | 0.966 |

| cos ≥ | 标签 | 预估精度 |
|---|---|---|
| 0.72 | 高置信命中 | 0.95 |
| 0.66 | 较可能命中 | 0.72 |
| 0.60 | 弱匹配 | 0.60 |
| else | 证据不足（不宣称"未出现"） | 0.15 |

**关键洞察**：阈值依赖**音频长度而非语言**——越长，max-over-time 撞随机词高分的上限越高
（4.6s → p95 0.670；14s → 0.699；67s+ → 0.72+）。因此**绝对阈值只是粗略参考**，
优先看 `margin_over_pool`（本音频上超出随机词最高分的余量）。

## 时间定位

- 帧率非均匀 ~13 fps：`frame_times()` 反演 vendor 长度公式（`_get_feat_extract_output_lengths`），逐帧精确时间戳
  （块内 0.04s、块间 0.08s 间隙，按均匀 1/13s 推算 15s 音频可漂移 ~1s）；
- 滑窗宽度 `window_frames(h) = 0.28s + 0.13s × 字数`（限 2–40 帧）；
- 实测定位误差：**中位数 0.51s，66% 落在 1s 内**（窗口中心 vs 标注词位置）。

## 使用

```bash
CUDA_VISIBLE_DEVICES=0 python3.10 web_demo/app.py --port 8899
# 支持浏览器录音（webm/opus 自动 ffmpeg 转码 16k）
# /api/selfcheck：核验 frame_times 反演、滑窗和、batch 一致性、打分协议四件事
# /api/sample：从已重定位的 data/glclap_hotword/valid.jsonl 随机选取样例
```
