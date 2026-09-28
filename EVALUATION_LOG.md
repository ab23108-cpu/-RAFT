# ブレ補正評価ログ

このファイルは `evalute/evalute only translation.py` の評価結果を記録するためのメモである。
GitHubには動画本体ではなく、評価結果と変更内容を残す。

## 評価コード

```text
evalute/evalute only translation.py
```

評価対象の入力動画:

```text
input_videos/test 30sec.mp4
```

フレーム数:

```text
935 frames
```

## 評価指標の意味

| 指標 | 意味 |
|---|---|
| 平均カメラ移動量 | フレーム間でどれだけ画面全体が動いたかの平均 |
| カメラ揺れ低減率 | Originalに対して平均移動量がどれだけ減ったか |
| フレーム間移動量 p95 / p99 / 最大 | 瞬間的な移動量の大きい側の値 |
| 軌跡揺れ量 p95 / p99 / 最大 | 累積軌跡から見た揺れの大きさ |
| 軌跡揺れ低減率 | Originalに対して軌跡の揺れがどれだけ減ったか |
| 急なジャンプ量 | フレーム間で急に動いた量 |
| 平均Laplacian分散 | 鮮明度の目安。低くなるとややぼけた可能性がある |

## これまでの主な評価結果

### 1. 初期Translation補正

CSV:

```text
results/CSV/evaluation_compare_translation_only.csv
```

| 項目 | Original | RAFT_Translation | 低減率 |
|---|---:|---:|---:|
| 平均カメラ移動量 | 0.010909 | 0.008367 | 23.30% |
| 軌跡揺れ量 | 0.022863 | 0.020314 | 11.15% |
| 平均Laplacian分散 | 248.14 | 234.61 | - |

メモ:

- Translation補正により平均移動量は約23%低減した。
- 一方で鮮明度の目安であるLaplacian分散は低下しており、補正による軽いぼけの可能性がある。

### 2. 改善版Translation補正

CSV:

```text
results/CSV/evaluation_compare_translation_improved.csv
```

評価対象補正動画:

```text
results/0928_144522_RAFT_Translation.mp4
```

| 項目 | Original | RAFT_Translation | 低減率 |
|---|---:|---:|---:|
| 平均カメラ移動量 | 0.021817 | 0.016094 | 26.23% |
| フレーム間移動量 p95 | 0.124181 | 0.090582 | - |
| フレーム間移動量 p99 | 0.209583 | 0.161710 | - |
| 最大ジャンプ量 | 0.281276 | 0.212713 | 24.38% |
| 軌跡揺れ量 p95 | 0.384981 | 0.307265 | 20.19% |
| 軌跡揺れ量 p99 | 1.004007 | 0.648777 | 35.38% |
| 1.0pxを超える急ジャンプ回数 | 0 | 0 | - |
| 平均Laplacian分散 | 248.14 | 234.06 | - |

メモ:

- 平均カメラ移動量は約26.23%低減した。
- 軌跡揺れ量p99は約35.38%低減しており、大きめの揺れに対して効果が出ている。
- 最大ジャンプ量も約24.38%低減した。
- 平均Laplacian分散は低下しているため、補正によりわずかにぼけが増えている可能性がある。

## 人物除外版の扱い

現在の作業ツリーでは、以下の変更を加えている。
まだGitには保存していない。

対象ファイル:

```text
step1.py
hosei houhou/translate/step5_translation.py
hosei houhou/translate/test_pipeline_1_7_translation.py
```

変更内容:

- STEP1で、芝生領域だけでなく人物領域を除外した背景領域からFarneback Flowの代表値を求める。
- STEP5で、RAFT Flowのうち人物領域を除外し、背景領域のみからTranslation量 `tx, ty` を推定する。
- pipelineで `PersonMaskYOLO` を初期化し、STEP1とSTEP5に渡す。

## 今後の評価結果を追記する欄

### 評価日: YYYY/MM/DD

補正動画:

```text
results/xxxx_RAFT_Translation.mp4
```

変更内容:

- 例: 人物除外を有効化
- 例: 閾値変更
- 例: correction_strength変更

| 項目 | Original | RAFT_Translation | 低減率 |
|---|---:|---:|---:|
| 平均カメラ移動量 |  |  |  |
| フレーム間移動量 p95 |  |  |  |
| フレーム間移動量 p99 |  |  |  |
| 最大ジャンプ量 |  |  |  |
| 軌跡揺れ量 p95 |  |  |  |
| 軌跡揺れ量 p99 |  |  |  |
| 1.0pxを超える急ジャンプ回数 |  |  |  |
| 平均Laplacian分散 |  |  |  |

考察:

- 
