# サッカー映像に対するRAFTを用いた人物除外付きブレ補正

## 概要

本リポジトリは，サッカー映像に残るカメラ由来のブレを補正するためのプログラムである。

本研究では，スマートフォンや三脚で撮影したアマチュアサッカー映像を対象とし，選手追跡を行う前処理として映像のブレ補正を行うことを目的とする。

現在の実装では，芝生領域を背景情報として用いるだけでなく，YOLOv8x と BoT-SORT により検出した人物領域を除外し，選手や審判の局所的な動きがブレ検出および補正量推定に混ざることを抑える。

## 研究目的

サッカー映像では，風や手ブレ，三脚の揺れなどにより，映像全体にブレが発生することがある。  
このようなブレは，選手の検出位置を不安定にし，トラッキング精度を低下させる可能性がある。

また，サッカー映像では選手や審判が常に移動しているため，人物の局所的な動きがカメラ由来のブレとして誤って扱われる可能性がある。

そこで本研究では，選手追跡の前段階として，撮影映像に含まれるブレを検出し，人物領域を除外した背景領域に基づいて補正する手法を検討する。

## 現在の処理手順

現在の処理は，以下のSTEPで構成される。

1. YOLOv8x + BoT-SORT による人物領域の検出
2. Farneback Optical Flow によるブレ特徴量の抽出
3. ブレ区間の検出
4. ブレ区間に含まれるフレームの抽出
5. RAFT による Optical Flow 推定
6. 人物領域を除外した背景FlowからTranslation量を推定
7. 推定したズレと逆方向へのTranslation補正
8. 補正済みフレームを元動画へ合成

## 人物除外の考え方

本研究では，YOLOv8xを用いて入力フレーム中の `person` クラスを検出し，BoT-SORTを用いて人物領域を追跡する。検出された選手や審判のbbox領域を人物マスクとして扱い，Optical Flowの代表値推定から除外する。

使用する背景領域は以下である。

```text
芝生領域 AND 人物ではない領域
```

これにより，選手の走行方向や審判の移動方向に補正量が引っ張られることを抑え，背景領域に基づいたカメラ由来のブレ推定を行う。

## STEP1: ブレ特徴量の抽出

まず，入力動画に対してFarneback Optical Flowを用い，隣接フレーム間の動きを推定する。  
全フレームに対してRAFTを適用すると計算負荷が大きいため，ブレ区間の候補を抽出する段階ではFarneback Optical Flowを用いる。

この段階では，芝生の色情報を背景情報として利用する。さらに，YOLOv8x + BoT-SORTで検出した人物領域を除外し，以下の領域のみから移動量を求める。

```text
芝生領域 AND 人物ではない領域
```

また，平均値ではなく中央値を用いることで，選手やボールなどの局所的な動きの影響を抑える。

## STEP2: ブレ区間の検出

STEP1で得られたフレームごとの移動量や鮮明度情報をもとに，ブレが発生している可能性のある区間を検出する。  
連続してブレと判定されたフレームをまとめ，補正対象となるブレ区間として抽出する。

## STEP3: フレーム抽出

STEP2で検出したブレ区間に含まれるフレームを動画から抽出する。  
抽出したフレームは，RAFTによるOptical Flow推定の入力として使用する。

## STEP4: RAFTによるOptical Flow推定

検出されたブレ区間に対して，RAFTを用いて隣接フレーム間のOptical Flowを推定する。  
RAFTは，各画素が次のフレームにおいてどの方向にどれだけ移動したかを推定する手法である。

本研究では，RAFTの出力を最終結果として用いるのではなく，カメラ由来のブレを推定するための移動量情報として利用する。

## STEP5: Translation量の推定

RAFTによって得られたOptical Flowから，背景領域の移動量を推定する。  
現在の実装では，芝生領域を背景情報として利用し，さらにYOLOv8x + BoT-SORTで検出した人物領域を除外する。

その上で，背景領域内のFlowの中央値を代表値として用い，フレーム間の移動量 `tx, ty` を推定する。

これにより，選手や審判の移動方向に補正量が引っ張られることを抑え，背景に基づいたTranslation補正量を求める。

## STEP6: Translation補正

STEP5で推定した移動量 `tx, ty` に対して，フレームを逆方向 `-(tx, ty)` に平行移動する。  
これにより，カメラ由来のブレ成分を打ち消す。

補正にはOpenCVの `cv2.warpAffine()` を用いる。

## STEP7: 動画合成

補正したフレームを元動画の該当フレームと差し替え，補正後の動画を出力する。

## 評価

補正前後の映像に対して，平均移動量，軌跡揺れ量，急なジャンプ量などを比較し，ブレ低減率を算出する。

これにより，補正処理によって映像中のブレがどの程度低減されたかを評価する。

主な評価コードは以下である。

```text
evalute/evalute only translation.py
```

## 主なファイル構成

```text
person_mask_yolo.py
    YOLOv8x + BoT-SORTによる人物領域マスク作成

step1.py
    Farneback Optical Flowによるブレ特徴量抽出
    人物領域を除外した背景領域から代表移動量を推定

step2.py
    ブレ区間の検出

step3.py
    ブレ区間のフレーム抽出

step4_raft.py
    RAFTによるOptical Flow推定

step4_visualize_flow.py
    RAFT Flowの可視化

step7_composite.py
    補正済みフレームの動画合成

hosei houhou/translate/test_pipeline_1_7_translation.py
    STEP1からSTEP7までを実行するメインパイプライン
    PersonMaskYOLOを初期化し，STEP1とSTEP5に渡す

hosei houhou/translate/step5_translation.py
    RAFT FlowからTranslation量を推定
    人物領域を除外した背景Flowの中央値を利用

hosei houhou/translate/step6_stabilize_translation.py
    Translation補正を実行

evalute/evalute only translation.py
    補正前後のブレ低減率を評価
```

## 実行に必要なもの

人物除外版では，YOLOv8xを利用するため `ultralytics` が必要である。

```powershell
pip install ultralytics
```

また，YOLOv8xの学習済み重み `yolov8x.pt` が必要である。初回実行時に自動ダウンロードされるが，ネットワーク環境によって失敗する場合は，以下の場所に手動で配置する。

```text
C:\Users\ab23108\Documents\Codex\卒論用RAFT\yolov8x.pt
```

## 現在の特徴

- Farneback Optical Flowによりブレ区間を検出する
- RAFTによりブレ区間のOptical Flowを高精度に推定する
- 芝生の色情報を背景情報として利用する
- YOLOv8x + BoT-SORTにより人物領域を検出する
- 人物領域をFlowの代表値推定から除外する
- 背景領域の移動量を中央値で求める
- 推定したズレと逆方向にTranslation補正を行う
- 補正前後のブレ低減率を評価する

## 注意

動画ファイル，RAFTの学習済みモデル，YOLOの学習済みモデル，出力結果などの大容量ファイルはGitHubには含めない。

以下のようなファイルやフォルダは `.gitignore` により管理対象外とする。

```text
models/
results/
input_videos/
*.mp4
*.mov
*.avi
*.pth
*.pt
```

## 今後の課題

- ブレ検出に用いる閾値の見直し
- 人物除外による効果の定量評価
- 補正前後での選手トラッキング精度の比較
- Translation補正とAffine補正，Homography補正の比較
- 累積移動量の平滑化やfade in / fade outによるかくつき低減
- 夜間や照明条件が異なる映像への適用
