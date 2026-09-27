import cv2
import csv
import os
from pathlib import Path

import numpy as np


# ==================================================
# パス設定
# ==================================================

CURRENT_DIR = Path(__file__).resolve().parent


def find_project_root(start_dir):
    """
    RAFT直下フォルダを自動で探す。

    この評価コードを evalute フォルダ内から実行しても、
    RAFT直下を基準に input_videos / results を読めるようにする。
    """

    candidates = [start_dir] + list(start_dir.parents)

    for d in candidates:
        if (d / "input_videos").exists() and (d / "results").exists():
            return d

        if (d / "core" / "raft.py").exists():
            return d

    return start_dir


PROJECT_ROOT = find_project_root(CURRENT_DIR)
os.chdir(PROJECT_ROOT)

print("PROJECT_ROOT:", PROJECT_ROOT)


# ==================================================
# フォルダ設定
# ==================================================

RESULTS_DIR = PROJECT_ROOT / "results"
MP4_DIR = RESULTS_DIR / "mp4"
CSV_DIR = RESULTS_DIR / "CSV"
IMAGE_DIR = RESULTS_DIR / "gazou"

MP4_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 入力動画
# ==================================================

INPUT_VIDEO = PROJECT_ROOT / "input_videos" / "test 30sec.mp4"

# 基本はこのファイル名を使う
# 見つからない場合は、results/mp4 内から同じ手法名を含むmp4を自動で探す
HOMOGRAPHY_FILENAME = "0619_1408_RAFT_Homography.mp4"
AFFINE_FILENAME = "0619_142227_RAFT_Affine.mp4"
TRANSLATION_FILENAME = "0702_140250_RAFT_Translation.mp4"

OUTPUT_CSV = CSV_DIR / "evaluation_compare_background.csv"


def resolve_video_path(filename, keyword):
    """
    評価対象動画のパスを解決する。

    優先順位:
        1. results/mp4/filename
        2. results/filename
        3. results/mp4/*keyword*.mp4 のうち最新っぽいもの
    """

    exact_candidates = [
        MP4_DIR / filename,
        RESULTS_DIR / filename
    ]

    for path in exact_candidates:
        if path.exists():
            return path

    keyword_candidates = sorted(
        MP4_DIR.glob(f"*{keyword}*.mp4")
    )

    if len(keyword_candidates) > 0:
        # ファイル名順で最後を採用
        # 例: 0610 より 0625, 0701 が後に来る
        selected = keyword_candidates[-1]

        print(
            f"指定動画が見つからないため、代わりに使用: {selected}"
        )

        return selected

    available_mp4 = sorted(MP4_DIR.glob("*.mp4"))

    print("\n===== results/mp4 内のmp4一覧 =====")

    for p in available_mp4:
        print(p.name)

    raise FileNotFoundError(
        f"評価対象動画が見つかりません: {filename}"
    )


HOMOGRAPHY_VIDEO = resolve_video_path(
    HOMOGRAPHY_FILENAME,
    "Homography"
)

AFFINE_VIDEO = resolve_video_path(
    AFFINE_FILENAME,
    "Affine"
)

TRANSLATION_VIDEO = resolve_video_path(
    TRANSLATION_FILENAME,
    "Translation"
)


# ==================================================
# 評価設定
# ==================================================

RESIZE_SCALE = 0.5

# 画像端は補正後に黒帯・歪みが出やすいので除外
BORDER_RATIO = 0.05

# Flowを全部使わず、格子状にサンプリングする間隔
GRID_STEP = 8

# 背景候補点が少なすぎる場合の最低点数
MIN_BACKGROUND_POINTS = 50

# MAD外れ値除去の強さ
# 小さいほど選手などを強く除外
MAD_THRESHOLD = 3.0


# ==================================================
# 芝生・背景マスク設定
# HSVで緑色の領域を背景として使う
# ==================================================
# OpenCVのHSV
# H: 0〜179
# S: 0〜255
# V: 0〜255

GREEN_LOWER = np.array([30, 40, 40])
GREEN_UPPER = np.array([90, 255, 255])


def laplacian_variance(frame):
    """
    画像全体の鮮明さを評価する。
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def create_field_mask(frame):
    """
    芝生・背景領域のマスクを作る。
    選手やボールは基本的にこのマスクから外れる。
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    field_mask = cv2.inRange(
        hsv,
        GREEN_LOWER,
        GREEN_UPPER
    )

    # ノイズ除去
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    field_mask = cv2.morphologyEx(
        field_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    field_mask = cv2.morphologyEx(
        field_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return field_mask


def robust_camera_translation(prev_frame, curr_frame):
    """
    選手の動きを除外し、背景領域だけから
    カメラのx方向・y方向移動量を推定する。

    Returns:
        tx:
            x方向のカメラ移動量

        ty:
            y方向のカメラ移動量

        motion:
            sqrt(tx^2 + ty^2)

        used_points:
            実際に評価に使った背景点数
    """

    prev_small = cv2.resize(
        prev_frame,
        None,
        fx=RESIZE_SCALE,
        fy=RESIZE_SCALE
    )

    curr_small = cv2.resize(
        curr_frame,
        None,
        fx=RESIZE_SCALE,
        fy=RESIZE_SCALE
    )

    prev_gray = cv2.cvtColor(
        prev_small,
        cv2.COLOR_BGR2GRAY
    )

    curr_gray = cv2.cvtColor(
        curr_small,
        cv2.COLOR_BGR2GRAY
    )

    # Farneback Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    height, width = prev_gray.shape[:2]

    # 芝生領域だけを背景として使う
    field_mask = create_field_mask(
        prev_small
    )

    # 画像端を除外
    margin_x = int(width * BORDER_RATIO)
    margin_y = int(height * BORDER_RATIO)

    valid_area = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    valid_area[
        margin_y:height - margin_y,
        margin_x:width - margin_x
    ] = 255

    background_mask = cv2.bitwise_and(
        field_mask,
        valid_area
    )

    # 格子状にサンプリング
    ys, xs = np.mgrid[
        0:height:GRID_STEP,
        0:width:GRID_STEP
    ]

    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    mask_values = background_mask[ys, xs]

    valid = mask_values > 0

    xs = xs[valid]
    ys = ys[valid]

    if len(xs) < MIN_BACKGROUND_POINTS:
        return 0.0, 0.0, 0.0, len(xs)

    flow_x = flow[ys, xs, 0].astype(np.float64)
    flow_y = flow[ys, xs, 1].astype(np.float64)

    finite = (
        np.isfinite(flow_x)
        & np.isfinite(flow_y)
    )

    flow_x = flow_x[finite]
    flow_y = flow_y[finite]

    if len(flow_x) < MIN_BACKGROUND_POINTS:
        return 0.0, 0.0, 0.0, len(flow_x)

    # --------------------------------------------------
    # 中央値で背景の代表移動を推定
    # --------------------------------------------------

    median_x = np.median(flow_x)
    median_y = np.median(flow_y)

    # --------------------------------------------------
    # MADで選手・ボールなどの外れ値を除外
    # --------------------------------------------------

    mad_x = np.median(
        np.abs(flow_x - median_x)
    )

    mad_y = np.median(
        np.abs(flow_y - median_y)
    )

    scale_x = max(
        1.4826 * mad_x,
        1e-6
    )

    scale_y = max(
        1.4826 * mad_y,
        1e-6
    )

    inlier_mask = (
        np.abs(flow_x - median_x)
        <= MAD_THRESHOLD * scale_x
    ) & (
        np.abs(flow_y - median_y)
        <= MAD_THRESHOLD * scale_y
    )

    background_flow_x = flow_x[inlier_mask]
    background_flow_y = flow_y[inlier_mask]

    if len(background_flow_x) < MIN_BACKGROUND_POINTS:
        tx = float(median_x)
        ty = float(median_y)
        used_points = len(flow_x)
    else:
        tx = float(
            np.median(background_flow_x)
        )

        ty = float(
            np.median(background_flow_y)
        )

        used_points = len(background_flow_x)

    motion = float(
        np.sqrt(tx ** 2 + ty ** 2)
    )

    return tx, ty, motion, used_points


def evaluate_one_video(video_path):
    """
    1本の動画を評価する。
    """

    video_path = Path(video_path)

    print(f"動画パス: {video_path}")

    if not video_path.exists():
        raise FileNotFoundError(
            f"動画ファイルが存在しません: {video_path}"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"動画を開けません: {video_path}"
        )

    tx_values = []
    ty_values = []
    camera_motion_values = []
    used_points_values = []
    lap_values = []

    prev_frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        lap_values.append(
            laplacian_variance(frame)
        )

        if prev_frame is not None:
            tx, ty, motion, used_points = (
                robust_camera_translation(
                    prev_frame,
                    frame
                )
            )

            tx_values.append(tx)
            ty_values.append(ty)
            camera_motion_values.append(motion)
            used_points_values.append(used_points)

        prev_frame = frame
        frame_count += 1

    cap.release()

    if frame_count < 2:
        raise RuntimeError(
            f"評価に必要なフレーム数が不足しています: {video_path}"
        )

    tx_values = np.array(
        tx_values,
        dtype=np.float64
    )

    ty_values = np.array(
        ty_values,
        dtype=np.float64
    )

    camera_motion_values = np.array(
        camera_motion_values,
        dtype=np.float64
    )

    lap_values = np.array(
        lap_values,
        dtype=np.float64
    )

    used_points_values = np.array(
        used_points_values,
        dtype=np.float64
    )

    avg_camera_motion = float(
        np.mean(camera_motion_values)
    )

    std_camera_motion = float(
        np.std(camera_motion_values)
    )

    avg_lap = float(
        np.mean(lap_values)
    )

    avg_used_points = float(
        np.mean(used_points_values)
    )

    if len(camera_motion_values) > 2:
        trajectory_variation = float(
            np.std(
                np.diff(camera_motion_values)
            )
        )
    else:
        trajectory_variation = 0.0

    mean_tx = float(
        np.mean(tx_values)
    )

    mean_ty = float(
        np.mean(ty_values)
    )

    mean_abs_tx = float(
        np.mean(
            np.abs(tx_values)
        )
    )

    mean_abs_ty = float(
        np.mean(
            np.abs(ty_values)
        )
    )

    return {
        "frame_count": frame_count,
        "avg_camera_motion": avg_camera_motion,
        "std_camera_motion": std_camera_motion,
        "trajectory_variation": trajectory_variation,
        "mean_tx": mean_tx,
        "mean_ty": mean_ty,
        "mean_abs_tx": mean_abs_tx,
        "mean_abs_ty": mean_abs_ty,
        "avg_laplacian": avg_lap,
        "avg_used_points": avg_used_points
    }


def main():
    videos = {
        "Original": INPUT_VIDEO,
        "RAFT_Homography": HOMOGRAPHY_VIDEO,
        "RAFT_Affine": AFFINE_VIDEO,
        "RAFT_Translation": TRANSLATION_VIDEO
    }

    print("\n===== 評価対象動画 =====")

    for name, path in videos.items():
        print(f"{name}: {path}")

    results = {}

    for name, path in videos.items():
        print(f"\n評価中: {name}")
        results[name] = evaluate_one_video(
            path
        )

    original_motion = (
        results["Original"]["avg_camera_motion"]
    )

    original_variation = (
        results["Original"]["trajectory_variation"]
    )

    rows = []

    for name, result in results.items():
        camera_motion_reduction = (
            100
            * (
                original_motion
                - result["avg_camera_motion"]
            )
            / original_motion
            if original_motion > 0
            else 0.0
        )

        trajectory_improvement = (
            100
            * (
                original_variation
                - result["trajectory_variation"]
            )
            / original_variation
            if original_variation > 0
            else 0.0
        )

        row = {
            "method": name,
            "frame_count": result["frame_count"],
            "avg_camera_motion": result["avg_camera_motion"],
            "camera_motion_reduction_percent": camera_motion_reduction,
            "std_camera_motion": result["std_camera_motion"],
            "trajectory_variation": result["trajectory_variation"],
            "trajectory_improvement_percent": trajectory_improvement,
            "mean_tx": result["mean_tx"],
            "mean_ty": result["mean_ty"],
            "mean_abs_tx": result["mean_abs_tx"],
            "mean_abs_ty": result["mean_abs_ty"],
            "avg_laplacian": result["avg_laplacian"],
            "avg_used_background_points": result["avg_used_points"]
        }

        rows.append(row)

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "frame_count",
                "avg_camera_motion",
                "camera_motion_reduction_percent",
                "std_camera_motion",
                "trajectory_variation",
                "trajectory_improvement_percent",
                "mean_tx",
                "mean_ty",
                "mean_abs_tx",
                "mean_abs_ty",
                "avg_laplacian",
                "avg_used_background_points"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n===== 背景ベース評価結果 =====")

    for row in rows:
        print(f"\n[{row['method']}]")

        print(
            f"平均カメラ移動量: "
            f"{row['avg_camera_motion']:.6f}"
        )

        print(
            f"カメラ揺れ低減率: "
            f"{row['camera_motion_reduction_percent']:.2f}%"
        )

        print(
            f"カメラ移動量標準偏差: "
            f"{row['std_camera_motion']:.6f}"
        )

        print(
            f"カメラ軌跡変動量: "
            f"{row['trajectory_variation']:.6f}"
        )

        print(
            f"軌跡平滑化改善率: "
            f"{row['trajectory_improvement_percent']:.2f}%"
        )

        print(
            f"平均tx: "
            f"{row['mean_tx']:.6f}"
        )

        print(
            f"平均ty: "
            f"{row['mean_ty']:.6f}"
        )

        print(
            f"平均|tx|: "
            f"{row['mean_abs_tx']:.6f}"
        )

        print(
            f"平均|ty|: "
            f"{row['mean_abs_ty']:.6f}"
        )

        print(
            f"Laplacian Variance: "
            f"{row['avg_laplacian']:.2f}"
        )

        print(
            f"平均使用背景点数: "
            f"{row['avg_used_background_points']:.1f}"
        )

    print(f"\nCSV保存完了: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
