import cv2
import csv
import os
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent


def find_project_root(start_dir):
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


RESULTS_DIR = PROJECT_ROOT / "results"
MP4_DIR = RESULTS_DIR / "mp4"
CSV_DIR = RESULTS_DIR / "CSV"

MP4_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


INPUT_VIDEO = PROJECT_ROOT / "input_videos" / "test 30sec.mp4"

TRANSLATION_FILENAME = "0710_153407_RAFT_Translation.mp4"
TRANSLATION_VIDEO = MP4_DIR / TRANSLATION_FILENAME

OUTPUT_CSV = CSV_DIR / "evaluation_compare_translation_improved.csv"


RESIZE_SCALE = 0.5
BORDER_RATIO = 0.05
GRID_STEP = 8
MIN_BACKGROUND_POINTS = 50
MAD_THRESHOLD = 3.0

GREEN_LOWER = np.array([30, 40, 40], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)

TRAJECTORY_SMOOTHING_KERNEL = 61
JUMP_ALERT_THRESHOLD_PX = 1.0


def resolve_video_path(filename, keyword):
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
        selected = keyword_candidates[-1]
        print(f"指定動画が見つからないため代わりに使用: {selected}")
        return selected

    available_mp4 = sorted(MP4_DIR.glob("*.mp4"))

    print("\n===== results/mp4 内のmp4一覧 =====")
    for p in available_mp4:
        print(p.name)

    raise FileNotFoundError(
        f"評価対象動画が見つかりません: {filename}"
    )


TRANSLATION_VIDEO = resolve_video_path(
    TRANSLATION_FILENAME,
    "Translation"
)


def laplacian_variance(frame):
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
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    field_mask = cv2.inRange(
        hsv,
        GREEN_LOWER,
        GREEN_UPPER
    )

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


def valid_area_mask(height, width):
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

    return valid_area


def robust_camera_translation(prev_frame, curr_frame):
    prev_small = cv2.resize(
        prev_frame,
        None,
        fx=RESIZE_SCALE,
        fy=RESIZE_SCALE,
        interpolation=cv2.INTER_AREA
    )

    curr_small = cv2.resize(
        curr_frame,
        None,
        fx=RESIZE_SCALE,
        fy=RESIZE_SCALE,
        interpolation=cv2.INTER_AREA
    )

    prev_gray = cv2.cvtColor(
        prev_small,
        cv2.COLOR_BGR2GRAY
    )

    curr_gray = cv2.cvtColor(
        curr_small,
        cv2.COLOR_BGR2GRAY
    )

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

    field_mask = create_field_mask(prev_small)
    valid_area = valid_area_mask(height, width)

    background_mask = cv2.bitwise_and(
        field_mask,
        valid_area
    )

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

    median_x = np.median(flow_x)
    median_y = np.median(flow_y)

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
        tx_small = float(median_x)
        ty_small = float(median_y)
        used_points = len(flow_x)
    else:
        tx_small = float(
            np.median(background_flow_x)
        )

        ty_small = float(
            np.median(background_flow_y)
        )

        used_points = len(background_flow_x)

    tx = tx_small / RESIZE_SCALE
    ty = ty_small / RESIZE_SCALE

    motion = float(
        np.sqrt(tx ** 2 + ty ** 2)
    )

    return tx, ty, motion, used_points


def smooth_signal(values, kernel_size):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) < 3:
        return values.copy()

    k = min(kernel_size, len(values))

    if k % 2 == 0:
        k -= 1

    if k < 3:
        return values.copy()

    padded = np.pad(
        values,
        pad_width=k // 2,
        mode="edge"
    )

    smoothed = np.zeros_like(values)

    for i in range(len(values)):
        smoothed[i] = np.median(
            padded[i:i + k]
        )

    return smoothed


def summarize_percentiles(values):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) == 0:
        return {
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0
        }

    return {
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values))
    }


def evaluate_one_video(video_path):
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

        if frame_count % 100 == 0:
            print(f"  評価中: {frame_count}")

    cap.release()

    if frame_count < 2:
        raise RuntimeError(
            f"評価に必要なフレーム数が不足しています: {video_path}"
        )

    tx_values = np.asarray(
        tx_values,
        dtype=np.float64
    )

    ty_values = np.asarray(
        ty_values,
        dtype=np.float64
    )

    camera_motion_values = np.asarray(
        camera_motion_values,
        dtype=np.float64
    )

    lap_values = np.asarray(
        lap_values,
        dtype=np.float64
    )

    used_points_values = np.asarray(
        used_points_values,
        dtype=np.float64
    )

    trajectory_x = np.cumsum(tx_values)
    trajectory_y = np.cumsum(ty_values)

    smooth_x = smooth_signal(
        trajectory_x,
        TRAJECTORY_SMOOTHING_KERNEL
    )

    smooth_y = smooth_signal(
        trajectory_y,
        TRAJECTORY_SMOOTHING_KERNEL
    )

    shake_x = trajectory_x - smooth_x
    shake_y = trajectory_y - smooth_y

    shake_magnitude = np.sqrt(
        shake_x ** 2
        + shake_y ** 2
    )

    frame_jump = np.sqrt(
        tx_values ** 2
        + ty_values ** 2
    )

    motion_stats = summarize_percentiles(
        camera_motion_values
    )

    shake_stats = summarize_percentiles(
        shake_magnitude
    )

    jump_stats = summarize_percentiles(
        frame_jump
    )

    jump_frame_indices = np.where(
        frame_jump > JUMP_ALERT_THRESHOLD_PX
    )[0] + 1

    top_jump_order = np.argsort(
        frame_jump
    )[::-1][:10]

    top_jumps = [
        (
            int(i + 1),
            float(frame_jump[i]),
            float(tx_values[i]),
            float(ty_values[i])
        )
        for i in top_jump_order
    ]

    return {
        "frame_count": frame_count,

        "avg_camera_motion": float(
            np.mean(camera_motion_values)
        ),
        "std_camera_motion": float(
            np.std(camera_motion_values)
        ),

        "motion_median": motion_stats["median"],
        "motion_p75": motion_stats["p75"],
        "motion_p90": motion_stats["p90"],
        "motion_p95": motion_stats["p95"],
        "motion_p99": motion_stats["p99"],
        "motion_max": motion_stats["max"],

        "shake_median": shake_stats["median"],
        "shake_p75": shake_stats["p75"],
        "shake_p90": shake_stats["p90"],
        "shake_p95": shake_stats["p95"],
        "shake_p99": shake_stats["p99"],
        "shake_max": shake_stats["max"],

        "frame_jump_median": jump_stats["median"],
        "frame_jump_p75": jump_stats["p75"],
        "frame_jump_p90": jump_stats["p90"],
        "frame_jump_p95": jump_stats["p95"],
        "frame_jump_p99": jump_stats["p99"],
        "frame_jump_max": jump_stats["max"],

        "jump_over_threshold_count": int(
            len(jump_frame_indices)
        ),
        "jump_over_threshold_frames": " ".join(
            map(str, jump_frame_indices[:50])
        ),

        "mean_tx": float(
            np.mean(tx_values)
        ),
        "mean_ty": float(
            np.mean(ty_values)
        ),
        "mean_abs_tx": float(
            np.mean(np.abs(tx_values))
        ),
        "mean_abs_ty": float(
            np.mean(np.abs(ty_values))
        ),

        "avg_laplacian": float(
            np.mean(lap_values)
        ),
        "avg_used_points": float(
            np.mean(used_points_values)
        ),

        "top_jumps": top_jumps
    }


def percent_change(original, current):
    if original == 0:
        return 0.0

    return 100.0 * (
        original - current
    ) / original


def main():
    videos = {
        "Original": INPUT_VIDEO,
        "RAFT_Translation": TRANSLATION_VIDEO
    }

    print("\n===== 評価対象動画 =====")
    for name, path in videos.items():
        print(f"{name}: {path}")

    results = {}

    for name, path in videos.items():
        print(f"\n評価中: {name}")
        results[name] = evaluate_one_video(path)

    original = results["Original"]

    rows = []

    for name, result in results.items():
        row = {
            "method": name,
            "frame_count": result["frame_count"],

            "avg_camera_motion": result["avg_camera_motion"],
            "avg_camera_motion_reduction_percent": percent_change(
                original["avg_camera_motion"],
                result["avg_camera_motion"]
            ),

            "motion_p95": result["motion_p95"],
            "motion_p99": result["motion_p99"],
            "motion_max": result["motion_max"],

            "shake_median": result["shake_median"],
            "shake_p95": result["shake_p95"],
            "shake_p99": result["shake_p99"],
            "shake_max": result["shake_max"],

            "shake_p95_reduction_percent": percent_change(
                original["shake_p95"],
                result["shake_p95"]
            ),

            "shake_p99_reduction_percent": percent_change(
                original["shake_p99"],
                result["shake_p99"]
            ),

            "frame_jump_p95": result["frame_jump_p95"],
            "frame_jump_p99": result["frame_jump_p99"],
            "frame_jump_max": result["frame_jump_max"],

            "frame_jump_max_reduction_percent": percent_change(
                original["frame_jump_max"],
                result["frame_jump_max"]
            ),

            "jump_over_threshold_count": result[
                "jump_over_threshold_count"
            ],

            "jump_over_threshold_frames": result[
                "jump_over_threshold_frames"
            ],

            "mean_tx": result["mean_tx"],
            "mean_ty": result["mean_ty"],
            "mean_abs_tx": result["mean_abs_tx"],
            "mean_abs_ty": result["mean_abs_ty"],

            "avg_laplacian": result["avg_laplacian"],
            "avg_used_background_points": result["avg_used_points"]
        }

        rows.append(row)

    fieldnames = list(rows[0].keys())

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n===== 改善版 評価結果 =====")

    for row in rows:
        print(f"\n[{row['method']}]")

        print(
            f"平均カメラ移動量: "
            f"{row['avg_camera_motion']:.6f} px/frame"
        )

        print(
            f"カメラ揺れ低減率: "
            f"{row['avg_camera_motion_reduction_percent']:.2f}%"
        )

        print(
            f"フレーム間移動量 p95 / p99 / 最大: "
            f"{row['motion_p95']:.6f} / "
            f"{row['motion_p99']:.6f} / "
            f"{row['motion_max']:.6f} px/frame"
        )

        print(
            f"軌跡揺れ量 p95 / p99 / 最大: "
            f"{row['shake_p95']:.6f} / "
            f"{row['shake_p99']:.6f} / "
            f"{row['shake_max']:.6f} px"
        )

        print(
            f"軌跡揺れ低減率 p95 / p99: "
            f"{row['shake_p95_reduction_percent']:.2f}% / "
            f"{row['shake_p99_reduction_percent']:.2f}%"
        )

        print(
            f"急なジャンプ量 p95 / p99 / 最大: "
            f"{row['frame_jump_p95']:.6f} / "
            f"{row['frame_jump_p99']:.6f} / "
            f"{row['frame_jump_max']:.6f} px/frame"
        )

        print(
            f"最大ジャンプ低減率: "
            f"{row['frame_jump_max_reduction_percent']:.2f}%"
        )

        print(
            f"1.0pxを超える急ジャンプ回数: "
            f"{row['jump_over_threshold_count']} 回"
        )

        if row["jump_over_threshold_frames"]:
            print(
                f"急ジャンプ発生フレーム: "
                f"{row['jump_over_threshold_frames']}"
            )

        print(
            f"平均Laplacian分散（鮮明度目安）: "
            f"{row['avg_laplacian']:.2f}"
        )
    print("\n===== 最大ジャンプ上位 =====")

    for name, result in results.items():
        print(f"\n[{name}]")

        for frame_idx, jump, tx, ty in result["top_jumps"]:
            print(
                f"frame {frame_idx}: "
                f"jump={jump:.4f}, "
                f"tx={tx:.4f}, "
                f"ty={ty:.4f}"
            )

    print(f"\nCSV保存完了: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
