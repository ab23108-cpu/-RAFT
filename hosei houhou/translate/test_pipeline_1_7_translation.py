import sys
import os
import numpy as np
import cv2
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent


def find_quiet_segments(
    score,
    existing_segments,
    fps=30,
    window_sec=1.0,
    num_segments=1
):
    score = np.asarray(score, dtype=np.float64)

    window = int(round(fps * window_sec))
    window = max(window, 1)

    if len(score) < window:
        return []

    blocked = np.zeros(len(score), dtype=bool)

    for start, end in existing_segments:
        blocked[start:end + 1] = True

    candidates = []

    for start in range(0, len(score) - window + 1):
        end = start + window - 1

        if np.any(blocked[start:end + 1]):
            continue

        avg_score = float(np.mean(score[start:end + 1]))

        candidates.append((avg_score, start, end))

    candidates.sort(key=lambda x: x[0])

    quiet_segments = []
    used = np.zeros(len(score), dtype=bool)

    for avg_score, start, end in candidates:
        if len(quiet_segments) >= num_segments:
            break

        if np.any(used[start:end + 1]):
            continue

        quiet_segments.append([start, end])
        used[start:end + 1] = True

        print(
            f"静かな区間: {start} -> {end}, "
            f"平均score={avg_score:.4f}"
        )

    return quiet_segments
def print_segment_times(title, segments, fps):
    print(f"\n{title}")

    if len(segments) == 0:
        print("  なし")
        return

    for segment_id, (start, end) in enumerate(segments):
        start_sec = start / fps
        end_sec = end / fps
        duration_sec = (end - start + 1) / fps

        print(
            f"  Segment {segment_id}: "
            f"{start} -> {end} frame, "
            f"{start_sec:.2f}秒 -> {end_sec:.2f}秒, "
            f"長さ {duration_sec:.2f}秒"
        )


def filter_flows_by_segments(flows_dict, target_segments):
    filtered = {}

    for frame_idx, flow in flows_dict.items():
        for start, end in target_segments:
            if start <= frame_idx < end:
                filtered[frame_idx] = flow
                break

    return filtered


def find_project_root(start_dir):
    candidates = [start_dir] + list(start_dir.parents)

    for d in candidates:
        if (d / "core" / "raft.py").exists():
            return d

        if (d / "models" / "raft-things.pth").exists():
            return d

    return start_dir


PROJECT_ROOT = find_project_root(CURRENT_DIR)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))
sys.path.insert(0, str(CURRENT_DIR))

os.chdir(PROJECT_ROOT)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CURRENT_DIR:", CURRENT_DIR)
print("実行ファイル:", Path(__file__).resolve())


from step1 import Step1MotionBlurDetector
from step2 import Step2BlurSegmentExtractor
from step3 import Step3FrameExtractor
from step4_raft import Step4RAFTFlowEstimator
from step4_visualize_flow import save_raft_flow_images
from person_mask_yolo import PersonMaskYOLO
from step5_translation import Step5TranslationEstimator
from step6_stabilize_translation import Step6TranslationVideoStabilizer
from step7_composite import Step7VideoCompositor

from datetime import datetime
import contextlib
import io
import time


timestamp = datetime.now().strftime("%m%d_%H%M%S")

video_path = r"input_videos\test 30sec.mp4"
output_path = rf"results\{timestamp}_RAFT_Translation.mp4"

model_path = r"models\raft-things.pth"

motion_threshold = 0.025
laplacian_threshold = 300.0
resize_scale = 0.5

translation_grid_step = 10
translation_mad_threshold = 2.5
translation_border_ratio = 0.05
translation_min_points = 80
translation_green_lower = (30, 40, 40)
translation_green_upper = (90, 255, 255)
translation_use_grass_mask = True

correction_strength = 0.2
smoothing_kernel = 61
zoom_scale = 1.0
max_correction_px = 0.3

raft_iters = 16

# Noneなら全flow画像を保存する
flow_visualize_count = None

os.makedirs("results", exist_ok=True)

total_start = time.time()


print("\n===== STEP1 ブレ特徴量抽出 =====")

step_start = time.time()

step1 = Step1MotionBlurDetector(
    resize_width=640,
    use_grass_mask=True,
    use_median=True,
    green_lower=(30, 40, 40),
    green_upper=(90, 255, 255),
    min_mask_pixels=500,
    mad_threshold=3.0
)

frame_stats = step1.run(video_path)

step1_time = time.time() - step_start
print(f"STEP1時間: {step1_time:.2f} 秒")
print(f"frame_stats数: {len(frame_stats)}")


print("\n===== STEP2 ブレ区間抽出 =====")

step_start = time.time()

step2 = Step2BlurSegmentExtractor()

segments, score = step2.run(frame_stats)

step2_time = time.time() - step_start

print(f"STEP2時間: {step2_time:.2f} 秒")
print(f"検出区間数: {len(segments)}")

if len(segments) == 0:
    print("\n補正対象となる区間が検出されませんでした。")
    sys.exit(0)

print("検出区間一覧:")

for segment_id, (start, end) in enumerate(segments):
    print(
        f"  Segment {segment_id}: "
        f"{start} -> {end} "
        f"({end - start + 1} frames)"
    )

quiet_segments = find_quiet_segments(
    score=score,
    existing_segments=segments,
    fps=30,
    window_sec=1.0,
    num_segments=1
)

visual_segments = segments + quiet_segments

cap_info = cv2.VideoCapture(video_path)
fps = cap_info.get(cv2.CAP_PROP_FPS)
cap_info.release()

if fps <= 0:
    fps = 30.0

print(f"動画FPS: {fps:.2f}")

print_segment_times(
    "揺れが多かった区間",
    segments,
    fps
)

print_segment_times(
    "揺れが少なかった区間",
    quiet_segments,
    fps
)

print(f"静かな比較区間数: {len(quiet_segments)}")
print("quiet_segments:", quiet_segments)
print("visual_segments:", visual_segments)
print(f"静かな比較区間数: {len(quiet_segments)}")


print("\n===== STEP3 対象フレーム抽出 =====")
step_start = time.time()

step3 = Step3FrameExtractor()

with contextlib.redirect_stdout(io.StringIO()):
      frames_dict = step3.extract_frames(
      video_path,
       visual_segments
  )

step3_time = time.time() - step_start

print(f"STEP3時間: {step3_time:.2f} 秒")
print(f"抽出フレーム数: {len(frames_dict)}")

if len(frames_dict) < 2:
    print("\nRAFTに必要なフレーム数が不足しています。")
    sys.exit(0)


print("\n===== STEP4 RAFT Optical Flow =====")
step_start = time.time()

step4 = Step4RAFTFlowEstimator(
    model_path=model_path
)

with contextlib.redirect_stdout(io.StringIO()):
    flows_dict = step4.estimate_flows(
        frames_dict,
        visual_segments,
        iters=raft_iters
    )

step4_time = time.time() - step_start

print(f"STEP4時間: {step4_time:.2f} 秒")
print(f"RAFT Flow数: {len(flows_dict)}")

if len(flows_dict) == 0:
    print("\nOptical Flowが生成されませんでした。")
    sys.exit(0)


print("\n===== STEP4.5 RAFT Flow発表用画像保存 =====")
flow_output_dir = rf"results\{timestamp}_raft_flow_images"

blur_flows_dict = filter_flows_by_segments(
    flows_dict,
    segments
)

quiet_flows_dict = filter_flows_by_segments(
    flows_dict,
    quiet_segments
)

print("\n===== STEP4.5 RAFT Flow発表用画像保存 =====")
print(f"全flows_dict数: {len(flows_dict)}")
print(f"ブレ区間flow数: {len(blur_flows_dict)}")
print(f"静かな区間flow数: {len(quiet_flows_dict)}")
print(f"保存先: {Path(flow_output_dir).resolve()}")

save_raft_flow_images(
    frames_dict=frames_dict,
    flows_dict=blur_flows_dict,
    output_dir=rf"{flow_output_dir}\blur",
    max_examples=flow_visualize_count
)

save_raft_flow_images(
    frames_dict=frames_dict,
    flows_dict=quiet_flows_dict,
    output_dir=rf"{flow_output_dir}\quiet",
    max_examples=flow_visualize_count
)


print("\n===== STEP5 Translation推定 =====")
step_start = time.time()

print(f"Step5 grid_step: {translation_grid_step}")
print(f"Step5 mad_threshold: {translation_mad_threshold}")
print(f"Step5 min_points: {translation_min_points}")

step5 = Step5TranslationEstimator(
    grid_step=translation_grid_step,
    mad_threshold=translation_mad_threshold,
    border_ratio=translation_border_ratio,
    min_points=translation_min_points,
    green_lower=translation_green_lower,
    green_upper=translation_green_upper,
    use_grass_mask=translation_use_grass_mask
)

translations_dict = step5.estimate_translations(
    flows_dict,
    frames_dict=frames_dict
)

step5_time = time.time() - step_start

print(f"STEP5時間: {step5_time:.2f} 秒")
print(f"Translation数: {len(translations_dict)}")


print("\n===== STEP6 Translation安定化 =====")
step_start = time.time()

step6 = Step6TranslationVideoStabilizer(
    correction_strength=correction_strength,
    smoothing_kernel=smoothing_kernel,
    zoom_scale=zoom_scale,
    max_correction_px=max_correction_px
)

with contextlib.redirect_stdout(io.StringIO()):
    stabilized_frames_dict = step6.stabilize_segments(
        frames_dict,
        segments,
        translations_dict
    )

step6_time = time.time() - step_start

print(f"STEP6時間: {step6_time:.2f} 秒")
print(f"補正強度: {correction_strength}")
print(f"smoothing_kernel: {smoothing_kernel}")
print(f"zoom_scale: {zoom_scale}")
print(f"max_correction_px: {max_correction_px}")
print(f"安定化フレーム数: {len(stabilized_frames_dict)}")


print("\n===== STEP7 合成・出力 =====")
step_start = time.time()

step7 = Step7VideoCompositor()

success = step7.composite_video(
    input_video_path=video_path,
    stabilized_frames_dict=stabilized_frames_dict,
    output_video_path=output_path
)

step7_time = time.time() - step_start
total_time = time.time() - total_start


print("\n===== Translation版 STEP1〜7 完了 =====")
print(f"出力成功: {success}")
print(f"出力動画: {output_path}")
print(f"RAFT Flow画像保存先: {flow_output_dir}")

print("\n==============================")
print("処理時間まとめ")
print("==============================")
print(f"STEP1: {step1_time:.2f} 秒")
print(f"STEP2: {step2_time:.2f} 秒")
print(f"STEP3: {step3_time:.2f} 秒")
print(f"STEP4: {step4_time:.2f} 秒")
print(f"STEP5: {step5_time:.2f} 秒")
print(f"STEP6: {step6_time:.2f} 秒")
print(f"STEP7: {step7_time:.2f} 秒")
print("------------------------------")
print(f"合計処理時間: {total_time:.2f} 秒")
print(f"合計処理時間: {total_time / 60:.2f} 分")