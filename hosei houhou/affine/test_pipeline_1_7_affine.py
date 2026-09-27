from step1 import Step1MotionBlurDetector
from step2 import Step2BlurSegmentExtractor
from step3 import Step3FrameExtractor
from step4_raft import Step4RAFTFlowEstimator
from affine.step5_affine import Step5AffineEstimator
from affine.step6_stabilize_affine import Step6AffineVideoStabilizer
from step7_composite import Step7VideoCompositor

from datetime import datetime
import contextlib
import io
import os
import sys
import time


# ==================================================
# 設定
# ==================================================

timestamp = datetime.now().strftime("%m%d_%H%M%S")

video_path = r"input_videos\test 30sec.mp4"
output_path = rf"results\{timestamp}_RAFT_Affine.mp4"

model_path = r"models\raft-things.pth"

# Step1：微細なカメラ揺れ検出設定
motion_threshold = 0.025
laplacian_threshold = 300.0
resize_scale = 0.5

os.makedirs("results", exist_ok=True)

total_start = time.time()


# ==================================================
# STEP1：ブレ特徴量抽出
# ==================================================

print("\n===== STEP1 ブレ特徴量抽出 =====")
step_start = time.time()

step1 = Step1MotionBlurDetector(
    motion_threshold=motion_threshold,
    laplacian_threshold=laplacian_threshold,
    resize_scale=resize_scale
)

frame_stats, camera_motion_frames = step1.run(
    video_path
)

step1_time = time.time() - step_start

print(f"\nSTEP1時間: {step1_time:.2f} 秒")
print(f"フレーム解析結果数: {len(frame_stats)}")
print(f"カメラ移動候補数: {len(camera_motion_frames)}")


# ==================================================
# STEP2：ブレ区間抽出
# ==================================================

print("\n===== STEP2 ブレ区間抽出 =====")
step_start = time.time()

step2 = Step2BlurSegmentExtractor()

segments, score = step2.run(
    frame_stats
)

step2_time = time.time() - step_start

print(f"STEP2時間: {step2_time:.2f} 秒")
print(f"検出区間数: {len(segments)}")

if len(segments) == 0:
    print("\n補正対象となる区間が検出されませんでした。")
    print("Step1またはStep2のしきい値を確認してください。")
    sys.exit(0)

print("検出区間一覧:")

for segment_id, (start, end) in enumerate(segments):
    print(
        f"  Segment {segment_id}: "
        f"{start} → {end} "
        f"({end - start + 1}フレーム)"
    )


# ==================================================
# STEP3：対象フレーム抽出
# ==================================================

print("\n===== STEP3 対象フレーム抽出 =====")
step_start = time.time()

step3 = Step3FrameExtractor()

# 長いフレーム単位ログを非表示
with contextlib.redirect_stdout(io.StringIO()):
    frames_dict = step3.extract_frames(
        video_path,
        segments
    )

step3_time = time.time() - step_start

print(f"STEP3時間: {step3_time:.2f} 秒")
print(f"抽出フレーム数: {len(frames_dict)}")

if len(frames_dict) < 2:
    print("\nRAFTに必要なフレーム数が不足しています。")
    sys.exit(0)


# ==================================================
# STEP4：RAFT Optical Flow
# ==================================================

print("\n===== STEP4 RAFT Optical Flow =====")
step_start = time.time()

step4 = Step4RAFTFlowEstimator(
    model_path=model_path
)

# Flowごとの長いログを非表示
with contextlib.redirect_stdout(io.StringIO()):
    flows_dict = step4.estimate_flows(
        frames_dict,
        segments
    )

step4_time = time.time() - step_start

print(f"STEP4時間: {step4_time:.2f} 秒")
print(f"RAFT Flow数: {len(flows_dict)}")

if len(flows_dict) == 0:
    print("\nOptical Flowが生成されませんでした。")
    sys.exit(0)


# ==================================================
# STEP5：Affine推定
# ==================================================

print("\n===== STEP5 Affine推定 =====")
step_start = time.time()

step5 = Step5AffineEstimator(
    grid_step=40,
    ransac_thresh=3.0,
    min_points=50
)

# Affineごとの長いログを非表示
with contextlib.redirect_stdout(io.StringIO()):
    affines_dict, inlier_counts = step5.estimate_affines(
        flows_dict
    )

step5_time = time.time() - step_start

print(f"STEP5時間: {step5_time:.2f} 秒")
print(f"Affine数: {len(affines_dict)}")

if len(inlier_counts) > 0:
    inlier_values = list(inlier_counts.values())

    average_inliers = (
        sum(inlier_values)
        / len(inlier_values)
    )

    print(f"平均inlier数: {average_inliers:.1f}")


# ==================================================
# STEP6：Affine安定化
# ==================================================

print("\n===== STEP6 Affine安定化 =====")
step_start = time.time()

step6 = Step6AffineVideoStabilizer()

# 補正フレームごとの長いログを非表示
with contextlib.redirect_stdout(io.StringIO()):
    stabilized_frames_dict = step6.stabilize_segments(
        frames_dict,
        segments,
        affines_dict
    )

step6_time = time.time() - step_start

print(f"STEP6時間: {step6_time:.2f} 秒")
print(f"安定化フレーム数: {len(stabilized_frames_dict)}")


# ==================================================
# STEP7：元動画へ合成・出力
# ==================================================

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

print(f"STEP7時間: {step7_time:.2f} 秒")


# ==================================================
# 処理結果
# ==================================================

print("\n===== Affine版 STEP1〜7 完了 =====")
print(f"出力成功: {success}")
print(f"出力動画: {output_path}")

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
print(f"総処理時間: {total_time:.2f} 秒")
print(f"総処理時間: {total_time / 60:.2f} 分")