from step1 import Step1MotionBlurDetector
from step2 import Step2BlurSegmentExtractor
from step3 import Step3FrameExtractor
from step4_raft import Step4RAFTFlowEstimator
from step5_homography import Step5HomographyEstimator
from step6_stabilize import Step6VideoStabilizer
from step7_composite import Step7VideoCompositor
from datetime import datetime
import time


timestamp = datetime.now().strftime("%m%d_%H%M")

video_path = r"input_videos\test 30sec.mp4"
output_path = rf"results\{timestamp}_RAFT_Homography.mp4"

total_start = time.time()


print("\n===== STEP1 =====")
step_start = time.time()

step1 = Step1MotionBlurDetector(
    motion_threshold=0.025,
    laplacian_threshold=300.0,
    resize_scale=0.5
)

frame_stats, camera_motion_frames = step1.run(
    video_path
)

print(f"STEP1時間: {time.time() - step_start:.2f} 秒")
print(f"フレーム解析結果数: {len(frame_stats)}")
print(f"カメラ移動候補数: {len(camera_motion_frames)}")
print("\n===== STEP2 =====")
step_start = time.time()

step2 = Step2BlurSegmentExtractor()
segments, score = step2.run(frame_stats)

print(f"STEP2時間: {time.time() - step_start:.2f} 秒")

for s, e in segments:
    print(f"{s} → {e}")


print("\n===== STEP3 =====")
step_start = time.time()

step3 = Step3FrameExtractor()
frames_dict = step3.extract_frames(video_path, segments)

print(f"STEP3時間: {time.time() - step_start:.2f} 秒")


print("\n===== STEP4 RAFT =====")
step_start = time.time()

step4 = Step4RAFTFlowEstimator(
    model_path=r"models\raft-things.pth"
)

flows_dict = step4.estimate_flows(
    frames_dict,
    segments
)

print(f"STEP4時間: {time.time() - step_start:.2f} 秒")



print("\n===== STEP5 Homography =====")
step_start = time.time()

step5 = Step5HomographyEstimator(
    grid_step=40,
    ransac_thresh=3.0,
    min_points=50
)

homographies_dict, inlier_counts = step5.estimate_homographies(
    flows_dict
)

print(f"STEP5時間: {time.time() - step_start:.2f} 秒")


print("\n===== STEP6 Homography Stabilize =====")
step_start = time.time()

step6 = Step6VideoStabilizer()

stabilized_frames_dict = step6.stabilize_segments(
    frames_dict,
    segments,
    homographies_dict
)

print(f"STEP6時間: {time.time() - step_start:.2f} 秒")


print("\n===== STEP7 Composite =====")
step_start = time.time()

step7 = Step7VideoCompositor()

success = step7.composite_video(
    input_video_path=video_path,
    stabilized_frames_dict=stabilized_frames_dict,
    output_video_path=output_path
)

print(f"STEP7時間: {time.time() - step_start:.2f} 秒")


total_time = time.time() - total_start

print("\n===== Homography版 STEP1〜7 完了 =====")
print(f"出力成功: {success}")
print(f"出力動画: {output_path}")

print("\n====================")
print("処理時間まとめ")
print("====================")
print(f"総処理時間: {total_time:.2f} 秒")
print(f"総処理時間: {total_time / 60:.2f} 分")