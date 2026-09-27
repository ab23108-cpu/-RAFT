from step1 import Step1MotionBlurDetector
from step2 import Step2BlurSegmentExtractor
from step3 import Step3FrameExtractor
from step4_raft import Step4RAFTFlowEstimator
from step5_homography import Step5HomographyEstimator
from step5_affine import Step5AffineEstimator
from step5_translation import Step5TranslationEstimator

import numpy as np
import csv
import os


video_path = r"input_videos\test 30sec.mp4"
output_csv = r"results\camera_motion_evaluation.csv"


def homography_to_tx_ty(H):
    tx = float(H[0, 2])
    ty = float(H[1, 2])
    return tx, ty


def affine_to_tx_ty(A):
    tx = float(A[0, 2])
    ty = float(A[1, 2])
    return tx, ty


def compute_motion_stats(motions):
    """
    motions:
        [{"frame_idx": int, "tx": float, "ty": float}, ...]
    """
    if len(motions) == 0:
        return {
            "count": 0,
            "avg_camera_motion": 0.0,
            "std_camera_motion": 0.0,
            "smoothness": 0.0
        }

    tx = np.array([m["tx"] for m in motions])
    ty = np.array([m["ty"] for m in motions])

    mag = np.sqrt(tx ** 2 + ty ** 2)

    avg_camera_motion = float(np.mean(mag))
    std_camera_motion = float(np.std(mag))

    if len(mag) > 2:
        smoothness = float(np.std(np.diff(mag)))
    else:
        smoothness = 0.0

    return {
        "count": len(motions),
        "avg_camera_motion": avg_camera_motion,
        "std_camera_motion": std_camera_motion,
        "smoothness": smoothness
    }


print("\n===== STEP1 =====")
step1 = Step1MotionBlurDetector()
frame_stats, blur_frames = step1.run(video_path)


print("\n===== STEP2 =====")
step2 = Step2BlurSegmentExtractor()
segments, score = step2.run(frame_stats)

print("検出区間:")
for s, e in segments:
    print(f"{s} → {e}")


print("\n===== STEP3 =====")
step3 = Step3FrameExtractor()
frames_dict = step3.extract_frames(video_path, segments)


print("\n===== STEP4 RAFT =====")
step4 = Step4RAFTFlowEstimator(
    model_path=r"models\raft-things.pth"
)

flows_dict = step4.estimate_flows(
    frames_dict,
    segments
)


# =========================
# Homography
# =========================
print("\n===== Homography Camera Motion =====")
step5_h = Step5HomographyEstimator(
    grid_step=40,
    ransac_thresh=3.0,
    min_points=50
)

homographies_dict, h_inliers = step5_h.estimate_homographies(flows_dict)

h_motions = []

for frame_idx, H in homographies_dict.items():
    tx, ty = homography_to_tx_ty(H)
    h_motions.append({
        "method": "RAFT_Homography",
        "frame_idx": frame_idx,
        "tx": tx,
        "ty": ty,
        "motion": float(np.sqrt(tx ** 2 + ty ** 2))
    })


# =========================
# Affine
# =========================
print("\n===== Affine Camera Motion =====")
step5_a = Step5AffineEstimator(
    grid_step=40,
    ransac_thresh=3.0,
    min_points=50
)

affines_dict, a_inliers = step5_a.estimate_affines(flows_dict)

a_motions = []

for frame_idx, A in affines_dict.items():
    tx, ty = affine_to_tx_ty(A)
    a_motions.append({
        "method": "RAFT_Affine",
        "frame_idx": frame_idx,
        "tx": tx,
        "ty": ty,
        "motion": float(np.sqrt(tx ** 2 + ty ** 2))
    })


# =========================
# Translation
# =========================
print("\n===== Translation Camera Motion =====")

step5_t = Step5TranslationEstimator(
    grid_step=40,
    mad_threshold=3.0,
    border_ratio=0.05,
    min_points=30,
    green_lower=(30, 40, 40),
    green_upper=(90, 255, 255),
    use_grass_mask=True
)

translations_dict = step5_t.estimate_translations(
    flows_dict,
    frames_dict=frames_dict
)

translations_dict = step5_t.estimate_translations(flows_dict)

t_motions = []

for frame_idx, T in translations_dict.items():
    tx = float(T["tx"])
    ty = float(T["ty"])

    t_motions.append({
        "method": "RAFT_Translation",
        "frame_idx": frame_idx,
        "tx": tx,
        "ty": ty,
        "motion": float(np.sqrt(tx ** 2 + ty ** 2))
    })


# =========================
# Summary
# =========================
summary = []

for method_name, motions in [
    ("RAFT_Homography", h_motions),
    ("RAFT_Affine", a_motions),
    ("RAFT_Translation", t_motions)
]:
    stats = compute_motion_stats(motions)

    summary.append({
        "method": method_name,
        "count": stats["count"],
        "avg_camera_motion": stats["avg_camera_motion"],
        "std_camera_motion": stats["std_camera_motion"],
        "smoothness": stats["smoothness"]
    })


os.makedirs(os.path.dirname(output_csv), exist_ok=True)

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "method",
        "count",
        "avg_camera_motion",
        "std_camera_motion",
        "smoothness"
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary)


print("\n===== カメラ揺れ評価結果 =====")

for row in summary:
    print(f"\n[{row['method']}]")
    print(f"評価フレーム間数: {row['count']}")
    print(f"平均カメラ移動量: {row['avg_camera_motion']:.4f}")
    print(f"カメラ移動量標準偏差: {row['std_camera_motion']:.4f}")
    print(f"カメラ軌跡変動量: {row['smoothness']:.4f}")

print(f"\nCSV保存: {output_csv}")