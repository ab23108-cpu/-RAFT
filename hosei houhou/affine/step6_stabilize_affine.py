import cv2
import numpy as np


class Step6AffineVideoStabilizer:
    """
    Step6 Affine版:
    Affine変換を使ってブレ区間のフレームを安定化する
    """

    def __init__(self, border_mode="replicate"):
        if border_mode == "replicate":
            self.border_mode = cv2.BORDER_REPLICATE
        else:
            self.border_mode = cv2.BORDER_CONSTANT

    def affine_2x3_to_3x3(self, A):
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = A
        return H

    def stabilize_segments(
        self,
        frames_dict,
        segments,
        affines_dict
    ):
        stabilized_frames_dict = {}

        for seg_id, (start, end) in enumerate(segments):
            print(f"[Affine Stabilize Segment {seg_id}] {start} → {end}")

            if start in frames_dict:
                stabilized_frames_dict[start] = frames_dict[start]

            cumulative_H = np.eye(3, dtype=np.float32)

            for frame_idx in range(start + 1, end + 1):
                prev_idx = frame_idx - 1

                if prev_idx not in affines_dict:
                    continue

                if frame_idx not in frames_dict:
                    continue

                A_prev_to_now = affines_dict[prev_idx]
                H_prev_to_now = self.affine_2x3_to_3x3(A_prev_to_now)

                cumulative_H = H_prev_to_now @ cumulative_H

                H_inv = np.linalg.inv(cumulative_H)

                A_inv = H_inv[:2, :]

                frame = frames_dict[frame_idx]
                h, w = frame.shape[:2]

                stabilized = cv2.warpAffine(
                    frame,
                    A_inv,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=self.border_mode
                )

                stabilized_frames_dict[frame_idx] = stabilized

                print(f"  affine stabilized frame {frame_idx}")

        print(f"Affine安定化フレーム数: {len(stabilized_frames_dict)}")

        return stabilized_frames_dict