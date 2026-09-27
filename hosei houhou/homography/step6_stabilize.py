import cv2
import numpy as np


class Step6VideoStabilizer:
    """
    Step6:
    Homographyを使ってブレ区間のフレームを安定化する
    """

    def __init__(self, border_mode="replicate"):
        if border_mode == "replicate":
            self.border_mode = cv2.BORDER_REPLICATE
        else:
            self.border_mode = cv2.BORDER_CONSTANT

    def stabilize_segments(
        self,
        frames_dict,
        segments,
        homographies_dict
    ):
        """
        Args:
            frames_dict:
                {frame_idx: frame}

            segments:
                [[start, end], ...]

            homographies_dict:
                {frame_idx: H from frame_idx to frame_idx+1}

        Returns:
            stabilized_frames_dict:
                {frame_idx: stabilized_frame}
        """

        stabilized_frames_dict = {}

        for seg_id, (start, end) in enumerate(segments):

            print(f"[Stabilize Segment {seg_id}] {start} → {end}")

            # 基準フレームはそのまま
            if start in frames_dict:
                stabilized_frames_dict[start] = frames_dict[start]

            # startから各フレームまでの累積Homography
            cumulative_H = np.eye(3, dtype=np.float32)

            for frame_idx in range(start + 1, end + 1):

                prev_idx = frame_idx - 1

                if prev_idx not in homographies_dict:
                    continue

                if frame_idx not in frames_dict:
                    continue

                H_prev_to_now = homographies_dict[prev_idx]

                # start → current への累積変換
                cumulative_H = H_prev_to_now @ cumulative_H

                # current を start 座標系へ戻す
                H_inv = np.linalg.inv(cumulative_H)

                frame = frames_dict[frame_idx]
                h, w = frame.shape[:2]

                stabilized = cv2.warpPerspective(
                    frame,
                    H_inv,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=self.border_mode
                )

                stabilized_frames_dict[frame_idx] = stabilized

                print(f"  stabilized frame {frame_idx}")

        print(f"安定化フレーム数: {len(stabilized_frames_dict)}")

        return stabilized_frames_dict