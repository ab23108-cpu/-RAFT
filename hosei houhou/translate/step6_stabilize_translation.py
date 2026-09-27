import cv2
import numpy as np
from scipy.signal import medfilt


class Step6TranslationVideoStabilizer:
    """
    Step6:
    Step5で推定したtranslationを使って映像を安定化する。

    改善版:
    - 累積translationをそのまま打ち消さない
    - 累積trajectoryを平滑化する
    - 元trajectory - 平滑trajectory = shake residual
    - shake residualだけを逆方向に補正する

    これにより、サッカー中継の自然なパン・チルトを残し、
    カメラ固有の細かい揺れだけを抑える。
    """

    def __init__(
    self,
    correction_strength=0.3,
    smoothing_kernel=61,
    border_mode="reflect_101",
    zoom_scale=1.0,
    border_value=(0, 0, 0),
    max_correction_px=0.5
):
        if not 0.0 <= correction_strength <= 1.0:
            raise ValueError(
                "correction_strength must be between 0.0 and 1.0"
            )

        if smoothing_kernel < 3:
            raise ValueError(
                "smoothing_kernel must be >= 3"
            )

        if zoom_scale < 1.0:
            raise ValueError(
                "zoom_scale must be >= 1.0"
            )

        self.correction_strength = float(correction_strength)
        self.smoothing_kernel = int(smoothing_kernel)
        self.border_mode_name = border_mode
        self.zoom_scale = float(zoom_scale)
        self.border_value = border_value
        self.max_correction_px = float(max_correction_px)

        if border_mode == "replicate":
            self.border_mode = cv2.BORDER_REPLICATE

        elif border_mode == "reflect":
            self.border_mode = cv2.BORDER_REFLECT

        elif border_mode == "reflect_101":
            self.border_mode = cv2.BORDER_REFLECT_101

        elif border_mode == "constant":
            self.border_mode = cv2.BORDER_CONSTANT

        else:
            raise ValueError(
                "border_mode must be one of: "
                "replicate, reflect, reflect_101, constant"
            )

    def _valid_kernel_size(self, length):
        """
        medfilt用に、系列長以下の奇数kernelを作る。
        """

        if length < 3:
            return 1

        k = self.smoothing_kernel

        if k % 2 == 0:
            k += 1

        if k > length:
            k = length if length % 2 == 1 else length - 1

        if k < 3:
            k = 3 if length >= 3 else 1

        return k

    def smooth_trajectory(self, trajectory):
        """
        累積軌跡を時間方向に平滑化する。
        """

        trajectory = np.asarray(
            trajectory,
            dtype=np.float64
        )

        if len(trajectory) < 3:
            return trajectory.copy()

        k = self._valid_kernel_size(len(trajectory))

        if k <= 1:
            return trajectory.copy()

        return medfilt(
            trajectory,
            kernel_size=k
        )

    def apply_zoom(self, frame):
        """
        補正後の端領域を目立ちにくくするため中央基準で拡大する。
        """

        if self.zoom_scale == 1.0:
            return frame

        height, width = frame.shape[:2]

        center_x = width / 2.0
        center_y = height / 2.0

        zoom_matrix = cv2.getRotationMatrix2D(
            (center_x, center_y),
            0,
            self.zoom_scale
        )

        zoomed_frame = cv2.warpAffine(
            frame,
            zoom_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
            borderValue=self.border_value
        )

        return zoomed_frame

    def stabilize_one_frame(
        self,
        frame,
        correction_tx,
        correction_ty
    ):
        """
        1フレームにtranslation補正を適用する。
        """

        height, width = frame.shape[:2]

        transform_matrix = np.array(
            [
                [1.0, 0.0, correction_tx],
                [0.0, 1.0, correction_ty]
            ],
            dtype=np.float32
        )

        stabilized_frame = cv2.warpAffine(
            frame,
            transform_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
            borderValue=self.border_value
        )

        stabilized_frame = self.apply_zoom(
            stabilized_frame
        )

        return stabilized_frame

    def build_segment_trajectory(
        self,
        start,
        end,
        translations_dict
    ):
        """
        segment内のtranslation列と累積trajectoryを作る。

        frame start を基準0とし、
        frame start+1 以降に対してtrajectoryを持つ。

        Returns:
            frame_indices:
                補正対象フレーム番号

            traj_x, traj_y:
                各フレームの累積移動量

            valid_mask:
                translationが存在したかどうか
        """

        frame_indices = list(
            range(start, end + 1)
        )

        traj_x = [0.0]
        traj_y = [0.0]
        valid_mask = [True]

        cumulative_x = 0.0
        cumulative_y = 0.0

        for frame_idx in range(start + 1, end + 1):
            previous_idx = frame_idx - 1

            if previous_idx in translations_dict:
                translation = translations_dict[previous_idx]

                tx = float(translation.get("tx", 0.0))
                ty = float(translation.get("ty", 0.0))

                cumulative_x += tx
                cumulative_y += ty

                valid_mask.append(True)

            else:
                valid_mask.append(False)

            traj_x.append(cumulative_x)
            traj_y.append(cumulative_y)

        return (
            frame_indices,
            np.asarray(traj_x, dtype=np.float64),
            np.asarray(traj_y, dtype=np.float64),
            np.asarray(valid_mask, dtype=bool)
        )

    def compute_corrections_for_segment(
        self,
        start,
        end,
        translations_dict
    ):
        """
        segment内の各フレームに対する補正量を計算する。

        correction = -(trajectory - smooth_trajectory)
        """

        (
            frame_indices,
            traj_x,
            traj_y,
            valid_mask
        ) = self.build_segment_trajectory(
            start,
            end,
            translations_dict
        )

        smooth_x = self.smooth_trajectory(traj_x)
        smooth_y = self.smooth_trajectory(traj_y)

        shake_x = traj_x - smooth_x
        shake_y = traj_y - smooth_y

        correction_x = (
            -shake_x
            * self.correction_strength
        )

        correction_y = (
            -shake_y
            * self.correction_strength
        )

        correction_x = np.clip(
            correction_x,
            -self.max_correction_px,
            self.max_correction_px
        )

        correction_y = np.clip(
            correction_y,
            -self.max_correction_px,
            self.max_correction_px
        )

        corrections = {}

        for i, frame_idx in enumerate(frame_indices):
            corrections[frame_idx] = {
                "correction_tx": float(correction_x[i]),
                "correction_ty": float(correction_y[i]),
                "trajectory_x": float(traj_x[i]),
                "trajectory_y": float(traj_y[i]),
                "smooth_x": float(smooth_x[i]),
                "smooth_y": float(smooth_y[i]),
                "shake_x": float(shake_x[i]),
                "shake_y": float(shake_y[i]),
                "valid": bool(valid_mask[i])
            }

        return corrections

    def stabilize_segments(
        self,
        frames_dict,
        segments,
        translations_dict
    ):
        """
        ブレ区間ごとに映像を安定化する。

        Args:
            frames_dict:
                {
                    frame_idx: frame
                }

            segments:
                [
                    [start, end],
                    ...
                ]

            translations_dict:
                {
                    frame_idx: {
                        "tx": float,
                        "ty": float
                    }
                }

                frame_idx -> frame_idx + 1 のtranslation

        Returns:
            stabilized_frames_dict:
                {
                    frame_idx: stabilized_frame
                }
        """

        stabilized_frames_dict = {}

        print("===== Step6 Translation Stabilization =====")
        print(f"correction_strength: {self.correction_strength}")
        print(f"smoothing_kernel: {self.smoothing_kernel}")
        print(f"border_mode: {self.border_mode_name}")
        print(f"zoom_scale: {self.zoom_scale}")
        print(f"max_correction_px: {self.max_correction_px}")

        for segment_id, (start, end) in enumerate(segments):
            print(
                f"[Translation Segment {segment_id}] "
                f"{start} -> {end}"
            )

            corrections = self.compute_corrections_for_segment(
                start,
                end,
                translations_dict
            )

            for frame_idx in range(start, end + 1):
                if frame_idx not in frames_dict:
                    print(
                        f"  skip: frame {frame_idx} not found"
                    )
                    continue

                frame = frames_dict[frame_idx]

                correction = corrections[frame_idx]

                correction_tx = correction["correction_tx"]
                correction_ty = correction["correction_ty"]

                stabilized_frame = self.stabilize_one_frame(
                    frame,
                    correction_tx,
                    correction_ty
                )

                stabilized_frames_dict[frame_idx] = stabilized_frame

            max_shake = 0.0

            if corrections:
                shake_values = [
                    np.sqrt(
                        c["shake_x"] ** 2
                        + c["shake_y"] ** 2
                    )
                    for c in corrections.values()
                ]

                max_shake = float(np.max(shake_values))

            print(
                f"  stabilized frames: {end - start + 1}, "
                f"max shake residual: {max_shake:.4f}"
            )

        print(
            "Translation stabilization frames: "
            f"{len(stabilized_frames_dict)}"
        )

        return stabilized_frames_dict
