import numpy as np
import cv2
from scipy.signal import medfilt


class Step2BlurSegmentExtractor:
    """
    Step2:
    frame_stats -> blur segments に変換する。
    教師なしで、カメラ移動の高周波成分をブレ候補として使う。
    """

    def __init__(
        self,
        score_weight_motion=0.9,
        score_weight_laplacian=0.1,
        threshold=0.35,
        smoothing_kernel=11,
        min_segment_length=8,
        merge_gap=10
    ):
        self.w_m = float(score_weight_motion)
        self.w_l = float(score_weight_laplacian)
        self.threshold = float(threshold)
        self.smoothing_kernel = int(smoothing_kernel)
        self.min_segment_length = int(min_segment_length)
        self.merge_gap = int(merge_gap)

    def robust_normalize(self, values):
        values = np.asarray(values, dtype=np.float64)

        if len(values) == 0:
            return values

        lo = np.percentile(values, 5)
        hi = np.percentile(values, 95)

        return np.clip(
            (values - lo) / (hi - lo + 1e-8),
            0.0,
            1.0
        )

    def _valid_kernel_size(self, length):
        if length < 3:
            return 1

        k = self.smoothing_kernel

        if k % 2 == 0:
            k += 1

        if k > length:
            k = length if length % 2 == 1 else length - 1

        if k < 3:
            return 1

        return k

    def compute_score(self, frame_stats):
        dx = np.array(
            [f.get("global_dx", 0.0) for f in frame_stats],
            dtype=np.float64
        )

        dy = np.array(
            [f.get("global_dy", 0.0) for f in frame_stats],
            dtype=np.float64
        )

        lap = np.array(
            [f.get("laplacian_variance", 0.0) for f in frame_stats],
            dtype=np.float64
        )

        k = self._valid_kernel_size(len(dx))

        if k > 1:
            dx_smooth = medfilt(dx, k)
            dy_smooth = medfilt(dy, k)
        else:
            dx_smooth = dx.copy()
            dy_smooth = dy.copy()

        residual_dx = dx - dx_smooth
        residual_dy = dy - dy_smooth

        shake = np.sqrt(
            residual_dx ** 2
            + residual_dy ** 2
        )

        shake_n = self.robust_normalize(shake)

        lap_n = 1.0 - self.robust_normalize(lap)

        score = (
            self.w_m * shake_n
            + self.w_l * lap_n
        )

        return score

    def smooth(self, score):
        score = np.asarray(score, dtype=np.float64)

        k = self._valid_kernel_size(len(score))

        if k > 1:
            return medfilt(score, k)

        return score

    def to_segments(self, score):
        is_blur = score > self.threshold

        segments = []
        start = None

        for i, v in enumerate(is_blur):
            if v and start is None:
                start = i

            elif not v and start is not None:
                segments.append([start, i - 1])
                start = None

        if start is not None:
            segments.append([start, len(score) - 1])

        return segments

    def filter_short(self, segments):
        return [
            s for s in segments
            if (s[1] - s[0] + 1) >= self.min_segment_length
        ]

    def merge_segments(self, segments):
        if not segments:
            return []

        merged = [segments[0]]

        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]

            if start - prev_end <= self.merge_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])

        return merged

    def run(self, frame_stats):
        score = self.compute_score(frame_stats)
        score = self.smooth(score)

        segments = self.to_segments(score)
        segments = self.merge_segments(segments)
        segments = self.filter_short(segments)

        return segments, score
    

    