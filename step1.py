import cv2
import numpy as np


class Step1MotionBlurDetector:
    """
    Step1:
    Farneback Optical Flowでフレーム間の動きを推定し、
    ブレ区間検出用の frame_stats を作成する。

    改善点:
    - 芝マスクを使い、背景寄りの領域だけを見る
    - 平均値ではなく中央値を使い、選手・ボールの局所運動に強くする
    - MAD外れ値除去で極端な動きを除外する
    """

    def __init__(
        self,
        resize_width=640,
        use_grass_mask=True,
        use_median=True,
        green_lower=(30, 40, 40),
        green_upper=(90, 255, 255),
        min_mask_pixels=500,
        mad_threshold=3.0
    ):
        self.resize_width = resize_width
        self.use_grass_mask = use_grass_mask
        self.use_median = use_median
        self.green_lower = np.array(green_lower, dtype=np.uint8)
        self.green_upper = np.array(green_upper, dtype=np.uint8)
        self.min_mask_pixels = min_mask_pixels
        self.mad_threshold = mad_threshold

    def resize_keep_aspect(self, frame):
        h, w = frame.shape[:2]

        if self.resize_width is None or w <= self.resize_width:
            return frame

        scale = self.resize_width / w
        new_h = int(h * scale)

        return cv2.resize(frame, (self.resize_width, new_h))

    def create_grass_mask(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            self.green_lower,
            self.green_upper
        )

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # 選手の足元や境界の影響を少し減らす
        mask = cv2.erode(
            mask,
            kernel,
            iterations=1
        )

        return mask > 0

    def remove_flow_outliers(self, dx, dy):
        mag = np.sqrt(dx ** 2 + dy ** 2)

        med = np.median(mag)
        mad = np.median(np.abs(mag - med)) + 1e-8

        keep = np.abs(mag - med) < self.mad_threshold * mad

        if np.count_nonzero(keep) < 30:
            return dx, dy

        return dx[keep], dy[keep]

    def estimate_global_motion(self, flow, frame_bgr):
        dx = flow[..., 0]
        dy = flow[..., 1]

        if self.use_grass_mask:
            mask = self.create_grass_mask(frame_bgr)

            if np.count_nonzero(mask) >= self.min_mask_pixels:
                dx_valid = dx[mask]
                dy_valid = dy[mask]
            else:
                dx_valid = dx.reshape(-1)
                dy_valid = dy.reshape(-1)
        else:
            dx_valid = dx.reshape(-1)
            dy_valid = dy.reshape(-1)

        dx_valid, dy_valid = self.remove_flow_outliers(
            dx_valid,
            dy_valid
        )

        if self.use_median:
            global_dx = float(np.median(dx_valid))
            global_dy = float(np.median(dy_valid))
        else:
            global_dx = float(np.mean(dx_valid))
            global_dy = float(np.mean(dy_valid))

        motion_magnitude = float(
            np.sqrt(global_dx ** 2 + global_dy ** 2)
        )

        return global_dx, global_dy, motion_magnitude, len(dx_valid)

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")

        frame_stats = []

        ret, prev_frame = cap.read()

        if not ret:
            cap.release()
            raise RuntimeError("最初のフレームを読み込めませんでした")

        prev_frame = self.resize_keep_aspect(prev_frame)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_idx = 1

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = self.resize_keep_aspect(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0
            )

            global_dx, global_dy, motion_magnitude, valid_points = (
                self.estimate_global_motion(flow, frame)
            )

            laplacian_variance = float(
                cv2.Laplacian(gray, cv2.CV_64F).var()
            )

            frame_stats.append(
                {
                    "frame_idx": frame_idx,
                    "global_dx": global_dx,
                    "global_dy": global_dy,
                    "motion_magnitude": motion_magnitude,
                    "laplacian_variance": laplacian_variance,
                    "valid_points": valid_points
                }
            )

            prev_gray = gray
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"STEP1解析中: {frame_idx} frames")

        cap.release()

        print("===== STEP1 完了 =====")
        print(f"解析フレーム数: {len(frame_stats)}")
        print(f"芝マスク使用: {self.use_grass_mask}")
        print(f"代表値: {'中央値' if self.use_median else '平均値'}")

        return frame_stats