import os

import cv2
import numpy as np
import torch
import torch.nn as nn


class FlowTranslationCNN(nn.Module):
    """
    RAFT flowからカメラの平行移動量 tx, ty を推定するCNN。

    入力:
        [B, 3, H, W]
        channel 0: flow_x
        channel 1: flow_y
        channel 2: background mask

    出力:
        [B, 2] = tx, ty
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


class Step5TranslationCNNEstimator:
    """
    STEP5 CNN版。

    model_pathに学習済み重みを指定した場合:
        CNNで tx, ty を推定する。

    model_pathがない場合:
        未学習CNNは使わず、中央値 + MADの従来推定にフォールバックする。
        これにより、未学習CNNによるランダム補正を防ぐ。
    """

    def __init__(
        self,
        model_path=None,
        input_size=(128, 224),
        grid_step=20,
        mad_threshold=3.0,
        border_ratio=0.05,
        min_points=30,
        green_lower=(30, 40, 40),
        green_upper=(90, 255, 255),
        use_grass_mask=True,
        use_median=True,
        max_abs_translation=5.0,
        cnn_weight=1.0,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.grid_step = grid_step
        self.mad_threshold = mad_threshold
        self.border_ratio = border_ratio
        self.min_points = min_points
        self.green_lower = np.array(green_lower, dtype=np.uint8)
        self.green_upper = np.array(green_upper, dtype=np.uint8)
        self.use_grass_mask = use_grass_mask
        self.use_median = use_median
        self.max_abs_translation = float(max_abs_translation)
        self.cnn_weight = float(cnn_weight)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FlowTranslationCNN().to(self.device)
        self.model.eval()
        self.cnn_available = False

        if model_path is not None and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            self.model.load_state_dict(checkpoint)
            self.cnn_available = True
            print(f"Step5 CNN model loaded: {model_path}")
        else:
            print("Step5 CNN model is not loaded. Use robust median fallback.")

        print(f"Step5 device: {self.device}")

    def create_grass_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        kernel = np.ones((5, 5), np.uint8)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)
        grass_mask = cv2.erode(grass_mask, kernel, iterations=1)
        return grass_mask

    def create_valid_area_mask(self, height, width):
        margin_x = int(width * self.border_ratio)
        margin_y = int(height * self.border_ratio)
        valid_area = np.zeros((height, width), dtype=np.uint8)
        valid_area[margin_y:height - margin_y, margin_x:width - margin_x] = 255
        return valid_area

    def create_background_mask(self, flow, frame=None):
        height, width = flow.shape[:2]
        valid_area = self.create_valid_area_mask(height, width)

        if self.use_grass_mask and frame is not None:
            grass_mask = self.create_grass_mask(frame)
            if grass_mask.shape[:2] != (height, width):
                grass_mask = cv2.resize(
                    grass_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            return cv2.bitwise_and(grass_mask, valid_area)

        return valid_area

    def sample_background_flow(self, flow, frame=None):
        height, width = flow.shape[:2]
        background_mask = self.create_background_mask(flow, frame)

        ys, xs = np.mgrid[0:height:self.grid_step, 0:width:self.grid_step]
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        valid = background_mask[ys, xs] > 0
        xs = xs[valid]
        ys = ys[valid]

        if len(xs) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0

        flow_x = flow[ys, xs, 0].astype(np.float64)
        flow_y = flow[ys, xs, 1].astype(np.float64)
        finite = np.isfinite(flow_x) & np.isfinite(flow_y)

        return flow_x[finite], flow_y[finite], int(np.count_nonzero(finite))

    @staticmethod
    def median_absolute_deviation(values, median_value):
        return np.median(np.abs(values - median_value))

    def robust_median_translation(self, flow_x, flow_y):
        if len(flow_x) < self.min_points:
            return 0.0, 0.0, 0

        median_x = np.median(flow_x)
        median_y = np.median(flow_y)

        mad_x = self.median_absolute_deviation(flow_x, median_x)
        mad_y = self.median_absolute_deviation(flow_y, median_y)

        scale_x = max(1.4826 * mad_x, 1e-6)
        scale_y = max(1.4826 * mad_y, 1e-6)

        inlier_mask = (
            np.abs(flow_x - median_x) <= self.mad_threshold * scale_x
        ) & (
            np.abs(flow_y - median_y) <= self.mad_threshold * scale_y
        )

        inlier_x = flow_x[inlier_mask]
        inlier_y = flow_y[inlier_mask]

        if len(inlier_x) < self.min_points:
            return float(median_x), float(median_y), len(flow_x)

        return float(np.median(inlier_x)), float(np.median(inlier_y)), len(inlier_x)

    def flow_to_cnn_input(self, flow, frame=None):
        input_h, input_w = self.input_size
        background_mask = self.create_background_mask(flow, frame).astype(np.float32) / 255.0

        flow_x = flow[:, :, 0].astype(np.float32)
        flow_y = flow[:, :, 1].astype(np.float32)
        mag = np.sqrt(flow_x ** 2 + flow_y ** 2)

        valid = background_mask > 0
        if np.count_nonzero(valid) >= self.min_points:
            scale = np.percentile(mag[valid], 95)
        else:
            scale = np.percentile(mag, 95)

        scale = max(float(scale), 1e-6)
        flow_x = np.clip(flow_x / scale, -3.0, 3.0) / 3.0
        flow_y = np.clip(flow_y / scale, -3.0, 3.0) / 3.0

        flow_x = cv2.resize(flow_x, (input_w, input_h), interpolation=cv2.INTER_AREA)
        flow_y = cv2.resize(flow_y, (input_w, input_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(background_mask, (input_w, input_h), interpolation=cv2.INTER_NEAREST)

        x = np.stack([flow_x, flow_y, mask], axis=0).astype(np.float32)
        return torch.from_numpy(x)[None].to(self.device), scale

    @torch.no_grad()
    def estimate_translation_by_cnn(self, flow, frame=None):
        x, scale = self.flow_to_cnn_input(flow, frame)
        pred = self.model(x)[0].detach().cpu().numpy().astype(np.float64)

        tx = float(pred[0] * scale)
        ty = float(pred[1] * scale)

        tx = float(np.clip(tx, -self.max_abs_translation, self.max_abs_translation))
        ty = float(np.clip(ty, -self.max_abs_translation, self.max_abs_translation))

        return tx, ty

    def estimate_translation_from_flow(self, flow, frame=None):
        flow_x, flow_y, sampled_points = self.sample_background_flow(flow, frame)
        robust_tx, robust_ty, used_points = self.robust_median_translation(flow_x, flow_y)

        if self.cnn_available:
            cnn_tx, cnn_ty = self.estimate_translation_by_cnn(flow, frame)
            tx = self.cnn_weight * cnn_tx + (1.0 - self.cnn_weight) * robust_tx
            ty = self.cnn_weight * cnn_ty + (1.0 - self.cnn_weight) * robust_ty
            method = "cnn"
        else:
            tx = robust_tx
            ty = robust_ty
            method = "robust_fallback"

        return {
            "tx": float(tx),
            "ty": float(ty),
            "motion_magnitude": float(np.sqrt(tx ** 2 + ty ** 2)),
            "background_points": int(used_points),
            "sampled_points": int(sampled_points),
            "grass_mask_used": self.use_grass_mask and frame is not None,
            "estimation_method": method,
            "cnn_available": self.cnn_available,
        }

    def estimate_translations(self, flows_dict, frames_dict=None):
        translations_dict = {}
        background_point_counts = []
        cnn_count = 0

        for frame_idx, flow in flows_dict.items():
            frame = frames_dict.get(frame_idx, None) if frames_dict is not None else None
            result = self.estimate_translation_from_flow(flow, frame)
            translations_dict[frame_idx] = result
            background_point_counts.append(result["background_points"])

            if result["estimation_method"] == "cnn":
                cnn_count += 1

        avg_background_points = (
            float(np.mean(background_point_counts))
            if len(background_point_counts) > 0
            else 0.0
        )

        print(f"Step5 CNN Translation推定完了: {len(translations_dict)} pairs")
        print(f"CNN使用数: {cnn_count}")
        print(f"平均使用背景点数: {avg_background_points:.1f}")

        return translations_dict


Step5TranslationEstimator = Step5TranslationCNNEstimator