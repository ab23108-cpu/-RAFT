import cv2
import numpy as np


class Step5TranslationEstimator:
    """
    Step5:
    RAFT Optical Flowから、フレーム間のTranslation量 tx, ty を推定する。

    改善点:
    - 芝生色マスクにより背景候補を抽出する
    - person_detectorを渡した場合、YOLOで検出した人物領域を除外する
    - 画面端を除外する
    - 背景候補のFlowから中央値を取り、MADで外れ値を除外する
    """

    def __init__(
        self,
        grid_step=20,
        mad_threshold=3.0,
        border_ratio=0.05,
        min_points=30,
        green_lower=(30, 40, 40),
        green_upper=(90, 255, 255),
        use_grass_mask=True,
        use_person_mask=False,
        person_detector=None,
        use_median=True
    ):
        self.grid_step = grid_step
        self.mad_threshold = mad_threshold
        self.border_ratio = border_ratio
        self.min_points = min_points

        self.green_lower = np.array(green_lower, dtype=np.uint8)
        self.green_upper = np.array(green_upper, dtype=np.uint8)

        self.use_grass_mask = use_grass_mask
        self.use_person_mask = use_person_mask
        self.person_detector = person_detector
        self.use_median = use_median

    def create_grass_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        grass_mask = cv2.inRange(
            hsv,
            self.green_lower,
            self.green_upper
        )

        kernel = np.ones((5, 5), np.uint8)

        grass_mask = cv2.morphologyEx(
            grass_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        grass_mask = cv2.morphologyEx(
            grass_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # 選手の足元や境界の影響を少し減らす
        grass_mask = cv2.erode(
            grass_mask,
            kernel,
            iterations=1
        )

        return grass_mask

    def create_valid_area_mask(self, height, width):
        margin_x = int(width * self.border_ratio)
        margin_y = int(height * self.border_ratio)

        valid_area = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        valid_area[
            margin_y:height - margin_y,
            margin_x:width - margin_x
        ] = 255

        return valid_area

    def create_person_background_mask(self, frame, height, width):
        """
        Returns:
            background_mask:
                255 = usable non-person area
                0 = person area
            person_count:
                number of detected person boxes
        """

        if not self.use_person_mask or self.person_detector is None:
            return np.full((height, width), 255, dtype=np.uint8), 0

        person_mask, boxes = self.person_detector.create_person_mask(frame)

        if person_mask.shape[:2] != (height, width):
            person_mask = cv2.resize(
                person_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        non_person_mask = cv2.bitwise_not(person_mask)

        return non_person_mask, len(boxes)

    def create_background_mask(self, flow, frame=None):
        height, width = flow.shape[:2]

        valid_area = self.create_valid_area_mask(
            height,
            width
        )

        background_mask = valid_area
        person_count = 0

        if self.use_grass_mask and frame is not None:
            grass_mask = self.create_grass_mask(frame)

            if grass_mask.shape[:2] != (height, width):
                grass_mask = cv2.resize(
                    grass_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            background_mask = cv2.bitwise_and(
                background_mask,
                grass_mask
            )

        if self.use_person_mask and self.person_detector is not None and frame is not None:
            non_person_mask, person_count = self.create_person_background_mask(
                frame,
                height,
                width
            )

            background_mask = cv2.bitwise_and(
                background_mask,
                non_person_mask
            )

        return background_mask, person_count

    def sample_background_flow(
        self,
        flow,
        frame=None
    ):
        height, width = flow.shape[:2]

        background_mask, person_count = self.create_background_mask(
            flow,
            frame
        )

        ys, xs = np.mgrid[
            0:height:self.grid_step,
            0:width:self.grid_step
        ]

        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        mask_values = background_mask[ys, xs]
        valid = mask_values > 0

        xs = xs[valid]
        ys = ys[valid]

        if len(xs) == 0:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                0,
                person_count
            )

        flow_x = flow[ys, xs, 0].astype(np.float64)
        flow_y = flow[ys, xs, 1].astype(np.float64)

        finite = (
            np.isfinite(flow_x)
            & np.isfinite(flow_y)
        )

        flow_x = flow_x[finite]
        flow_y = flow_y[finite]

        return flow_x, flow_y, len(flow_x), person_count

    @staticmethod
    def median_absolute_deviation(values, median_value):
        return np.median(
            np.abs(values - median_value)
        )

    def robust_median_translation(
        self,
        flow_x,
        flow_y
    ):
        if len(flow_x) < self.min_points:
            return 0.0, 0.0, 0

        median_x = np.median(flow_x)
        median_y = np.median(flow_y)

        mad_x = self.median_absolute_deviation(
            flow_x,
            median_x
        )

        mad_y = self.median_absolute_deviation(
            flow_y,
            median_y
        )

        scale_x = max(
            1.4826 * mad_x,
            1e-6
        )

        scale_y = max(
            1.4826 * mad_y,
            1e-6
        )

        inlier_mask = (
            np.abs(flow_x - median_x)
            <= self.mad_threshold * scale_x
        ) & (
            np.abs(flow_y - median_y)
            <= self.mad_threshold * scale_y
        )

        inlier_x = flow_x[inlier_mask]
        inlier_y = flow_y[inlier_mask]

        if len(inlier_x) < self.min_points:
            tx = float(median_x)
            ty = float(median_y)
            used_points = len(flow_x)
        else:
            if self.use_median:
                tx = float(np.median(inlier_x))
                ty = float(np.median(inlier_y))
            else:
                tx = float(np.mean(inlier_x))
                ty = float(np.mean(inlier_y))

            used_points = len(inlier_x)

        return tx, ty, used_points

    def estimate_translation_from_flow(
        self,
        flow,
        frame=None
    ):
        (
            flow_x,
            flow_y,
            sampled_points,
            person_count
        ) = self.sample_background_flow(
            flow,
            frame
        )

        if len(flow_x) < self.min_points:
            return {
                "tx": 0.0,
                "ty": 0.0,
                "motion_magnitude": 0.0,
                "background_points": 0,
                "sampled_points": sampled_points,
                "grass_mask_used": (
                    self.use_grass_mask
                    and frame is not None
                ),
                "person_mask_used": (
                    self.use_person_mask
                    and self.person_detector is not None
                    and frame is not None
                ),
                "person_count": person_count
            }

        tx, ty, used_points = (
            self.robust_median_translation(
                flow_x,
                flow_y
            )
        )

        motion_magnitude = float(
            np.sqrt(tx ** 2 + ty ** 2)
        )

        return {
            "tx": tx,
            "ty": ty,
            "motion_magnitude": motion_magnitude,
            "background_points": used_points,
            "sampled_points": sampled_points,
            "grass_mask_used": (
                self.use_grass_mask
                and frame is not None
            ),
            "person_mask_used": (
                self.use_person_mask
                and self.person_detector is not None
                and frame is not None
            ),
            "person_count": person_count
        }

    def estimate_translations(
        self,
        flows_dict,
        frames_dict=None
    ):
        translations_dict = {}

        background_point_counts = []
        person_counts = []

        for frame_idx, flow in flows_dict.items():
            frame = None

            if frames_dict is not None:
                frame = frames_dict.get(
                    frame_idx,
                    None
                )

            result = self.estimate_translation_from_flow(
                flow,
                frame
            )

            translations_dict[frame_idx] = result

            background_point_counts.append(
                result["background_points"]
            )

            person_counts.append(
                result["person_count"]
            )

        if len(background_point_counts) > 0:
            avg_background_points = float(
                np.mean(background_point_counts)
            )
        else:
            avg_background_points = 0.0

        if len(person_counts) > 0:
            avg_person_count = float(np.mean(person_counts))
        else:
            avg_person_count = 0.0

        print(
            "人物除外つきTranslation推定完了: "
            f"{len(translations_dict)} pairs"
        )

        print(
            "平均使用背景点数: "
            f"{avg_background_points:.1f}"
        )

        print(
            "平均検出人物数: "
            f"{avg_person_count:.1f}"
        )

        print(
            "人物マスク使用: "
            f"{self.use_person_mask and self.person_detector is not None}"
        )

        return translations_dict
