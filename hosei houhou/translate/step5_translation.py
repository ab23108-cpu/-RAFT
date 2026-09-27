import cv2
import numpy as np


class Step5TranslationEstimator:
    """
    Step5 Translation推定

    RAFT Optical Flowから、カメラのx方向・y方向の平行移動量 tx, ty を推定する。

    この版では、
    ・芝生の色情報から背景マスクを作る
    ・背景領域のFlowだけを使う
    ・MADで外れ値を除去する
    ことで、選手やボールの動きをカメラ揺れ推定から除外する。
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
        use_median=True
    ):
        """
        Args:
            grid_step:
                Flowを何ピクセル間隔でサンプリングするか。

            mad_threshold:
                中央値から何MAD以内を背景候補として残すか。
                小さいほど外れ値を強く除外する。

            border_ratio:
                画像端を除外する割合。
                補正後の黒帯や歪みを使わないため。

            min_points:
                最低限必要な背景点数。

            green_lower:
                HSV色空間での芝生色の下限。

            green_upper:
                HSV色空間での芝生色の上限。

            use_grass_mask:
                Trueなら芝生マスクを使う。
                FalseならFlow全体から推定する。

            use_median:
                旧コードとの互換用。
                基本的にはTrueのままでよい。
        """

        self.grid_step = grid_step
        self.mad_threshold = mad_threshold
        self.border_ratio = border_ratio
        self.min_points = min_points

        self.green_lower = np.array(
            green_lower,
            dtype=np.uint8
        )

        self.green_upper = np.array(
            green_upper,
            dtype=np.uint8
        )

        self.use_grass_mask = use_grass_mask
        self.use_median = use_median

    # --------------------------------------------------
    # 芝生マスク作成
    # --------------------------------------------------

    def create_grass_mask(self, frame):
        """
        BGRフレームから芝生領域のマスクを作る。

        Returns:
            grass_mask:
                芝生領域が255、それ以外が0の画像
        """

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        grass_mask = cv2.inRange(
            hsv,
            self.green_lower,
            self.green_upper
        )

        # 小さいノイズを消す
        kernel = np.ones(
            (5, 5),
            np.uint8
        )

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

        return grass_mask

    # --------------------------------------------------
    # 画像端除外マスク
    # --------------------------------------------------

    def create_valid_area_mask(self, height, width):
        """
        画像端を除外するマスクを作る。
        """

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

    # --------------------------------------------------
    # Flowと背景マスクからサンプル点を作る
    # --------------------------------------------------

    def sample_background_flow(
        self,
        flow,
        frame=None
    ):
        """
        背景領域のFlowだけをサンプリングする。
        """

        height, width = flow.shape[:2]

        # ----------------------------------------------
        # 画像端を除外
        # ----------------------------------------------

        valid_area = self.create_valid_area_mask(
            height,
            width
        )

        # ----------------------------------------------
        # 芝生マスクを作成
        # ----------------------------------------------

        if self.use_grass_mask and frame is not None:

            grass_mask = self.create_grass_mask(frame)

            # frameとflowのサイズが違う場合に合わせる
            if grass_mask.shape[:2] != (height, width):
                grass_mask = cv2.resize(
                    grass_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            background_mask = cv2.bitwise_and(
                grass_mask,
                valid_area
            )

        else:
            background_mask = valid_area

        # ----------------------------------------------
        # 格子状にサンプリング
        # ----------------------------------------------

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
                0
            )

        flow_x = flow[ys, xs, 0].astype(np.float64)
        flow_y = flow[ys, xs, 1].astype(np.float64)

        finite = (
            np.isfinite(flow_x)
            & np.isfinite(flow_y)
        )

        flow_x = flow_x[finite]
        flow_y = flow_y[finite]

        return flow_x, flow_y, len(flow_x)

    # --------------------------------------------------
    # MAD外れ値除去
    # --------------------------------------------------

    @staticmethod
    def median_absolute_deviation(values, median_value):
        """
        Median Absolute Deviationを計算する。
        """
        return np.median(
            np.abs(values - median_value)
        )

    def robust_median_translation(
        self,
        flow_x,
        flow_y
    ):
        """
        背景Flowからロバストに tx, ty を推定する。
        """

        if len(flow_x) < self.min_points:
            return 0.0, 0.0, 0

        # 最初の代表値
        median_x = np.median(flow_x)
        median_y = np.median(flow_y)

        # MADを計算
        mad_x = self.median_absolute_deviation(
            flow_x,
            median_x
        )

        mad_y = self.median_absolute_deviation(
            flow_y,
            median_y
        )

        # MADが0に近い場合の対策
        scale_x = max(
            1.4826 * mad_x,
            1e-6
        )

        scale_y = max(
            1.4826 * mad_y,
            1e-6
        )

        # 背景と違う動きをする点を外れ値として除外
        inlier_mask = (
            np.abs(flow_x - median_x)
            <= self.mad_threshold * scale_x
        ) & (
            np.abs(flow_y - median_y)
            <= self.mad_threshold * scale_y
        )

        inlier_x = flow_x[inlier_mask]
        inlier_y = flow_y[inlier_mask]

        # 残った点が少なすぎる場合は最初の中央値を使う
        if len(inlier_x) < self.min_points:
            tx = float(median_x)
            ty = float(median_y)
            used_points = len(flow_x)

        else:
            tx = float(np.median(inlier_x))
            ty = float(np.median(inlier_y))
            used_points = len(inlier_x)

        return tx, ty, used_points

    # --------------------------------------------------
    # 1つのFlowからTranslation推定
    # --------------------------------------------------

    def estimate_translation_from_flow(
        self,
        flow,
        frame=None
    ):
        """
        1フレーム間のFlowから tx, ty を推定する。

        Args:
            flow:
                RAFTで推定したOptical Flow。
                shape = (H, W, 2)

            frame:
                Flowの始点側フレーム。
                芝生マスク作成に使う。

        Returns:
            result:
                tx, tyなどを含む辞書
        """

        flow_x, flow_y, sampled_points = (
            self.sample_background_flow(
                flow,
                frame
            )
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
                )
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
            )
        }

    # --------------------------------------------------
    # 複数FlowからTranslation推定
    # --------------------------------------------------

    def estimate_translations(
        self,
        flows_dict,
        frames_dict=None
    ):
        """
        全Flowに対してTranslationを推定する。

        Args:
            flows_dict:
                {
                    frame_idx: flow
                }

                frame_idx → frame_idx + 1 のFlow

            frames_dict:
                {
                    frame_idx: frame
                }

                芝生マスク作成に使用する。
                Noneの場合は芝生マスクなしで推定する。

        Returns:
            translations_dict:
                {
                    frame_idx: {
                        "tx": ...,
                        "ty": ...
                    }
                }
        """

        translations_dict = {}

        background_point_counts = []

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

        if len(background_point_counts) > 0:
            avg_background_points = float(
                np.mean(background_point_counts)
            )
        else:
            avg_background_points = 0.0

        print(
            "芝生マスク付きTranslation推定完了: "
            f"{len(translations_dict)} pairs"
        )

        print(
            "平均使用背景点数: "
            f"{avg_background_points:.1f}"
        )

        return translations_dict