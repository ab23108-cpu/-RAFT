import cv2
import numpy as np


class Step5AffineEstimator:
    """
    Step5 Affine版:
    RAFTのOptical FlowからAffine変換を推定する
    """

    def __init__(
        self,
        grid_step=40,
        ransac_thresh=3.0,
        min_points=50
    ):
        self.grid_step = grid_step
        self.ransac_thresh = ransac_thresh
        self.min_points = min_points

    def flow_to_points(self, flow):
        h, w = flow.shape[:2]

        ys, xs = np.mgrid[
            self.grid_step // 2:h:self.grid_step,
            self.grid_step // 2:w:self.grid_step
        ]

        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        flow_x = flow[ys, xs, 0]
        flow_y = flow[ys, xs, 1]

        pts1 = np.stack([xs, ys], axis=1).astype(np.float32)
        pts2 = np.stack([xs + flow_x, ys + flow_y], axis=1).astype(np.float32)

        return pts1, pts2

    def estimate_affine_from_flow(self, flow):
        pts1, pts2 = self.flow_to_points(flow)

        if len(pts1) < self.min_points:
            return np.eye(2, 3, dtype=np.float32), 0

        A, mask = cv2.estimateAffinePartial2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh
        )

        if A is None:
            return np.eye(2, 3, dtype=np.float32), 0

        inliers = int(mask.sum()) if mask is not None else 0

        return A.astype(np.float32), inliers

    def estimate_affines(self, flows_dict):
        affines_dict = {}
        inlier_counts = {}

        for frame_idx, flow in flows_dict.items():
            A, inliers = self.estimate_affine_from_flow(flow)

            affines_dict[frame_idx] = A
            inlier_counts[frame_idx] = inliers

            print(f"A {frame_idx} → {frame_idx + 1}, inliers={inliers}")

        print(f"Affine推定完了: {len(affines_dict)} pairs")

        return affines_dict, inlier_counts