import cv2
import numpy as np


class Step5HomographyEstimator:
    """
    Step5:
    RAFTのOptical FlowからHomographyを推定する
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
        """
        flow: (H, W, 2)

        Returns:
            pts1: 元の点
            pts2: flow後の点
        """
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

    def estimate_homography_from_flow(self, flow):
        """
        1つのflowからHomography推定
        """
        pts1, pts2 = self.flow_to_points(flow)

        if len(pts1) < self.min_points:
            return np.eye(3, dtype=np.float32), 0

        H, mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            self.ransac_thresh
        )

        if H is None:
            return np.eye(3, dtype=np.float32), 0

        inliers = int(mask.sum()) if mask is not None else 0

        return H.astype(np.float32), inliers

    def estimate_homographies(self, flows_dict):
        """
        Args:
            flows_dict:
                {
                    frame_idx: flow from frame_idx to frame_idx+1
                }

        Returns:
            homographies_dict:
                {
                    frame_idx: H
                }
        """
        homographies_dict = {}
        inlier_counts = {}

        for frame_idx, flow in flows_dict.items():

            H, inliers = self.estimate_homography_from_flow(flow)

            homographies_dict[frame_idx] = H
            inlier_counts[frame_idx] = inliers

            print(f"H {frame_idx} → {frame_idx + 1}, inliers={inliers}")

        print(f"Homography推定完了: {len(homographies_dict)} pairs")

        return homographies_dict, inlier_counts