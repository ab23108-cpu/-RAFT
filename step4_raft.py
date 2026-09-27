import sys
sys.path.append("core")

import cv2
import torch
import numpy as np
from argparse import Namespace

from core.raft import RAFT
from core.utils.utils import InputPadder


class Step4RAFTFlowEstimator:
    """
    Step4:
    RAFTを使って、隣接フレーム間のOptical Flowを推定する。

    改善点:
    - InputPadderでpaddingした出力flowをunpadして元画像サイズに戻す
    - DataParallel形式 / 通常形式のcheckpoint両方に対応
    """

    def __init__(
        self,
        model_path="models/raft-things.pth",
        small=False,
        mixed_precision=False,
        alternate_corr=False
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision = mixed_precision

        args = Namespace(
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr
        )

        self.model = RAFT(args)

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # DataParallelで保存されたcheckpointは "module." が付く
        if any(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {
                k.replace("module.", "", 1): v
                for k, v in checkpoint.items()
            }

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print(f"RAFT model loaded: {model_path}")
        print(f"Device: {self.device}")

    def frame_to_tensor(self, frame):
        """
        BGR OpenCV frame -> RGB torch tensor
        shape: [1, 3, H, W]
        """

        if frame is None:
            raise ValueError("frame is None")

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb = np.ascontiguousarray(rgb)

        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .float()[None]
            .to(self.device)
        )

        return tensor

    @torch.no_grad()
    def estimate_flow_pair(
        self,
        frame1,
        frame2,
        iters=20
    ):
        """
        frame1 -> frame2 のOptical Flowを推定する。

        Returns:
            flow:
                shape = (H, W, 2)
                flow[..., 0] = x方向
                flow[..., 1] = y方向
        """

        image1 = self.frame_to_tensor(frame1)
        image2 = self.frame_to_tensor(frame2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        if self.device == "cuda" and self.mixed_precision:
            with torch.cuda.amp.autocast():
                _, flow_up = self.model(
                    image1,
                    image2,
                    iters=iters,
                    test_mode=True
                )
        else:
            _, flow_up = self.model(
                image1,
                image2,
                iters=iters,
                test_mode=True
            )

        # 重要:
        # RAFT入力時にpaddingしているため、flowも元サイズへ戻す
        flow_up = padder.unpad(flow_up)

        flow = (
            flow_up[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        return flow

    def estimate_flows(
        self,
        frames_dict,
        segments,
        iters=20
    ):
        """
        ブレ区間内の隣接フレームペアに対してflowを推定する。

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

        Returns:
            flows_dict:
                {
                    frame_idx: flow
                }

                frame_idx -> frame_idx + 1 のflow
        """

        flows_dict = {}

        for seg_id, (start, end) in enumerate(segments):
            print(f"[Segment {seg_id}] {start} -> {end}")

            for frame_idx in range(start, end):
                if frame_idx not in frames_dict:
                    print(f"  skip: frame {frame_idx} not found")
                    continue

                if frame_idx + 1 not in frames_dict:
                    print(f"  skip: frame {frame_idx + 1} not found")
                    continue

                frame1 = frames_dict[frame_idx]
                frame2 = frames_dict[frame_idx + 1]

                flow = self.estimate_flow_pair(
                    frame1,
                    frame2,
                    iters=iters
                )

                flows_dict[frame_idx] = flow

                print(
                    f"  flow {frame_idx} -> {frame_idx + 1}: "
                    f"{flow.shape}"
                )

        print(f"RAFT完了: {len(flows_dict)} flow pairs")

        return flows_dict