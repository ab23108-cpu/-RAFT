import cv2
import numpy as np


class PersonMaskYOLO:
    """
    YOLOv8x + BoT-SORT based person mask generator.

    This class is intended to be used as a preprocessing module for:
    - STEP1 blur segment detection
    - STEP5 translation estimation from RAFT flow

    It detects the "person" class and returns a mask where player/referee
    regions are set to 255. The caller should exclude these pixels from
    optical-flow based blur estimation.
    """

    def __init__(
        self,
        model_name="yolov8x.pt",
        conf=0.25,
        iou=0.5,
        bbox_expand_ratio=0.15,
        use_tracking=True,
        tracker="botsort.yaml",
        device=None
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install it with: pip install ultralytics"
            ) from exc

        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.bbox_expand_ratio = bbox_expand_ratio
        self.use_tracking = use_tracking
        self.tracker = tracker
        self.device = device

    @staticmethod
    def _clip_box(x1, y1, x2, y2, width, height):
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
        return x1, y1, x2, y2

    def _expand_box(self, x1, y1, x2, y2, width, height):
        box_w = x2 - x1
        box_h = y2 - y1

        pad_x = box_w * self.bbox_expand_ratio
        pad_y = box_h * self.bbox_expand_ratio

        return self._clip_box(
            x1 - pad_x,
            y1 - pad_y,
            x2 + pad_x,
            y2 + pad_y,
            width,
            height
        )

    def detect_person_boxes(self, frame_bgr):
        """
        Returns:
            boxes:
                [
                    {
                        "xyxy": [x1, y1, x2, y2],
                        "conf": float,
                        "track_id": int or None
                    },
                    ...
                ]
        """

        height, width = frame_bgr.shape[:2]

        if self.use_tracking:
            results = self.model.track(
                frame_bgr,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
                iou=self.iou,
                classes=[0],
                device=self.device,
                verbose=False
            )
        else:
            results = self.model.predict(
                frame_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=[0],
                device=self.device,
                verbose=False
            )

        boxes = []

        if len(results) == 0 or results[0].boxes is None:
            return boxes

        result_boxes = results[0].boxes

        xyxy = result_boxes.xyxy.detach().cpu().numpy()
        confs = result_boxes.conf.detach().cpu().numpy()

        track_ids = None
        if getattr(result_boxes, "id", None) is not None:
            track_ids = result_boxes.id.detach().cpu().numpy()

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = self._expand_box(
                x1,
                y1,
                x2,
                y2,
                width,
                height
            )

            track_id = None
            if track_ids is not None:
                track_id = int(track_ids[i])

            boxes.append(
                {
                    "xyxy": [x1, y1, x2, y2],
                    "conf": float(confs[i]),
                    "track_id": track_id
                }
            )

        return boxes

    def create_person_mask(self, frame_bgr):
        """
        Returns:
            person_mask:
                uint8 image.
                255 = person area to exclude
                0 = usable background candidate
        """

        height, width = frame_bgr.shape[:2]
        person_mask = np.zeros((height, width), dtype=np.uint8)

        boxes = self.detect_person_boxes(frame_bgr)

        for item in boxes:
            x1, y1, x2, y2 = item["xyxy"]
            person_mask[y1:y2 + 1, x1:x2 + 1] = 255

        kernel = np.ones((7, 7), np.uint8)
        person_mask = cv2.dilate(person_mask, kernel, iterations=1)

        return person_mask, boxes


def create_background_mask_from_person_mask(person_mask):
    """
    Converts a person mask into a background mask.

    Returns:
        background_mask:
            255 = background area
            0 = person area
    """

    return cv2.bitwise_not(person_mask)