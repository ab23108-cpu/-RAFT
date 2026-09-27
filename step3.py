import cv2


class Step3FrameExtractor:
    """
    Step3:
    ブレ区間に基づいてフレームを抽出する。
    RAFTは frame_idx -> frame_idx + 1 を使うため、end + 1 も抽出する。
    """

    def __init__(self):
        pass

    def extract_frames(self, video_path, segments):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"動画を開けません: {video_path}")

        frames_dict = {}
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"総フレーム数: {total_frames}")
        print(f"対象区間数: {len(segments)}")

        for seg_id, (start, end) in enumerate(segments):
            print(f"[Segment {seg_id}] {start} -> {end}")

            last_frame = min(end + 1, total_frames - 1)

            for frame_idx in range(start, last_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                frames_dict[frame_idx] = frame

        cap.release()

        print(f"抽出完了フレーム数: {len(frames_dict)}")

        return frames_dict