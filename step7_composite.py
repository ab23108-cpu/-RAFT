import cv2
import os


class Step7VideoCompositor:
    """
    Step7:
    補正済みフレームを元動画に合成して出力動画を作成する
    """

    def __init__(self, codec="mp4v"):
        self.codec = codec

    def composite_video(
        self,
        input_video_path,
        stabilized_frames_dict,
        output_video_path
    ):
        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            print(f"動画を開けません: {input_video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("===== Step7 動画合成 =====")
        print(f"入力動画: {input_video_path}")
        print(f"出力動画: {output_video_path}")
        print(f"FPS: {fps}")
        print(f"解像度: {width}x{height}")
        print(f"総フレーム数: {total_frames}")
        print(f"差し替えフレーム数: {len(stabilized_frames_dict)}")

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )

        if not out.isOpened():
            print("VideoWriterを開けませんでした")
            cap.release()
            return False

        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx in stabilized_frames_dict:
                new_frame = stabilized_frames_dict[frame_idx]

                # サイズが違う場合に補正
                if new_frame.shape[1] != width or new_frame.shape[0] != height:
                    new_frame = cv2.resize(new_frame, (width, height))

                frame = new_frame

            out.write(frame)

            if frame_idx % 30 == 0:
                print(f"合成中: {frame_idx}/{total_frames}")

            frame_idx += 1

        cap.release()
        out.release()

        if os.path.exists(output_video_path):
            size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
            print(f"出力完了: {output_video_path}")
            print(f"ファイルサイズ: {size_mb:.2f} MB")
            return True

        print("出力ファイルが作成されませんでした")
        return False