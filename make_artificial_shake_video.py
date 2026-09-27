import cv2
import numpy as np
from pathlib import Path


def make_artificial_shake_video(
    input_video_path,
    output_video_path,
    max_shift_px=8,
    frequency=0.8,
    random_jitter_px=1.5,
    codec="mp4v"
):
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)

    cap = cv2.VideoCapture(str(input_video_path))

    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"出力動画を作成できません: {output_video_path}")

    print("===== 人工ブレ動画作成 =====")
    print(f"入力: {input_video_path}")
    print(f"出力: {output_video_path}")
    print(f"FPS: {fps}")
    print(f"解像度: {width}x{height}")
    print(f"総フレーム数: {total_frames}")

    frame_idx = 0
    rng = np.random.default_rng(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        t = frame_idx / fps

        # ゆっくりした手ブレ風の揺れ
        dx = max_shift_px * np.sin(2.0 * np.pi * frequency * t)
        dy = max_shift_px * np.cos(2.0 * np.pi * frequency * 0.7 * t)

        # 小さいランダム揺れ
        dx += rng.normal(0.0, random_jitter_px)
        dy += rng.normal(0.0, random_jitter_px)

        matrix = np.float32([
            [1, 0, dx],
            [0, 1, dy]
        ])

        shaken = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        out.write(shaken)

        if frame_idx % 30 == 0:
            print(
                f"処理中: {frame_idx}/{total_frames}, "
                f"dx={dx:.2f}, dy={dy:.2f}"
            )

        frame_idx += 1

    cap.release()
    out.release()

    print("完了")
    print(f"保存先: {output_video_path}")


if __name__ == "__main__":
    input_video = r"D:\卒論用RAFT\input_videos\test 30sec.mp4"
    output_video = r"D:\卒論用RAFT\input_videos\test_30sec_artificial_shake.mp4"

    make_artificial_shake_video(
        input_video,
        output_video,
        max_shift_px=8,
        frequency=0.8,
        random_jitter_px=1.5
    )