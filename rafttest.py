import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch


# =========================================================
# 設定
# =========================================================
PROJECT_ROOT = Path(r"D:\卒論用RAFT")

INPUT_VIDEO = PROJECT_ROOT / "input_videos" / "スライド5ページ開始0秒ブレ人口的.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "raft-things.pth"

OUTPUT_DIR = PROJECT_ROOT / "results" / "raft_first_1sec_flow_color"

RAFT_ITERS = 20
TARGET_SECONDS = 1.0

# background_emphasis_color の強さ
BACKGROUND_AMPLIFY = 12.0
BACKGROUND_PERCENTILE = 90
BACKGROUND_GAMMA = 0.4


# =========================================================
# RAFT import
# =========================================================
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "core"))

from core.raft import RAFT
from core.utils.utils import InputPadder
from utils import flow_viz


# =========================================================
# 日本語パス対応 imwrite
# =========================================================
def imwrite_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix
    if ext == "":
        ext = ".png"

    ok, encoded = cv2.imencode(ext, image)

    if not ok:
        return False

    encoded.tofile(str(path))
    return path.exists()


# =========================================================
# 芝マスク
# =========================================================
def create_grass_mask(frame_bgr):
    """
    緑色の芝領域を抽出する。
    選手や空、建物をなるべく除外して、
    背景寄りのflowを強調するために使う。
    """

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 40, 40], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)

    # 小さいノイズ除去
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 穴埋め
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# =========================================================
# RAFT model load
# =========================================================
def load_raft_model(model_path, device):
    args = Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False
    )

    model = RAFT(args)

    checkpoint = torch.load(
        str(model_path),
        map_location=device
    )

    # DataParallel形式に対応
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {
            k.replace("module.", "", 1): v
            for k, v in checkpoint.items()
        }

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"RAFT model loaded: {model_path}")
    print(f"Device: {device}")

    return model


# =========================================================
# frame -> tensor
# =========================================================
def frame_to_tensor(frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)

    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .float()[None]
        .to(device)
    )

    return tensor


# =========================================================
# RAFT flow 推定
# =========================================================
@torch.no_grad()
def estimate_flow(model, frame1, frame2, device, iters=20):
    image1 = frame_to_tensor(frame1, device)
    image2 = frame_to_tensor(frame2, device)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    _, flow_up = model(
        image1,
        image2,
        iters=iters,
        test_mode=True
    )

    # paddingを戻す
    flow_up = padder.unpad(flow_up)

    flow = (
        flow_up[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    return flow


# =========================================================
# RAFT公式風 flow color
# =========================================================
def create_raft_official_flow_image(flow):
    flow_img = flow_viz.flow_to_image(flow)

    if flow_img.dtype != np.uint8:
        flow_img = np.clip(flow_img, 0, 255).astype(np.uint8)

    return flow_img


# =========================================================
# 02_background_emphasis_color
# =========================================================
def create_background_emphasis_flow_image(
    flow,
    frame_bgr,
    amplify=12.0,
    percentile=90,
    gamma=0.4
):
    """
    背景の動きを見やすくする画像。

    色相:
        動きの方向

    明るさ:
        動きの大きさ

    芝マスク:
        正規化の基準を芝背景に寄せる。
        ただし表示自体は画面全体に出す。
    """

    h, w = flow.shape[:2]

    fx = flow[..., 0]
    fy = flow[..., 1]

    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    angle = np.arctan2(fy, fx)

    grass_mask = create_grass_mask(frame_bgr)

    if grass_mask.shape[:2] != (h, w):
        grass_mask = cv2.resize(
            grass_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    bg_mag = magnitude[grass_mask > 0]

    if len(bg_mag) > 30:
        scale = np.percentile(bg_mag, percentile)
    else:
        scale = np.percentile(magnitude, percentile)

    scale = max(scale, 1e-6)

    mag_norm = magnitude / scale
    mag_norm = mag_norm * amplify
    mag_norm = np.clip(mag_norm, 0.0, 1.0)

    # 小さい動きを少し抑える
    mag_norm[magnitude < 0.05] = 0.0

    # 弱い動きも見やすくする
    mag_norm = mag_norm ** gamma

    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # OpenCVのHは0-179
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)

    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return color


# =========================================================
# 元画像にタイトルを付ける
# =========================================================
def add_title(image, text):
    image = image.copy()

    title_h = 42
    h, w = image.shape[:2]

    canvas = np.zeros((h + title_h, w, 3), dtype=np.uint8)
    canvas[title_h:] = image

    cv2.putText(
        canvas,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return canvas


# =========================================================
# montage保存
# =========================================================
def save_montage(images, output_path, cols=5):
    if len(images) == 0:
        return False

    thumbs = []

    for img in images:
        thumb = cv2.resize(img, (320, 180))
        thumbs.append(thumb)

    rows = int(np.ceil(len(thumbs) / cols))

    montage = np.zeros(
        (rows * 180, cols * 320, 3),
        dtype=np.uint8
    )

    for i, img in enumerate(thumbs):
        r = i // cols
        c = i % cols

        montage[
            r * 180:(r + 1) * 180,
            c * 320:(c + 1) * 320
        ] = img

    return imwrite_unicode(output_path, montage)


# =========================================================
# main
# =========================================================
def make_first_second_raft_flow_images(
    video_path,
    model_path,
    output_dir,
    target_seconds=1.0,
    iters=20
):
    print("===== RAFT First 1sec Flow Visualization =====")

    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    official_dir = output_dir / "01_raft_official_color"
    background_dir = output_dir / "02_background_emphasis_color"
    original_dir = output_dir / "00_original_frames"

    official_dir.mkdir(parents=True, exist_ok=True)
    background_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"RAFTモデルが見つかりません: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_raft_model(model_path, device)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    target_pairs = int(round(fps * target_seconds))
    target_pairs = min(target_pairs, total_frames - 1)

    print(f"Input video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Target flow pairs: {target_pairs}")
    print(f"Output dir: {output_dir}")

    ret, prev_frame = cap.read()

    if not ret:
        cap.release()
        raise RuntimeError("最初のフレームを読み込めませんでした")

    official_montage_images = []
    background_montage_images = []

    for i in range(target_pairs):
        ret, next_frame = cap.read()

        if not ret:
            print(f"frame {i + 1} を読み込めなかったため終了")
            break

        print(f"RAFT estimating: {i} -> {i + 1}")

        flow = estimate_flow(
            model=model,
            frame1=prev_frame,
            frame2=next_frame,
            device=device,
            iters=iters
        )

        original_img = add_title(
            prev_frame,
            f"Original frame {i}"
        )

        official_img = create_raft_official_flow_image(flow)
        official_img = add_title(
            official_img,
            f"RAFT official flow {i} -> {i + 1}"
        )

        background_img = create_background_emphasis_flow_image(
            flow=flow,
            frame_bgr=prev_frame,
            amplify=BACKGROUND_AMPLIFY,
            percentile=BACKGROUND_PERCENTILE,
            gamma=BACKGROUND_GAMMA
        )

        background_img = add_title(
            background_img,
            f"Background emphasis flow {i} -> {i + 1}"
        )

        original_path = original_dir / f"original_frame_{i:04d}.png"
        official_path = official_dir / f"raft_flow_color_{i:04d}_to_{i + 1:04d}.png"
        background_path = background_dir / f"background_emphasis_{i:04d}_to_{i + 1:04d}.png"

        if not imwrite_unicode(original_path, original_img):
            raise RuntimeError(f"画像保存に失敗しました: {original_path}")

        if not imwrite_unicode(official_path, official_img):
            raise RuntimeError(f"画像保存に失敗しました: {official_path}")

        if not imwrite_unicode(background_path, background_img):
            raise RuntimeError(f"画像保存に失敗しました: {background_path}")

        official_montage_images.append(official_img)
        background_montage_images.append(background_img)

        prev_frame = next_frame

    cap.release()

    official_montage_path = output_dir / "raft_flow_first_1sec_montage.png"
    background_montage_path = output_dir / "background_emphasis_first_1sec_montage.png"

    save_montage(
        official_montage_images,
        official_montage_path,
        cols=5
    )

    save_montage(
        background_montage_images,
        background_montage_path,
        cols=5
    )

    print("\n===== 完了 =====")
    print(f"元フレーム保存先: {original_dir}")
    print(f"RAFT公式色画像保存先: {official_dir}")
    print(f"背景強調画像保存先: {background_dir}")
    print(f"RAFT一覧画像: {official_montage_path}")
    print(f"背景強調一覧画像: {background_montage_path}")


if __name__ == "__main__":
    make_first_second_raft_flow_images(
        video_path=INPUT_VIDEO,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        target_seconds=TARGET_SECONDS,
        iters=RAFT_ITERS
    )