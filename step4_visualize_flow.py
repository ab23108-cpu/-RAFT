import os
import cv2
import numpy as np

from utils import flow_viz


def to_numpy_flow(flow):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    return flow.astype(np.float32)


def create_grass_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array((30, 40, 40), dtype=np.uint8),
        np.array((90, 255, 255), dtype=np.uint8)
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def create_raft_official_flow_image(flow, target_shape=None):
    flow = to_numpy_flow(flow)

    flow_rgb = flow_viz.flow_to_image(flow)
    flow_bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)

    if target_shape is not None:
        h, w = target_shape[:2]

        if flow_bgr.shape[:2] != (h, w):
            flow_bgr = cv2.resize(flow_bgr, (w, h))

    return flow_bgr


def create_background_emphasis_flow_image(
    flow,
    frame_bgr,
    amplify=6.0,
    percentile=90,
    gamma=0.4,
    target_shape=None
):
    flow = to_numpy_flow(flow)

    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    magnitude, angle = cv2.cartToPolar(
        flow_x,
        flow_y,
        angleInDegrees=True
    )

    grass_mask = create_grass_mask(frame_bgr)

    if grass_mask.shape[:2] != magnitude.shape[:2]:
        grass_mask = cv2.resize(
            grass_mask,
            (magnitude.shape[1], magnitude.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    bg_mag = magnitude[grass_mask > 0]

    if len(bg_mag) > 0:
        scale = np.percentile(bg_mag, percentile)
    else:
        scale = np.percentile(magnitude, percentile)

    scale = max(scale, 1e-6)

    threshold = 0.05

    visible_mask = magnitude >= threshold

    mag_norm = magnitude / scale
    mag_norm = mag_norm * amplify
    mag_norm = np.clip(mag_norm, 0.0, 1.0)
    mag_norm = mag_norm ** gamma

    mag_norm[~visible_mask] = 0.0

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    hsv[:, :, 0] = (angle / 2).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = (mag_norm * 255).astype(np.uint8)

    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if target_shape is not None:
        h, w = target_shape[:2]

        if flow_bgr.shape[:2] != (h, w):
            flow_bgr = cv2.resize(flow_bgr, (w, h))

    return flow_bgr


def motion_color(value):
    value = float(np.clip(value, 0.0, 1.0))

    color = cv2.applyColorMap(
        np.array([[int(value * 255)]], dtype=np.uint8),
        cv2.COLORMAP_JET
    )[0, 0]

    return tuple(int(c) for c in color)


def create_arrow_grid_overlay_image(
    flow,
    frame_bgr,
    grid_step=80,
    arrow_scale=50.0,
    percentile=95,
    min_magnitude=0.05,
    strength_threshold=0.45
):
    flow = to_numpy_flow(flow)

    image = frame_bgr.copy()
    h, w = image.shape[:2]

    if flow.shape[:2] != (h, w):
        resized_flow = np.zeros((h, w, 2), dtype=np.float32)
        resized_flow[:, :, 0] = cv2.resize(flow[:, :, 0], (w, h))
        resized_flow[:, :, 1] = cv2.resize(flow[:, :, 1], (w, h))
        flow = resized_flow

    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)

    scale = np.percentile(magnitude, percentile)
    scale = max(scale, 1e-6)

    overlay = (image * 0.65).astype(np.uint8)

    for y in range(grid_step // 2, h, grid_step):
        for x in range(grid_step // 2, w, grid_step):
            dx = float(flow_x[y, x])
            dy = float(flow_y[y, x])

            mag = float(np.sqrt(dx ** 2 + dy ** 2))

            if mag < min_magnitude:
                continue

            strength = np.clip(mag / scale, 0.0, 1.0)

            if strength < strength_threshold:
                continue

            color = motion_color(strength)

            start = (x, y)
            end = (
                int(x + dx * arrow_scale),
                int(y + dy * arrow_scale)
            )

            cv2.arrowedLine(
                overlay,
                start,
                end,
                color,
                4,
                tipLength=0.45
            )

    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)

    cv2.putText(
        overlay,
        "RAFT Flow Arrows: blue/green hidden, yellow/red = stronger motion",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    bar_x = 20
    bar_y = h - 50
    bar_w = 320
    bar_h = 20

    for i in range(bar_w):
        v = i / max(bar_w - 1, 1)
        color = motion_color(v)
        cv2.line(
            overlay,
            (bar_x + i, bar_y),
            (bar_x + i, bar_y + bar_h),
            color,
            1
        )

    cv2.putText(
        overlay,
        "weak hidden",
        (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        overlay,
        "strong",
        (bar_x + bar_w - 70, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return overlay


def add_label(image, label):
    labeled = image.copy()

    cv2.rectangle(
        labeled,
        (0, 0),
        (labeled.shape[1], 50),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        labeled,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return labeled


def resize_to_width(image, width=960):
    h, w = image.shape[:2]

    if w == width:
        return image

    scale = width / w
    new_h = int(h * scale)

    return cv2.resize(image, (width, new_h))


def save_raft_flow_images(
    frames_dict,
    flows_dict,
    output_dir,
    max_examples=None
):
    os.makedirs(output_dir, exist_ok=True)

    official_dir = os.path.join(output_dir, "01_raft_official_color")
    emphasis_dir = os.path.join(output_dir, "02_background_emphasis_color")
    arrow_dir = os.path.join(output_dir, "03_arrow_grid_overlay")
    presentation_dir = os.path.join(output_dir, "04_presentation_vertical")

    os.makedirs(official_dir, exist_ok=True)
    os.makedirs(emphasis_dir, exist_ok=True)
    os.makedirs(arrow_dir, exist_ok=True)
    os.makedirs(presentation_dir, exist_ok=True)

    saved_count = 0

    for frame_idx in sorted(flows_dict.keys()):
        if max_examples is not None and saved_count >= max_examples:
            break

        next_frame_idx = frame_idx + 1

        if frame_idx not in frames_dict:
            continue

        frame = frames_dict[frame_idx]
        flow = flows_dict[frame_idx]

        official = create_raft_official_flow_image(
            flow,
            target_shape=frame.shape
        )

        emphasis = create_background_emphasis_flow_image(
            flow,
            frame,
            amplify=12.0,
            percentile=90,
            gamma=0.4,
            target_shape=frame.shape
        )

        arrow = create_arrow_grid_overlay_image(
            flow,
            frame,
            grid_step=80,
            arrow_scale=50.0,
            percentile=95
        )

        cv2.imwrite(
            os.path.join(
                official_dir,
                f"raft_official_{frame_idx:04d}_to_{next_frame_idx:04d}.png"
            ),
            official
        )

        cv2.imwrite(
            os.path.join(
                emphasis_dir,
                f"background_emphasis_{frame_idx:04d}_to_{next_frame_idx:04d}.png"
            ),
            emphasis
        )

        cv2.imwrite(
            os.path.join(
                arrow_dir,
                f"arrow_grid_{frame_idx:04d}_to_{next_frame_idx:04d}.png"
            ),
            arrow
        )

        top = resize_to_width(
            add_label(frame, f"Original frame {frame_idx}"),
            width=960
        )

        mid = resize_to_width(
            add_label(emphasis, "Background-emphasized RAFT flow"),
            width=960
        )

        bottom = resize_to_width(
            add_label(arrow, "RAFT arrows: blue=quiet, red=strong"),
            width=960
        )

        presentation = np.vstack([top, mid, bottom])

        cv2.imwrite(
            os.path.join(
                presentation_dir,
                f"presentation_{frame_idx:04d}_to_{next_frame_idx:04d}.png"
            ),
            presentation
        )

        saved_count += 1
                

    print("\n===== STEP4.5 RAFT Flow画像保存 =====")
    print(f"保存枚数: {saved_count}")
    print(f"保存先: {os.path.abspath(output_dir)}")
    print(f"公式RAFT色画像: {official_dir}")
    print(f"背景強調色画像: {emphasis_dir}")
    print(f"矢印グリッド画像: {arrow_dir}")
    print(f"発表用縦並び画像: {presentation_dir}")