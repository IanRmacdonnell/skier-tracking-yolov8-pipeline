from typing import Tuple

import numpy as np


def box_center(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def box_size(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    return np.array([max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=float)


def box_from_center_size(cx: float, cy: float, width: float, height: float) -> np.ndarray:
    half_w = max(1.0, width) / 2.0
    half_h = max(1.0, height) / 2.0
    return np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=float)


def smooth_box(old_box: np.ndarray, new_box: np.ndarray, smoothing: float) -> np.ndarray:
    smoothing = min(0.98, max(0.0, smoothing))
    return (smoothing * old_box.astype(float)) + ((1.0 - smoothing) * new_box.astype(float))


def xyxy_to_xywh(box: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(float)
    return (int(round(x1)), int(round(y1)), int(round(max(1.0, x2 - x1))), int(round(max(1.0, y2 - y1))))


def xywh_to_xyxy(box: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)


def clamp_box(box: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box.astype(float)
    x1 = min(max(0.0, x1), float(width - 2))
    y1 = min(max(0.0, y1), float(height - 2))
    x2 = min(max(x1 + 1.0, x2), float(width - 1))
    y2 = min(max(y1 + 1.0, y2), float(height - 1))
    return np.array([x1, y1, x2, y2], dtype=float)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def expand_box(box: np.ndarray, frame_shape: Tuple[int, int, int], scale_x: float, scale_y: float) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    cx, cy = box_center(box)
    half_w = ((x2 - x1) * scale_x) / 2.0
    half_h = ((y2 - y1) * scale_y) / 2.0
    return clamp_box(np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=float), frame_shape)
