import cv2
import numpy as np


def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect[:4]
    return x1 <= px <= x2 and y1 <= py <= y2


def draw_zone(frame, zone, color=(255, 255, 255), thickness=2):
    x1, y1, x2, y2, label = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_zone_overlay(frame, zones):
    overlay = frame.copy()
    for zone in zones:
        x1, y1, x2, y2, label = zone
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), -1)
        draw_zone(frame, zone, color=(200, 200, 200), thickness=1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def draw_detections(frame, results, track_history=None, fps=30.0):
    """Draw bounding boxes with track IDs and optional dwell time."""
    if results is None:
        return
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes.xywh.cpu()
        ids = r.boxes.id
        confs = r.boxes.conf.cpu()

        for i in range(len(boxes)):
            x, y, w, h = boxes[i].numpy()
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            tid = int(ids[i]) if ids is not None and ids[i] is not None else -1
            conf = float(confs[i])

            color = (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{tid} {conf:.2f}"
            if track_history and tid in track_history:
                td = track_history[tid]
                dwell_sec = td.total_dwell_frames / fps
                label += f" {dwell_sec:.0f}s"

            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def get_frame_center(box_xywh):
    x, y, w, h = box_xywh
    return (int(x), int(y))
