import argparse
import cv2
import sys
import os

import config
from tracker import Tracker
from analytics import Analytics
from heatmap import Heatmap
from dashboard import draw_dashboard
from utils import draw_zone_overlay, draw_detections, get_frame_center


def main():
    parser = argparse.ArgumentParser(description="Retail Customer Intelligence")
    parser.add_argument("--input", "-i", default=config.VIDEO_SOURCE, help="Input video path")
    parser.add_argument("--output", "-o", default=config.OUTPUT_PATH, help="Output video path")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable heatmap overlay")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open video source '{args.input}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Input:  {args.input} ({width}x{height} @ {fps:.1f}fps, {total_frames} frames)")
    print(f"Output: {args.output}")

    tracker = Tracker()
    analytics = Analytics(fps)
    heatmap = Heatmap(width, height) if not args.no_heatmap else None

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % 50 == 0:
            pct = (frame_num / total_frames * 100) if total_frames else 0
            print(f"  Processing frame {frame_num}/{total_frames} ({pct:.0f}%)")

        # 1. Detect & track
        results = tracker.detect_and_track(frame)

        # 2. Update analytics
        analytics.update(results, frame_num)

        # 3. Draw zone overlay
        draw_zone_overlay(frame, config.ZONES)

        # 4. Draw detections with IDs + dwell
        draw_detections(frame, results, analytics.tracks, fps)

        # 5. Heatmap
        if heatmap and results and results.boxes is not None:
            centers = []
            ids = results.boxes.id
            if ids is not None:
                for i in range(len(results.boxes.xywh)):
                    if ids[i] is not None:
                        centers.append(get_frame_center(results.boxes.xywh[i].numpy()))
            heatmap.update(centers)
            frame = heatmap.render(frame)

        # 6. Dashboard
        frame = draw_dashboard(frame, analytics, frame.shape)

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"\nDone. {frame_num} frames processed.")
    print(f"Entries: {analytics.entries}  Peak occupancy: {analytics.peak_occupancy}")
    print(f"Business Health Score: {analytics.business_health_score():.0f}/100")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
