import cv2
import numpy as np
import config


def draw_dashboard(frame, analytics, frame_shape):
    h, w = frame_shape[:2]
    dw = config.DASHBOARD_WIDTH
    dh = config.DASHBOARD_HEIGHT
    dx, dy = config.DASHBOARD_POSITION

    overlay = frame.copy()
    cv2.rectangle(overlay, (dx, dy), (dx + dw, dy + dh), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (100, 100, 100), 1)

    y = dy + 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    white = (255, 255, 255)
    green = (0, 220, 0)
    yellow = (0, 220, 220)

    cv2.putText(frame, "RETAIL INTELLIGENCE", (dx + 10, y), font, 0.55, (0, 180, 255), 1, cv2.LINE_AA)
    y += 28

    occupancy = len([t for t in analytics.tracks.values() if t.last_seen_frame > 0])
    cv2.putText(frame, f"Current Occupancy: {occupancy}", (dx + 10, y), font, fs, white, 1, cv2.LINE_AA)
    y += 22

    cv2.putText(frame, f"Peak Occupancy:    {analytics.peak_occupancy}", (dx + 10, y), font, fs, white, 1, cv2.LINE_AA)
    y += 22

    avg_dwell = analytics.get_avg_store_dwell()
    cv2.putText(frame, f"Avg Dwell:         {avg_dwell:.1f}s", (dx + 10, y), font, fs, white, 1, cv2.LINE_AA)
    y += 22

    cv2.putText(frame, f"Entries: {analytics.entries}   Exits: {analytics.exits}", (dx + 10, y), font, fs, white, 1, cv2.LINE_AA)
    y += 22

    bhs = analytics.business_health_score()
    bhs_color = green if bhs >= 70 else yellow if bhs >= 40 else (0, 0, 220)
    cv2.putText(frame, f"Health Score: {bhs:.0f}/100", (dx + 10, y), font, 0.6, bhs_color, 2, cv2.LINE_AA)
    y += 30

    # zone ranking
    cv2.line(frame, (dx + 10, y), (dx + dw - 10, y), (80, 80, 80), 1)
    y += 18
    cv2.putText(frame, "Zone Popularity", (dx + 10, y), font, 0.5, (0, 180, 255), 1, cv2.LINE_AA)
    y += 20

    pop = analytics.get_zone_popularity()[:4]
    for zone_label, frames in pop:
        dwell_sec = frames / analytics.fps
        text = f"  {zone_label}: {dwell_sec:.0f}s"
        cv2.putText(frame, text, (dx + 10, y), font, fs, white, 1, cv2.LINE_AA)
        y += 18

    return frame
