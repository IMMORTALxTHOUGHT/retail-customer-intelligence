import cv2
import numpy as np
import config


class Heatmap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.accumulator = np.zeros((height, width), dtype=np.float32)
        self._frame_counter = 0

    def update(self, centers):
        """Add weight at each center point."""
        for (cx, cy) in centers:
            if 0 <= cx < self.width and 0 <= cy < self.height:
                r = 15
                y1 = max(0, cy - r)
                y2 = min(self.height, cy + r)
                x1 = max(0, cx - r)
                x2 = min(self.width, cx + r)
                self.accumulator[y1:y2, x1:x2] += 1.0

    def render(self, frame):
        self._frame_counter += 1
        if self._frame_counter % config.HEATMAP_UPDATE_EVERY != 0:
            return frame

        acc = self.accumulator.copy()
        acc_max = acc.max()
        if acc_max > 0:
            acc = acc / acc_max

        acc_u8 = (acc * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(acc_u8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 1 - config.HEATMAP_ALPHA, heatmap_color, config.HEATMAP_ALPHA, 0)

        # decay accumulator so old heat fades
        self.accumulator *= 0.995

        return blended

    def reset(self):
        self.accumulator[:] = 0
