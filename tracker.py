from ultralytics import YOLO
import config


class Tracker:
    def __init__(self):
        self.model = YOLO(config.MODEL_NAME)

    def detect_and_track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker=config.TRACKER,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=config.CLASSES,
            verbose=False,
        )
        return results[0] if results else None
