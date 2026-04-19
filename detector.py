from ultralytics import YOLO
from settings import VEHICLE_CLASSES, CONF_THRESHOLD


class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, roi_box=None):
        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            roi_frame = frame[ry1:ry2, rx1:rx2]
            results = self.model(roi_frame, verbose=False)[0]
        else:
            rx1, ry1 = 0, 0
            results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in VEHICLE_CLASSES:
                continue
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 += rx1
            x2 += rx1
            y1 += ry1
            y2 += ry1

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "conf": conf,
                "class_id": cls,
            })

        return detections
