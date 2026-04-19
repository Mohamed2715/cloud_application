import math
from config.settings import TRACK_MAX_DISTANCE, TRACK_MAX_LOST


class SimpleTracker:
    def __init__(self, max_distance=TRACK_MAX_DISTANCE, max_lost=TRACK_MAX_LOST):
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.objects = {}
        self.next_id = 0

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, detections):
        updated_objects = {}
        matched_old_ids = set()

        for det in detections:
            center = det["center"]
            bbox = det["bbox"]

            matched_id = None
            min_dist = float("inf")

            for obj_id, obj in self.objects.items():
                if obj_id in matched_old_ids:
                    continue

                dist = self._distance(center, obj["center"])
                if dist < self.max_distance and dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id

            if matched_id is not None:
                old_obj = self.objects[matched_id]
                history = old_obj["history"] + [center]
                if len(history) > 15:
                    history = history[-15:]

                updated_objects[matched_id] = {
                    "center": center,
                    "bbox": bbox,
                    "lost": 0,
                    "history": history,
                }
                matched_old_ids.add(matched_id)
            else:
                updated_objects[self.next_id] = {
                    "center": center,
                    "bbox": bbox,
                    "lost": 0,
                    "history": [center],
                }
                self.next_id += 1

        for obj_id, obj in self.objects.items():
            if obj_id not in matched_old_ids:
                lost_count = obj["lost"] + 1
                if lost_count <= self.max_lost:
                    updated_objects[obj_id] = {
                        "center": obj["center"],
                        "bbox": obj["bbox"],
                        "lost": lost_count,
                        "history": obj["history"],
                    }

        self.objects = updated_objects
        return self.objects