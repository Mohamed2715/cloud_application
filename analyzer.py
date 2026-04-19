import math
from collections import Counter
from config.settings import (
    SLOW_SPEED_THRESHOLD,
    MIN_VEHICLES_FOR_CONGESTION,
    MIN_SLOW_VEHICLES_FOR_CONGESTION,
    ACCIDENT_SLOW_COUNT_THRESHOLD,
    ACCIDENT_LOW_MOTION_THRESHOLD,
)


class Analyzer:
    def __init__(self):
        self.frame_statuses = []
        self.last_metrics = {}

    def _compute_speed_and_direction(self, history):
        if len(history) < 2:
            return 0.0, 0.0, 0.0

        x1, y1 = history[-2]
        x2, y2 = history[-1]

        dx = x2 - x1
        dy = y2 - y1
        speed = math.hypot(dx, dy)

        return speed, dx, dy

    def _side_wrong_count(self, side_objects):
        if len(side_objects) < 4:
            return 0

        avg_dx = sum(obj["dx"] for obj in side_objects) / len(side_objects)
        avg_dy = sum(obj["dy"] for obj in side_objects) / len(side_objects)

        dominant_mag = math.hypot(avg_dx, avg_dy)
        if dominant_mag < 1.5:
            return 0

        count = 0
        for obj in side_objects:
            dot_product = obj["dx"] * avg_dx + obj["dy"] * avg_dy
            if dot_product < 0:
                count += 1

        return count

    def analyze(self, objects):
        valid_objects = []
        motion_values = []
        slow_count = 0
        centers_x = []

        for obj_id, obj in objects.items():
            if obj["lost"] > 0:
                continue

            history = obj["history"]
            if len(history) < 2:
                continue

            speed, dx, dy = self._compute_speed_and_direction(history)
            cx, cy = obj["center"]

            valid_objects.append({
                "center": (cx, cy),
                "dx": dx,
                "dy": dy,
                "speed": speed,
            })

            centers_x.append(cx)
            motion_values.append(speed)

            if speed < SLOW_SPEED_THRESHOLD:
                slow_count += 1

        vehicle_count = len(valid_objects)
        avg_motion = sum(motion_values) / len(motion_values) if motion_values else 0.0

        wrong_way_count = 0

        if centers_x and len(valid_objects) >= 6:
            split_x = (min(centers_x) + max(centers_x)) / 2

            left_side = [obj for obj in valid_objects if obj["center"][0] < split_x]
            right_side = [obj for obj in valid_objects if obj["center"][0] >= split_x]

            left_wrong = self._side_wrong_count(left_side)
            right_wrong = self._side_wrong_count(right_side)

            wrong_way_count = left_wrong + right_wrong

        status = "Normal"

        if (
            slow_count >= ACCIDENT_SLOW_COUNT_THRESHOLD
            and avg_motion < ACCIDENT_LOW_MOTION_THRESHOLD
        ):
            status = "Possible Accident"

        elif (
            vehicle_count >= MIN_VEHICLES_FOR_CONGESTION
            and slow_count >= MIN_SLOW_VEHICLES_FOR_CONGESTION
        ):
            status = "Congestion"

        elif (
            wrong_way_count >= 6
            and vehicle_count <= 12
            and slow_count <= 3
            and avg_motion > 2.5
        ):
            status = "Wrong Way Detected"

        self.frame_statuses.append(status)

        self.last_metrics = {
            "vehicle_count": vehicle_count,
            "slow_count": slow_count,
            "wrong_way_count": wrong_way_count,
            "avg_motion": round(avg_motion, 2),
            "status": status,
        }

        return self.last_metrics

    def final_summary(self):
        if not self.frame_statuses:
            return {
                "final_status": "No Data",
                "status_counts": {},
            }

        counts = Counter(self.frame_statuses)
        final_status = counts.most_common(1)[0][0]

        return {
            "final_status": final_status,
            "status_counts": dict(counts),
        }