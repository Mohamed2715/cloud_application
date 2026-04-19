import cv2


def get_roi_box(frame_shape, top_ratio, bottom_ratio, left_ratio, right_ratio):
    h, w = frame_shape[:2]

    x1 = int(w * left_ratio)
    x2 = int(w * right_ratio)
    y1 = int(h * top_ratio)
    y2 = int(h * bottom_ratio)

    return (x1, y1, x2, y2)


def draw_roi(frame, roi_box):
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
    cv2.putText(
        frame,
        "ROI",
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
    )


def draw_boxes(frame, objects):
    for obj_id, obj in objects.items():
        if obj["lost"] > 0:
            continue

        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = obj["center"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        cv2.putText(
            frame,
            f"ID {obj_id}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def draw_dashboard(frame, metrics):
    overlay_lines = [
        f"Status: {metrics.get('status', 'N/A')}",
        f"Vehicles: {metrics.get('vehicle_count', 0)}",
        f"Slow vehicles: {metrics.get('slow_count', 0)}",
        f"Wrong way: {metrics.get('wrong_way_count', 0)}",
        f"Avg motion: {metrics.get('avg_motion', 0.0)}",
    ]

    y = 40
    for line in overlay_lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )
        y += 40

    return frame