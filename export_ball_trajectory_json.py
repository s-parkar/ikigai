"""Export AI Ball and Player Tracking Coordinates to JSON/JSONL.

Extracts frame-by-frame (cx, cy, ball_detected, players) coordinates for the full match.
"""

import os
import sys
import json
import time
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def export_tracking_coordinates(input_video, output_jsonl="ball_trajectory_events.jsonl", 
                                output_json="ball_trajectory.json", model_name="yolov8n.pt",
                                progress_callback=None, cancel_flag=None):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Loading YOLO Model: {model_name}...")
    model = YOLO(model_name)

    print(f"Extracting tracking coordinates from {input_video} ({total_frames} frames)...")

    jsonl_file = open(output_jsonl, "w")
    all_frames_data = []

    count = 0
    t0 = time.time()

    infer_w = 1280
    infer_h = int(infer_w * h / w)
    scale_x = w / float(infer_w)
    scale_y = h / float(infer_h)

    # State tracking
    last_cx = w / 2.0
    last_cy = h / 2.0

    while True:
        if cancel_flag and cancel_flag():
            print("Export cancelled by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        infer_frame = cv2.resize(frame, (infer_w, infer_h))
        results = model.predict(infer_frame, classes=[0, 32], conf=0.15, verbose=False, device=0)

        ball_data = None
        players = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1 = float(xyxy[0] * scale_x)
                y1 = float(xyxy[1] * scale_y)
                x2 = float(xyxy[2] * scale_x)
                y2 = float(xyxy[3] * scale_y)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if cls_id == 32: # sports ball
                    ball_data = {
                        "x": round(cx, 1),
                        "y": round(cy, 1),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "confidence": round(conf, 3)
                    }
                elif cls_id == 0: # person
                    players.append({
                        "x": round(cx, 1),
                        "y": round(cy, 1),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "confidence": round(conf, 3)
                    })

        # Calculate action centroid
        if ball_data is not None:
            focus_cx = ball_data["x"]
            focus_cy = ball_data["y"]
        elif len(players) > 0:
            focus_cx = float(np.median([p["x"] for p in players]))
            focus_cy = float(np.median([p["y"] for p in players]))
        else:
            focus_cx = last_cx
            focus_cy = last_cy

        last_cx, last_cy = focus_cx, focus_cy

        # Yaw in degrees (-45 to +45 deg for pano)
        yaw_deg = round((focus_cx / float(w) - 0.5) * 90.0, 2)
        pitch_deg = round((focus_cy / float(h) - 0.5) * 45.0, 2)

        # Standard event object
        event_entry = {
            "frame_index": count,
            "timestamp_ms": round(count / fps * 1000.0, 1),
            "kind": "pan_decision",
            "pose": {
                "yaw": yaw_deg,
                "pitch": pitch_deg,
                "fov_degrees": 55.0,
                "target_cx": round(focus_cx, 1),
                "target_cy": round(focus_cy, 1)
            },
            "ball": ball_data,
            "active_players_count": len(players)
        }

        jsonl_file.write(json.dumps(event_entry) + "\n")
        all_frames_data.append(event_entry)

        count += 1
        if count % 150 == 0:
            elapsed = time.time() - t0
            cur_fps = count / elapsed if elapsed > 0 else 0
            eta = (total_frames - count) / cur_fps if cur_fps > 0 and total_frames > count else 0
            pct = count / total_frames * 100 if total_frames > 0 else 0
            if progress_callback:
                progress_callback(count, total_frames, cur_fps, elapsed, eta)
            else:
                print(f"[{pct:5.1f}%] {count}/{total_frames} frames | {cur_fps:4.1f} FPS | ETA: {eta:5.1f}s")
                sys.stdout.flush()

    cap.release()
    jsonl_file.close()

    # Also save complete JSON array
    with open(output_json, "w") as jf:
        json.dump(all_frames_data, jf, indent=2)

    print(f"SUCCESS! Exported {count} frames of tracking coordinates to:\n1. {output_jsonl}\n2. {output_json}")
    return output_jsonl, output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Ball Tracking Coordinates")
    parser.add_argument("--input", default="stitched_panorama_full.mp4", help="Input panoramic video")
    parser.add_argument("--output_jsonl", default="ball_trajectory_events.jsonl", help="Output JSONL events file")
    parser.add_argument("--output_json", default="ball_trajectory.json", help="Output JSON array file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    args = parser.parse_args()

    export_tracking_coordinates(args.input, args.output_jsonl, args.output_json, args.model)
