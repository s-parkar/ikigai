"""Generate Annotated AI Ball & Player Tracking Video Feed.

Extracts ball detections and tracking trajectories and renders an annotated video.
"""

import os
import sys
import time
import argparse
import subprocess
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

FFMPEG_BIN = r'C:\Users\yashs\ffmpeg-7.1-full_build-shared\bin\ffmpeg.exe'

def generate_ball_tracking_feed(input_video, output_video, model_name="yolov8n.pt", 
                                start_time=None, duration=None,
                                show_players=True, show_ball_trail=True):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Loading YOLO Model: {model_name}...")
    model = YOLO(model_name)

    # Audio extraction
    temp_audio = f"temp_audio_track_{int(time.time())}.aac"
    cmd_audio = [FFMPEG_BIN, '-y']
    if start_time and start_time != "00:00:00":
        cmd_audio += ['-ss', str(start_time)]
    cmd_audio += ['-i', input_video]
    if duration:
        cmd_audio += ['-t', str(duration)]
    cmd_audio += ['-vn', '-c:a', 'copy', temp_audio]
    subprocess.run(cmd_audio, check=False, stderr=subprocess.DEVNULL)

    # Video reader
    cmd_dec = [FFMPEG_BIN]
    if start_time and start_time != "00:00:00":
        cmd_dec += ['-ss', str(start_time)]
    cmd_dec += ['-i', input_video]
    if duration:
        cmd_dec += ['-t', str(duration)]
    cmd_dec += ['-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']

    # Video encoder (NVENC)
    cmd_enc = [
        FFMPEG_BIN, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', 'pipe:0'
    ]
    if os.path.exists(temp_audio):
        cmd_enc += ['-i', temp_audio, '-map', '0:v', '-map', '1:a:0?', '-c:a', 'aac']
    cmd_enc += [
        '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '15', '-b:v', '60M', '-maxrate', '80M', '-bufsize', '100M',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        output_video
    ]

    proc_dec = subprocess.Popen(cmd_dec, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_enc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_bytes = w * h * 3
    count = 0
    t0 = time.time()
    total_est = int(float(duration) * fps) if duration else total_frames

    # Ball trajectory queue (stores last 30 ball centers)
    ball_trail = deque(maxlen=30)

    print(f"Rendering Annotated Ball Tracking Feed -> {output_video}...")

    while True:
        raw_bytes = proc_dec.stdout.read(frame_bytes)
        if len(raw_bytes) < frame_bytes:
            break

        frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))

        # Downscale for fast YOLO inference
        infer_w = 1280
        infer_h = int(infer_w * h / w)
        infer_frame = cv2.resize(frame, (infer_w, infer_h))
        scale_x = w / float(infer_w)
        scale_y = h / float(infer_h)

        results = model.predict(infer_frame, classes=[0, 32], conf=0.15, verbose=False, device=0)

        current_ball = None
        player_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1 = int(xyxy[0] * scale_x)
                y1 = int(xyxy[1] * scale_y)
                x2 = int(xyxy[2] * scale_x)
                y2 = int(xyxy[3] * scale_y)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if cls_id == 32: # Ball
                    current_ball = (cx, cy, conf, x1, y1, x2, y2)
                elif cls_id == 0 and show_players: # Person
                    player_count += 1
                    # Draw sleek player box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 2)
                    cv2.putText(frame, f"Player {conf:.2f}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 1, cv2.LINE_AA)

        # Draw Ball Tracking & Trajectory
        if current_ball is not None:
            bx, by, bconf, bx1, by1, bx2, by2 = current_ball
            ball_trail.append((bx, by))
            
            # Glowing ball marker (multi-ring)
            cv2.circle(frame, (bx, by), 24, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 16, (0, 200, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 4, (0, 255, 255), -1, cv2.LINE_AA)
            
            # Label
            cv2.putText(frame, f"BALL ({bconf:.2f})", (bx - 30, by - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        else:
            # Decay trail if ball lost
            if len(ball_trail) > 0 and count % 3 == 0:
                ball_trail.popleft()

        # Draw trajectory comet tail
        if show_ball_trail and len(ball_trail) > 1:
            for i in range(1, len(ball_trail)):
                pt1 = ball_trail[i - 1]
                pt2 = ball_trail[i]
                alpha = float(i) / len(ball_trail)
                thickness = int(1 + alpha * 4)
                color = (int(0 * alpha), int(160 * alpha), int(255 * alpha))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        # HUD Overlay
        cv2.rectangle(frame, (20, 20), (360, 110), (15, 23, 42), -1)
        cv2.rectangle(frame, (20, 20), (360, 110), (51, 65, 85), 2)
        cv2.putText(frame, "RECO AI BALL TRACKER", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 185, 129), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Ball Status: {'LOCKED' if current_ball else 'SEARCHING'}", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255) if current_ball else (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Active Players: {player_count} | Frame: {count}", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 232, 240), 1, cv2.LINE_AA)

        proc_enc.stdin.write(frame.tobytes())
        count += 1

        if count % 30 == 0:
            elapsed = time.time() - t0
            cur_fps = count / elapsed if elapsed > 0 else 0
            eta = (total_est - count) / cur_fps if cur_fps > 0 and total_est > count else 0
            pct = count / total_est * 100 if total_est > 0 else 0
            print(f"[{pct:5.1f}%] {count}/{total_est} frames | {cur_fps:4.1f} FPS | ETA: {eta:5.1f}s")
            sys.stdout.flush()

    proc_dec.stdout.close()
    proc_enc.stdin.close()
    proc_dec.wait()
    proc_enc.wait()

    if os.path.exists(temp_audio):
        try: os.remove(temp_audio)
        except: pass

    print(f"FINISHED! Saved annotated ball tracking video to {output_video}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Ball Tracking Feed Generator')
    parser.add_argument('--input', default='stitched_panorama_full.mp4', help='Input panoramic or raw video')
    parser.add_argument('--output', default='ball_tracking_feed.mp4', help='Output annotated tracking video')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model weights')
    parser.add_argument('--start', default=None, help='Start time (HH:MM:SS)')
    parser.add_argument('--duration', default=None, type=float, help='Duration in seconds')
    args = parser.parse_args()

    generate_ball_tracking_feed(args.input, args.output, model_name=args.model, start_time=args.start, duration=args.duration)
