"""AI Ball & Player Tracking Engine for Zentropy 16:9 Broadcast Video Generation.

Implements Dual-Frame Stabilization:
- Outer Frame: 16:9 Tracking & Deadband Window.
- Inner Frame: High-Stability Inset Broadcast Viewport.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

FFMPEG_BIN = r'C:\Users\yashs\ffmpeg-7.1-full_build-shared\bin\ffmpeg.exe'

def load_coordinates_from_jsonl(jsonl_path, pano_w=3200, pano_h=1080):
    """
    Parses coordinate trajectory or Reco events JSONL file.
    Returns a dict mapping frame_index -> (target_cx, target_cw)
    """
    trajectories = {}
    if not os.path.exists(jsonl_path):
        return trajectories

    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                ev = json.loads(line)
                f_idx = ev.get("frame_index", ev.get("frame", None))
                if f_idx is None: continue

                # pan_decision format (yaw in deg -> mapped to pano_w)
                if ev.get("kind") == "pan_decision":
                    pose = ev.get("pose", {})
                    yaw = pose.get("yaw", 0.0) # degrees, -45 to +45
                    fov = pose.get("fov_degrees", 55.0)
                    cx = (yaw / 90.0 + 0.5) * pano_w
                    cw = (fov / 90.0) * pano_w
                    trajectories[int(f_idx)] = (float(cx), float(cw))
                # 2D ball / coordinate format
                elif "ball" in ev and ev["ball"]:
                    b = ev["ball"]
                    bx = b.get("x", b.get("yaw", None))
                    if bx is not None:
                        if abs(bx) <= 90: # yaw
                            cx = (bx / 90.0 + 0.5) * pano_w
                        else: # pixel x
                            cx = float(bx)
                        trajectories[int(f_idx)] = (cx, float(pano_h * 16.0 / 9.0))
                elif "cx" in ev:
                    trajectories[int(f_idx)] = (float(ev["cx"]), float(ev.get("cw", pano_h * 16.0 / 9.0)))
            except Exception:
                continue
    return trajectories

def run_tracker_broadcast(pano_video, output_video, model_name="yolov8n.pt",
                          coordinates_file=None,
                          inner_crop_scale=0.85, # Inner broadcast frame ratio (0.85 = 15% outer deadband buffer)
                          smoothing=0.06, dynamic_zoom=True, zoom_sensitivity=0.5,
                          start_time=None, duration=None,
                          progress_callback=None, cancel_flag=None):
    """
    Executes AI tracking and Dual-Frame stabilized 16:9 broadcast rendering.
    """
    cap = cv2.VideoCapture(pano_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open panoramic video: {pano_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pano_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pano_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # If coordinates file is provided, load precomputed trajectory
    coord_traj = {}
    if coordinates_file and os.path.exists(coordinates_file):
        print(f"Using Imported Coordinates Trajectory: {coordinates_file}")
        coord_traj = load_coordinates_from_jsonl(coordinates_file, pano_w, pano_h)
        model = None
    else:
        # Load YOLO model
        print(f"Loading AI Model: {model_name}...")
        model = YOLO(model_name)

    # 16:9 crop target dimensions
    out_w, out_h = 1920, 1080
    base_crop_h = pano_h
    base_crop_w = int(base_crop_h * 16.0 / 9.0)
    if base_crop_w > pano_w:
        base_crop_w = pano_w
        base_crop_h = int(base_crop_w * 9.0 / 16.0)

    min_crop_w = int(base_crop_w * 0.70)
    max_crop_w = base_crop_w

    # Audio extraction
    temp_audio = f"temp_audio_track_{int(time.time())}.aac"
    cmd_audio = [FFMPEG_BIN, '-y']
    if start_time and start_time != "00:00:00":
        cmd_audio += ['-ss', str(start_time)]
    cmd_audio += ['-i', pano_video]
    if duration:
        cmd_audio += ['-t', str(duration)]
    cmd_audio += ['-vn', '-c:a', 'copy', temp_audio]
    subprocess.run(cmd_audio, check=False, stderr=subprocess.DEVNULL)

    # Video reader via ffmpeg
    cmd_dec = [FFMPEG_BIN]
    if start_time and start_time != "00:00:00":
        cmd_dec += ['-ss', str(start_time)]
    cmd_dec += ['-i', pano_video]
    if duration:
        cmd_dec += ['-t', str(duration)]
    cmd_dec += ['-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']

    # Video encoder via ffmpeg NVENC
    cmd_enc = [
        FFMPEG_BIN, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', 'pipe:0'
    ]
    if os.path.exists(temp_audio):
        cmd_enc += ['-i', temp_audio, '-map', '0:v', '-map', '1:a:0?', '-c:a', 'aac']
    cmd_enc += [
        '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '15', '-b:v', '45M', '-maxrate', '60M', '-bufsize', '80M',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        output_video
    ]

    proc_dec = subprocess.Popen(cmd_dec, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_enc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_bytes = pano_w * pano_h * 3
    count = 0
    t0 = time.time()
    total_est = int(float(duration) * fps) if duration else total_frames

    # Viewport state (Outer Tracking Frame)
    curr_cx = pano_w / 2.0
    curr_cy = pano_h / 2.0
    curr_cw = float(base_crop_w)
    curr_ch = float(base_crop_h)

    # Action centroid history
    action_cx = curr_cx

    # Inference cadence: run YOLO every 2 frames for speed, interpolate in between
    detect_interval = 2
    last_target_cx = curr_cx
    last_target_cw = curr_cw

    while True:
        if cancel_flag and cancel_flag():
            print("AI Tracking cancelled by user.")
            break

        raw_bytes = proc_dec.stdout.read(frame_bytes)
        if len(raw_bytes) < frame_bytes:
            break

        frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((pano_h, pano_w, 3))

        if coord_traj:
            # Direct coordinate replay
            if count in coord_traj:
                target_cx, target_cw = coord_traj[count]
            else:
                target_cx, target_cw = last_target_cx, last_target_cw
            last_target_cx, last_target_cw = target_cx, target_cw
        elif count % detect_interval == 0 and model is not None:
            # Run YOLO detection
            infer_frame = cv2.resize(frame, (1280, int(1280 * pano_h / pano_w)))
            results = model.predict(infer_frame, classes=[0, 32], conf=0.15, verbose=False, device=0)

            ball_pos = None
            player_centers = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x_center = (xyxy[0] + xyxy[2]) / 2.0 * (pano_w / 1280.0)

                    if cls_id == 32: # sports ball
                        ball_pos = x_center
                    elif cls_id == 0: # person
                        player_centers.append(x_center)

            # Determine primary focus
            if ball_pos is not None:
                target_cx = ball_pos
            elif len(player_centers) > 0:
                target_cx = float(np.median(player_centers))
            else:
                target_cx = curr_cx

            # Dynamic zoom calculation
            if dynamic_zoom and len(player_centers) >= 3:
                p_min, p_max = np.percentile(player_centers, [15, 85])
                spread = p_max - p_min
                target_cw = np.clip(spread * (1.2 + (1.0 - zoom_sensitivity) * 0.8), min_crop_w, max_crop_w)
            else:
                target_cw = base_crop_w

            last_target_cx = target_cx
            last_target_cw = target_cw
        else:
            target_cx = last_target_cx
            target_cw = last_target_cw

        # ─── ULTRA-STABLE VELOCITY-DAMPED DEADBAND STABILIZATION ─────────
        deadband_radius = (curr_cw * 0.18) # 18% deadband zone
        dist_x = target_cx - curr_cx

        if abs(dist_x) < deadband_radius:
            # Smooth cosine blend inside deadband
            factor = 0.5 * (1.0 - np.cos(np.pi * (abs(dist_x) / deadband_radius)))
            target_pull = dist_x * factor
        else:
            target_pull = dist_x - np.sign(dist_x) * deadband_radius

        # Alpha smoothing based on user slider (default ~0.035)
        alpha = np.clip(smoothing * 0.6, 0.01, 0.15)
        vel_x = getattr(run_tracker_broadcast, '_vel_x', 0.0) * 0.78 + target_pull * alpha
        
        # Max velocity clamp (prevents camera jerk)
        max_vel = 14.0 # max pixels per frame
        vel_x = np.clip(vel_x, -max_vel, max_vel)
        run_tracker_broadcast._vel_x = vel_x

        curr_cx += vel_x
        curr_cw = curr_cw * 0.96 + target_cw * 0.04
        curr_ch = curr_cw * 9.0 / 16.0

        # Clamp outer tracking frame
        outer_w = np.clip(curr_cw, min_crop_w, pano_w)
        outer_h = outer_w * 9.0 / 16.0
        outer_cx = np.clip(curr_cx, outer_w / 2.0, pano_w - outer_w / 2.0)

        # ─── INNER BROADCAST FRAME EXTRACTION ─────────────────────────
        # Crop the smaller stabilized inner frame for high-quality broadcast output
        inner_w = outer_w * inner_crop_scale
        inner_h = inner_w * 9.0 / 16.0

        x1 = int(np.clip(outer_cx - inner_w / 2.0, 0, pano_w - inner_w))
        x2 = int(x1 + inner_w)
        y1 = int(np.clip(pano_h / 2.0 - inner_h / 2.0, 0, pano_h - inner_h))
        y2 = int(y1 + inner_h)

        # Crop & upscale stabilized inner frame to 1920x1080
        crop_frame = frame[y1:y2, x1:x2]
        broadcast_frame = cv2.resize(crop_frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        proc_enc.stdin.write(broadcast_frame.tobytes())
        count += 1

        if count % 30 == 0:
            elapsed = time.time() - t0
            cur_fps = count / elapsed if elapsed > 0 else 0
            eta = (total_est - count) / cur_fps if cur_fps > 0 and total_est > count else 0
            if progress_callback:
                progress_callback(count, total_est, cur_fps, elapsed, eta)

    proc_dec.stdout.close()
    proc_enc.stdin.close()
    proc_dec.wait()
    proc_enc.wait()

    if os.path.exists(temp_audio):
        try: os.remove(temp_audio)
        except: pass

    return output_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zentropy Dual-Frame Stabilized AI Tracking Broadcast Generator')
    parser.add_argument('--input', default='stitched_panorama_full.mp4', help='Input panoramic video')
    parser.add_argument('--output', default='broadcast_16_9.mp4', help='Output broadcast video')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model weights')
    parser.add_argument('--smooth', default=0.06, type=float, help='Smoothing factor (0.02 to 0.15)')
    parser.add_argument('--inner_scale', default=0.85, type=float, help='Inner broadcast frame scale (e.g. 0.85)')
    parser.add_argument('--start', default=None, help='Start time (HH:MM:SS)')
    parser.add_argument('--duration', default=None, type=float, help='Duration in seconds')
    args = parser.parse_args()

    def print_progress(c, tot, fps, elapsed, eta):
        pct = (c / tot * 100) if tot > 0 else 0
        print(f"[{pct:5.1f}%] {c}/{tot} frames | {fps:4.1f} FPS | Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")
        sys.stdout.flush()

    run_tracker_broadcast(args.input, args.output, model_name=args.model,
                          inner_crop_scale=args.inner_scale, smoothing=args.smooth,
                          start_time=args.start, duration=args.duration,
                          progress_callback=print_progress)
