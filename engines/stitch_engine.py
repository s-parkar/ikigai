"""Stitching Engine for Zentropy Panoramic Video Generation.

Provides both CLI and Python API with real-time callbacks.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import cv2
import numpy as np

FFMPEG_BIN = r'C:\Users\yashs\ffmpeg-7.1-full_build-shared\bin\ffmpeg.exe'

def load_or_extract_reference_frame(path_or_video):
    if not path_or_video:
        return None
    # 1. If it's directly a readable image
    if os.path.exists(path_or_video) and not path_or_video.lower().endswith(('.mov', '.mp4', '.mkv', '.avi')):
        img = cv2.imread(path_or_video)
        if img is not None:
            return img

    # 2. Check debug_artifacts/frames/
    base_name = os.path.basename(path_or_video)
    debug_path = os.path.join("debug_artifacts", "frames", base_name)
    if os.path.exists(debug_path) and not debug_path.lower().endswith(('.mov', '.mp4', '.mkv', '.avi')):
        img = cv2.imread(debug_path)
        if img is not None:
            return img

    # 3. If it's a video file, decode a frame at 3 seconds / 90 frames
    if os.path.exists(path_or_video):
        cap = cv2.VideoCapture(path_or_video)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return frame

    return None

def compute_calibration_maps(img_l_source, img_r_source, f=1450.0, target_w=3200, target_h=1080):
    img_l = load_or_extract_reference_frame(img_l_source)
    img_r = load_or_extract_reference_frame(img_r_source)
    if img_l is None or img_r is None:
        raise ValueError(f"Could not load or extract calibration reference frames from: {img_l_source}, {img_r_source}")
        
    w, h = img_l.shape[1], img_l.shape[0]

    y_i, x_i = np.indices((h, w), dtype=np.float32)
    x_c = (x_i - w/2) / f
    y_c = (y_i - h/2) / f
    X = np.sin(x_c)
    Y = y_c
    Z = np.cos(x_c)
    x_p = (f * (X / Z) + w/2).astype(np.float32)
    y_p = (f * (Y / Z) + h/2).astype(np.float32)

    cyl_l = cv2.remap(img_l, x_p, y_p, cv2.INTER_LANCZOS4)
    cyl_r = cv2.remap(img_r, x_p, y_p, cv2.INTER_LANCZOS4)

    mask_static_l = np.zeros((h, w), dtype=np.uint8)
    mask_static_l[:int(h*0.5), int(w*0.4):] = 255
    mask_static_r = np.zeros((h, w), dtype=np.uint8)
    mask_static_r[:int(h*0.5), :int(w*0.6)] = 255

    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01)
    kp1, des1 = sift.detectAndCompute(cyl_l, mask=mask_static_l)
    kp2, des2 = sift.detectAndCompute(cyl_r, mask=mask_static_r)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.70 * n.distance]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts)

    th = np.arctan2(M[1, 0], M[0, 0])
    M_l_leveled = cv2.getRotationMatrix2D((w/2, h/2), np.degrees(th/2), 1.0)
    R_r_orig = np.vstack([M, [0, 0, 1]])
    M_l_3 = np.vstack([M_l_leveled, [0, 0, 1]])
    M_r_leveled = (M_l_3 @ R_r_orig)[:2, :]

    corners_l = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    corners_r = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)

    c_l_trans = cv2.transform(corners_l, M_l_leveled).reshape(-1, 2)
    c_r_trans = cv2.transform(corners_r, M_r_leveled).reshape(-1, 2)

    all_c = np.vstack((c_l_trans, c_r_trans))
    x_min, y_min = np.int32(all_c.min(axis=0) - 0.5)
    x_max, y_max = np.int32(all_c.max(axis=0) + 0.5)

    out_w = int(x_max - x_min)
    out_h = int(y_max - y_min)
    if out_w % 2 != 0: out_w += 1
    if out_h % 2 != 0: out_h += 1

    T = np.array([[1, 0, -x_min], [0, 1, -y_min]], dtype=np.float32)
    M_l_final = T[:, :2] @ M_l_leveled[:, :2]
    M_l_final = np.hstack([M_l_final, (T[:, :2] @ M_l_leveled[:, 2:] + T[:, 2:])])

    M_r_final = T[:, :2] @ M_r_leveled[:, :2]
    M_r_final = np.hstack([M_r_final, (T[:, :2] @ M_r_leveled[:, 2:] + T[:, 2:])])

    M_l_inv = cv2.invertAffineTransform(M_l_final)
    M_r_inv = cv2.invertAffineTransform(M_r_final)

    crop_x1, crop_x2 = 250, 3150
    crop_y1, crop_y2 = 258, 1028
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    grid_y, grid_x = np.indices((out_h, out_w), dtype=np.float32)
    ones = np.ones_like(grid_x)
    canvas_coords = np.stack([grid_x, grid_y, ones], axis=-1)

    cyl_l_x = canvas_coords @ M_l_inv[0, :]
    cyl_l_y = canvas_coords @ M_l_inv[1, :]
    x_c_l = (cyl_l_x - w/2) / f
    y_c_l = (cyl_l_y - h/2) / f
    X_l = np.sin(x_c_l)
    Y_l = y_c_l
    Z_l = np.cos(x_c_l)
    raw_l_x = np.where(Z_l > 0, (f * (X_l / Z_l) + w/2).astype(np.float32), -100.0)
    raw_l_y = np.where(Z_l > 0, (f * (Y_l / Z_l) + h/2).astype(np.float32), -100.0)

    cyl_r_x = canvas_coords @ M_r_inv[0, :]
    cyl_r_y = canvas_coords @ M_r_inv[1, :]
    x_c_r = (cyl_r_x - w/2) / f
    y_c_r = (cyl_r_y - h/2) / f
    X_r = np.sin(x_c_r)
    Y_r = y_c_r
    Z_r = np.cos(x_c_r)
    raw_r_x = np.where(Z_r > 0, (f * (X_r / Z_r) + w/2).astype(np.float32), -100.0)
    raw_r_y = np.where(Z_r > 0, (f * (Y_r / Z_r) + h/2).astype(np.float32), -100.0)

    tgt_y, tgt_x = np.indices((target_h, target_w), dtype=np.float32)
    canv_x = tgt_x / float(target_w) * float(crop_w) + float(crop_x1)
    canv_y = tgt_y / float(target_h) * float(crop_h) + float(crop_y1)

    direct_l_x = cv2.remap(raw_l_x, canv_x, canv_y, cv2.INTER_LINEAR)
    direct_l_y = cv2.remap(raw_l_y, canv_x, canv_y, cv2.INTER_LINEAR)
    direct_r_x = cv2.remap(raw_r_x, canv_x, canv_y, cv2.INTER_LINEAR)
    direct_r_y = cv2.remap(raw_r_y, canv_x, canv_y, cv2.INTER_LINEAR)

    map1_l, map2_l = cv2.convertMaps(direct_l_x, direct_l_y, cv2.CV_16SC2)
    map1_r, map2_r = cv2.convertMaps(direct_r_x, direct_r_y, cv2.CV_16SC2)

    mask_l = ((direct_l_x >= 0) & (direct_l_x < w-1) & (direct_l_y >= 0) & (direct_l_y < h-1)).astype(np.uint8) * 255
    mask_r = ((direct_r_x >= 0) & (direct_r_x < w-1) & (direct_r_y >= 0) & (direct_r_y < h-1)).astype(np.uint8) * 255

    dist_l = cv2.distanceTransform(mask_l, cv2.DIST_L2, 5)
    dist_r = cv2.distanceTransform(mask_r, cv2.DIST_L2, 5)

    weight_l = (dist_l / (dist_l + dist_r + 1e-6))[:, :, np.newaxis].astype(np.float32)
    weight_r = (1.0 - weight_l).astype(np.float32)

    overlap_mask_3d = ((mask_l > 0) & (mask_r > 0))[:, :, np.newaxis]
    only_l_mask = ((mask_l > 0) & (mask_r == 0))[:, :, np.newaxis]
    only_r_mask = ((mask_r > 0) & (mask_l == 0))[:, :, np.newaxis]

    w_l_ref = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LANCZOS4)
    w_r_ref = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LANCZOS4)
    ov_bool = (mask_l > 0) & (mask_r > 0)

    lab_l = cv2.cvtColor(w_l_ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(w_r_ref, cv2.COLOR_BGR2LAB).astype(np.float32)

    mean_l = np.mean(lab_l[ov_bool], axis=0)
    std_l = np.std(lab_l[ov_bool], axis=0)
    mean_r = np.mean(lab_r[ov_bool], axis=0)
    std_r = np.std(lab_r[ov_bool], axis=0)

    lab_scale = std_l / (std_r + 1e-5)
    lab_shift = mean_l - mean_r * lab_scale

    rgb_grid = np.indices((32, 32, 32), dtype=np.uint8) * 8
    rgb_samples = rgb_grid.transpose(1, 2, 3, 0).reshape(-1, 1, 3)
    lab_samples = cv2.cvtColor(rgb_samples, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_matched = np.clip(lab_samples * lab_scale + lab_shift, 0, 255).astype(np.uint8)
    bgr_matched_lut = cv2.cvtColor(lab_matched, cv2.COLOR_LAB2BGR).reshape(32, 32, 32, 3)

    return {
        'map1_l': map1_l, 'map2_l': map2_l,
        'map1_r': map1_r, 'map2_r': map2_r,
        'weight_l': weight_l, 'weight_r': weight_r,
        'overlap_mask_3d': overlap_mask_3d,
        'only_l_mask': only_l_mask, 'only_r_mask': only_r_mask,
        'bgr_matched_lut': bgr_matched_lut,
        'target_w': target_w, 'target_h': target_h
    }

def run_stitching(lhs_video, rhs_video, output_video, start_time="00:00:00", duration=None, 
                  progress_callback=None, cancel_flag=None):
    cap_l = cv2.VideoCapture(lhs_video)
    fps_in = cap_l.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames_in = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_l.release()

    ref_img_l = 'lhs_raw_3m.jpg' if os.path.exists('lhs_raw_3m.jpg') else os.path.join("debug_artifacts", "frames", "lhs_raw_3m.jpg")
    if not os.path.exists(ref_img_l):
        ref_img_l = lhs_video

    ref_img_r = 'rhs_raw_3m.jpg' if os.path.exists('rhs_raw_3m.jpg') else os.path.join("debug_artifacts", "frames", "rhs_raw_3m.jpg")
    if not os.path.exists(ref_img_r):
        ref_img_r = rhs_video

    maps = compute_calibration_maps(ref_img_l, ref_img_r)

    target_w = maps['target_w']
    target_h = maps['target_h']
    m1_l, m2_l = maps['map1_l'], maps['map2_l']
    m1_r, m2_r = maps['map1_r'], maps['map2_r']
    w_l_fix = (maps['weight_l'] * 256.0).astype(np.int16)
    w_r_fix = (maps['weight_r'] * 256.0).astype(np.int16)
    ov = maps['overlap_mask_3d']
    ol = maps['only_l_mask']
    orr = maps['only_r_mask']
    lut = maps['bgr_matched_lut']

    temp_audio = f"temp_audio_{int(time.time())}.aac"
    cmd_audio = [FFMPEG_BIN, '-y']
    if start_time and start_time != "00:00:00":
        cmd_audio += ['-ss', str(start_time)]
    cmd_audio += ['-i', lhs_video]
    if duration:
        cmd_audio += ['-t', str(duration)]
    cmd_audio += ['-vn', '-c:a', 'copy', temp_audio]
    subprocess.run(cmd_audio, check=False, stderr=subprocess.DEVNULL)

    cmd_dec_l = [FFMPEG_BIN]
    cmd_dec_r = [FFMPEG_BIN]
    if start_time and start_time != "00:00:00":
        cmd_dec_l += ['-ss', str(start_time)]
        cmd_dec_r += ['-ss', str(start_time)]
    cmd_dec_l += ['-i', lhs_video]
    cmd_dec_r += ['-i', rhs_video]
    if duration:
        cmd_dec_l += ['-t', str(duration)]
        cmd_dec_r += ['-t', str(duration)]
    cmd_dec_l += ['-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    cmd_dec_r += ['-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']

    cmd_enc = [
        FFMPEG_BIN, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{target_w}x{target_h}', '-pix_fmt', 'bgr24', '-r', str(fps_in),
        '-i', 'pipe:0'
    ]
    if os.path.exists(temp_audio):
        cmd_enc += ['-i', temp_audio, '-map', '0:v', '-map', '1:a:0?', '-c:a', 'aac']
    cmd_enc += [
        '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '14', '-b:v', '75M', '-maxrate', '100M', '-bufsize', '120M',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        output_video
    ]

    proc_l = subprocess.Popen(cmd_dec_l, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_r = subprocess.Popen(cmd_dec_r, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_enc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_bytes = 1920 * 1080 * 3
    count = 0
    t0 = time.time()

    total_est = int(float(duration) * fps_in) if duration else total_frames_in

    while True:
        if cancel_flag and cancel_flag():
            print("Stitching cancelled by user.")
            break
        raw_l = proc_l.stdout.read(frame_bytes)
        raw_r = proc_r.stdout.read(frame_bytes)
        if len(raw_l) < frame_bytes or len(raw_r) < frame_bytes:
            break

        frame_l = np.frombuffer(raw_l, dtype=np.uint8).reshape((1080, 1920, 3))
        frame_r = np.frombuffer(raw_r, dtype=np.uint8).reshape((1080, 1920, 3))

        w_l = cv2.remap(frame_l, m1_l, m2_l, cv2.INTER_LINEAR)
        w_r = cv2.remap(frame_r, m1_r, m2_r, cv2.INTER_LINEAR)

        b = w_r[:, :, 0] >> 3
        g = w_r[:, :, 1] >> 3
        r = w_r[:, :, 2] >> 3
        w_r_matched = lut[b, g, r]

        pano = np.where(ov, 
                        ((w_l.astype(np.int16) * w_l_fix + w_r_matched.astype(np.int16) * w_r_fix) >> 8).astype(np.uint8),
               np.where(ol, w_l, 
               np.where(orr, w_r_matched, 0))).astype(np.uint8)

        proc_enc.stdin.write(pano.tobytes())
        count += 1

        if count % 30 == 0:
            elapsed = time.time() - t0
            cur_fps = count / elapsed if elapsed > 0 else 0
            eta = (total_est - count) / cur_fps if cur_fps > 0 and total_est > count else 0
            if progress_callback:
                progress_callback(count, total_est, cur_fps, elapsed, eta)

    proc_l.stdout.close()
    proc_r.stdout.close()
    proc_enc.stdin.close()
    proc_l.wait()
    proc_r.wait()
    proc_enc.wait()

    if os.path.exists(temp_audio):
        try: os.remove(temp_audio)
        except: pass

    return output_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Panoramic Stitching CLI Engine')
    parser.add_argument('--lhs', default='LHS.MOV', help='Left camera video file')
    parser.add_argument('--rhs', default='RHS.MOV', help='Right camera video file')
    parser.add_argument('--output', default='stitched_panorama_full.mp4', help='Output panorama file')
    parser.add_argument('--start', default='00:00:00', help='Start time (HH:MM:SS)')
    parser.add_argument('--duration', default=None, type=float, help='Duration in seconds')
    args = parser.parse_args()

    def print_progress(c, tot, fps, elapsed, eta):
        pct = (c / tot * 100) if tot > 0 else 0
        print(f"[{pct:5.1f}%] {c}/{tot} frames | {fps:4.1f} FPS | Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")
        sys.stdout.flush()

    run_stitching(args.lhs, args.rhs, args.output, start_time=args.start, duration=args.duration, progress_callback=print_progress)
