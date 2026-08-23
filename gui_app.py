"""Zentropy GUI Control Center & Pipeline Manager.

Graphical Desktop Suite for Panoramic Stitching, AI Tracking, and Live Broadcast Studio.
"""

import os
import sys
import json
import time
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np

# Set theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "lhs_video": "LHS.MOV",
    "rhs_video": "RHS.MOV",
    "pano_output": "stitched_panorama_full.mp4",
    "pano_input": "stitched_panorama_full.mp4",
    "broadcast_output": "broadcast_16_9.mp4",
    "yolo_model": "yolov8n.pt",
    "smoothing": 0.06,
    "dynamic_zoom": True,
    "zoom_sensitivity": 0.5,
    "stitch_start": "00:00:00",
    "stitch_duration": "",
    "track_start": "00:00:00",
    "track_duration": "",
    "live_pano_source": "stitched_panorama_full.mp4",
    "live_coords_source": "ball_trajectory_events.jsonl"
}

def parse_trajectory_file(file_path, pano_w=3200, pano_h=1080):
    trajectories = {}
    if not file_path or not os.path.exists(file_path):
        return trajectories

    try:
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        ev = json.loads(line)
                        f_idx = ev.get("frame_index", ev.get("frame", None))
                        if f_idx is None: continue
                        if ev.get("kind") == "pan_decision":
                            pose = ev.get("pose", {})
                            yaw = pose.get("yaw", 0.0)
                            fov = pose.get("fov_degrees", 55.0)
                            cx = (yaw / 90.0 + 0.5) * pano_w
                            cw = (fov / 90.0) * pano_w
                            trajectories[int(f_idx)] = (float(cx), float(cw))
                        elif "ball" in ev and ev["ball"]:
                            b = ev["ball"]
                            bx = b.get("x", b.get("yaw", None))
                            if bx is not None:
                                cx = (bx / 90.0 + 0.5) * pano_w if abs(bx) <= 90 else float(bx)
                                trajectories[int(f_idx)] = (cx, float(pano_h * 16.0 / 9.0))
                        elif "cx" in ev:
                            trajectories[int(f_idx)] = (float(ev["cx"]), float(ev.get("cw", pano_h * 16.0 / 9.0)))
                    except Exception:
                        continue
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for ev in data:
                        f_idx = ev.get("frame_index", ev.get("frame", None))
                        if f_idx is None: continue
                        if ev.get("kind") == "pan_decision":
                            pose = ev.get("pose", {})
                            yaw = pose.get("yaw", 0.0)
                            fov = pose.get("fov_degrees", 55.0)
                            cx = (yaw / 90.0 + 0.5) * pano_w
                            cw = (fov / 90.0) * pano_w
                            trajectories[int(f_idx)] = (float(cx), float(cw))
                        elif "ball" in ev and ev["ball"]:
                            b = ev["ball"]
                            bx = b.get("x", b.get("yaw", None))
                            if bx is not None:
                                cx = (bx / 90.0 + 0.5) * pano_w if abs(bx) <= 90 else float(bx)
                                trajectories[int(f_idx)] = (cx, float(pano_h * 16.0 / 9.0))
                        elif "cx" in ev:
                            trajectories[int(f_idx)] = (float(ev["cx"]), float(ev.get("cw", pano_h * 16.0 / 9.0)))
    except Exception as e:
        print(f"Error parsing trajectory file {file_path}: {e}")
    return trajectories

class ZentropyControlCenter(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Zentropy AI Broadcast Control Center - Pipeline Manager")
        self.geometry("1260x940")
        self.minsize(1100, 800)

        self.config_data = self.load_config()
        self.current_worker = None
        self.cancel_requested = False
        self.stream_server = None

        # Tab 2 Preview state
        self.preview_cap = None
        self.preview_playing = False

        # Tab 3 Live Broadcast Studio state
        self.live_playing = False
        self.live_cap = None
        self.live_ptz_x = 0.5 # 0.0 to 1.0 (center)
        self.live_ptz_w = 1920.0
        self.live_auto_track = True
        self.live_current_frame = None
        self.live_trajectory = {}

        self.create_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    cfg = json.load(f)
                    return {**DEFAULT_CONFIG, **cfg}
            except Exception:
                pass
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config_data, f, indent=4)
        except Exception:
            pass

    def create_layout(self):
        # Top Header Bar
        self.header_frame = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="#0f172a")
        self.header_frame.pack(fill="x", side="top")

        self.logo_label = ctk.CTkLabel(
            self.header_frame, 
            text="⚽ ZENTROPY AI BROADCAST STUDIO", 
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#10b981"
        )
        self.logo_label.pack(side="left", padx=20, pady=15)

        self.status_badge = ctk.CTkLabel(
            self.header_frame,
            text="● IDLE",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#94a3b8",
            fg_color="#1e293b",
            corner_radius=12,
            padx=12,
            pady=4
        )
        # Bottom Collapsible Log Console (Built first so logging is available during tab init)
        self.build_log_console()

        # Tabview
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=10)

        self.tab_stitch = self.tabview.add(" 📷 1. Pano Stitcher ")
        self.tab_track = self.tabview.add(" 🎯 2. AI Tracking & Broadcast ")
        self.tab_ptz = self.tabview.add(" 📡 3. Live Broadcast Studio ")

        self.build_stitch_tab()
        self.build_track_tab()
        self.build_ptz_tab()

    # ─── TAB 1: PANO STITCHER ──────────────────────────────────────────
    def build_stitch_tab(self):
        container = ctk.CTkScrollableFrame(self.tab_stitch, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        card_in = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_in.pack(fill="x", pady=5, padx=5)

        lbl_sec = ctk.CTkLabel(card_in, text="SOURCE CAMERA FEEDS (LHS + RHS)", font=ctk.CTkFont(size=14, weight="bold"), text_color="#38bdf8")
        lbl_sec.grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(10, 5))

        # LHS Picker
        ctk.CTkLabel(card_in, text="Left Camera (LHS):").grid(row=1, column=0, sticky="w", padx=15, pady=5)
        self.entry_lhs = ctk.CTkEntry(card_in, width=450)
        self.entry_lhs.insert(0, self.config_data.get("lhs_video", "LHS.MOV"))
        self.entry_lhs.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_in, text="Browse...", width=90, command=lambda: self.browse_file(self.entry_lhs)).grid(row=1, column=2, padx=15, pady=5)

        # RHS Picker
        ctk.CTkLabel(card_in, text="Right Camera (RHS):").grid(row=2, column=0, sticky="w", padx=15, pady=5)
        self.entry_rhs = ctk.CTkEntry(card_in, width=450)
        self.entry_rhs.insert(0, self.config_data.get("rhs_video", "RHS.MOV"))
        self.entry_rhs.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_in, text="Browse...", width=90, command=lambda: self.browse_file(self.entry_rhs)).grid(row=2, column=2, padx=15, pady=5)

        # Time range
        time_frame = ctk.CTkFrame(card_in, fg_color="transparent")
        time_frame.grid(row=3, column=0, columnspan=3, sticky="w", padx=15, pady=10)
        ctk.CTkLabel(time_frame, text="Start Time (HH:MM:SS):").pack(side="left", padx=(0, 5))
        self.entry_stitch_start = ctk.CTkEntry(time_frame, width=90)
        self.entry_stitch_start.insert(0, self.config_data.get("stitch_start", "00:00:00"))
        self.entry_stitch_start.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(time_frame, text="Duration (sec, empty=Full):").pack(side="left", padx=(0, 5))
        self.entry_stitch_dur = ctk.CTkEntry(time_frame, width=90)
        self.entry_stitch_dur.insert(0, self.config_data.get("stitch_duration", ""))
        self.entry_stitch_dur.pack(side="left", padx=(0, 20))

        # Output path
        ctk.CTkLabel(card_in, text="Output Panorama:").grid(row=4, column=0, sticky="w", padx=15, pady=5)
        self.entry_pano_out = ctk.CTkEntry(card_in, width=450)
        self.entry_pano_out.insert(0, self.config_data.get("pano_output", "stitched_panorama_full.mp4"))
        self.entry_pano_out.grid(row=4, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_in, text="Save As...", width=90, command=lambda: self.browse_save_file(self.entry_pano_out)).grid(row=4, column=2, padx=15, pady=5)

        card_in.columnconfigure(1, weight=1)

        # Actions & Progress
        card_act = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_act.pack(fill="x", pady=10, padx=5)

        btn_row = ctk.CTkFrame(card_act, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=10)

        self.btn_run_stitch = ctk.CTkButton(
            btn_row, 
            text="▶ GENERATE PANORAMIC VIDEO", 
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10b981", 
            hover_color="#059669",
            height=40,
            command=self.start_stitch_job
        )
        self.btn_run_stitch.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_abort_stitch = ctk.CTkButton(
            btn_row, 
            text="⏹ ABORT", 
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#ef4444", 
            hover_color="#dc2626",
            height=40,
            width=120,
            state="disabled",
            command=self.abort_job
        )
        self.btn_abort_stitch.pack(side="right")

        self.stitch_prog_bar = ctk.CTkProgressBar(card_act)
        self.stitch_prog_bar.set(0)
        self.stitch_prog_bar.pack(fill="x", padx=15, pady=(5, 10))

        self.lbl_stitch_stats = ctk.CTkLabel(card_act, text="Ready to stitch.", font=ctk.CTkFont(size=12), text_color="#94a3b8")
        self.lbl_stitch_stats.pack(anchor="w", padx=15, pady=(0, 10))

        # Preview card
        self.card_stitch_prev = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        self.card_stitch_prev.pack(fill="both", expand=True, pady=5, padx=5)
        ctk.CTkLabel(self.card_stitch_prev, text="PANORAMA REFERENCE PREVIEW", font=ctk.CTkFont(size=13, weight="bold"), text_color="#94a3b8").pack(anchor="w", padx=15, pady=5)
        
        self.lbl_pano_img = ctk.CTkLabel(self.card_stitch_prev, text="", height=180, fg_color="#0f172a", corner_radius=6)
        self.lbl_pano_img.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.load_static_pano_preview()

    def load_static_pano_preview(self):
        ref_path = "check_natural_3200x1080.jpg"
        if os.path.exists(ref_path):
            img = Image.open(ref_path)
            w, h = img.size
            tw = 680
            th = int(tw * h / w)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(tw, th))
            self.lbl_pano_img.configure(image=ctk_img, text="")
            self.lbl_pano_img.image = ctk_img

    # ─── TAB 2: AI TRACKING & AUTO-BROADCAST ───────────────────────────
    def build_track_tab(self):
        container = ctk.CTkScrollableFrame(self.tab_track, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        card_in = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_in.pack(fill="x", pady=5, padx=5)

        lbl_sec = ctk.CTkLabel(card_in, text="SOURCE PANORAMA & AI TRACKER", font=ctk.CTkFont(size=14, weight="bold"), text_color="#38bdf8")
        lbl_sec.grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(10, 5))

        # Pano input
        ctk.CTkLabel(card_in, text="Source Panorama Video:").grid(row=1, column=0, sticky="w", padx=15, pady=5)
        self.entry_pano_in = ctk.CTkEntry(card_in, width=450)
        self.entry_pano_in.insert(0, self.config_data.get("pano_input", "stitched_panorama_full.mp4"))
        self.entry_pano_in.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_in, text="Browse...", width=90, command=lambda: self.browse_file(self.entry_pano_in)).grid(row=1, column=2, padx=15, pady=5)

        # Model Weights & Coordinates Importer
        ctk.CTkLabel(card_in, text="YOLO Model Weights:").grid(row=2, column=0, sticky="w", padx=15, pady=5)
        self.entry_model = ctk.CTkEntry(card_in, width=450)
        self.entry_model.insert(0, self.config_data.get("yolo_model", "yolov8n.pt"))
        self.entry_model.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_in, text="Browse...", width=90, command=lambda: self.browse_file(self.entry_model)).grid(row=2, column=2, padx=15, pady=5)

        # Optional Coordinates JSONL
        ctk.CTkLabel(card_in, text="Import Coordinates (.jsonl):").grid(row=3, column=0, sticky="w", padx=15, pady=5)
        self.entry_coords = ctk.CTkEntry(card_in, width=450, placeholder_text="Optional: Select ball_trajectory_events.jsonl")
        self.entry_coords.insert(0, self.config_data.get("live_coords_source", "ball_trajectory_events.jsonl"))
        self.entry_coords.grid(row=3, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_in, text="Browse...", width=90, command=lambda: self.browse_file(self.entry_coords)).grid(row=3, column=2, padx=15, pady=5)

        # Time range
        time_frame = ctk.CTkFrame(card_in, fg_color="transparent")
        time_frame.grid(row=4, column=0, columnspan=3, sticky="w", padx=15, pady=5)
        ctk.CTkLabel(time_frame, text="Start Time (HH:MM:SS):").pack(side="left", padx=(0, 5))
        self.entry_track_start = ctk.CTkEntry(time_frame, width=90)
        self.entry_track_start.insert(0, self.config_data.get("track_start", "00:00:00"))
        self.entry_track_start.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(time_frame, text="Duration (sec, empty=Full):").pack(side="left", padx=(0, 5))
        self.entry_track_dur = ctk.CTkEntry(time_frame, width=90)
        self.entry_track_dur.insert(0, self.config_data.get("track_duration", ""))
        self.entry_track_dur.pack(side="left", padx=(0, 20))

        # Broadcast Controls
        ctk.CTkLabel(card_in, text="Camera Smoothing:").grid(row=5, column=0, sticky="w", padx=15, pady=5)
        self.slider_smooth = ctk.CTkSlider(card_in, from_=0.02, to=0.15, number_of_steps=13)
        self.slider_smooth.set(self.config_data.get("smoothing", 0.06))
        self.slider_smooth.grid(row=5, column=1, sticky="ew", padx=10, pady=5)
        self.lbl_smooth_val = ctk.CTkLabel(card_in, text=f"Ultra-Smooth ({self.slider_smooth.get():.2f})", width=110)
        self.lbl_smooth_val.grid(row=5, column=2, padx=15, pady=5)
        self.slider_smooth.configure(command=lambda v: self.lbl_smooth_val.configure(text=f"{'Cinematic' if v < 0.04 else 'Ultra-Smooth' if v < 0.09 else 'Fast'} ({v:.2f})"))

        # Output broadcast
        ctk.CTkLabel(card_in, text="Broadcast Output (16:9):").grid(row=6, column=0, sticky="w", padx=15, pady=5)
        self.entry_broad_out = ctk.CTkEntry(card_in, width=450)
        self.entry_broad_out.insert(0, self.config_data.get("broadcast_output", "broadcast_16_9.mp4"))
        self.entry_broad_out.grid(row=6, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_in, text="Save As...", width=90, command=lambda: self.browse_save_file(self.entry_broad_out)).grid(row=6, column=2, padx=15, pady=5)

        card_in.columnconfigure(1, weight=1)

        # Actions & Progress
        card_act = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_act.pack(fill="x", pady=10, padx=5)

        btn_row = ctk.CTkFrame(card_act, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=10)

        self.btn_run_track = ctk.CTkButton(
            btn_row, 
            text="▶ GENERATE 16:9 BROADCAST", 
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3b82f6", 
            hover_color="#2563eb",
            height=40,
            command=self.start_track_job
        )
        self.btn_run_track.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_run_ball_feed = ctk.CTkButton(
            btn_row, 
            text="🎯 BALL TRACKING FEED", 
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#f59e0b", 
            hover_color="#d97706",
            height=40,
            command=self.start_ball_feed_job
        )
        self.btn_run_ball_feed.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_export_json = ctk.CTkButton(
            btn_row, 
            text="📄 EXPORT JSON/JSONL", 
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#8b5cf6", 
            hover_color="#7c3aed",
            height=40,
            command=self.start_export_json_job
        )
        self.btn_export_json.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_abort_track = ctk.CTkButton(
            btn_row, 
            text="⏹ ABORT", 
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#ef4444", 
            hover_color="#dc2626",
            height=40,
            width=80,
            state="disabled",
            command=self.abort_job
        )
        self.btn_abort_track.pack(side="right")

        self.track_prog_bar = ctk.CTkProgressBar(card_act)
        self.track_prog_bar.set(0)
        self.track_prog_bar.pack(fill="x", padx=15, pady=(5, 10))

        self.lbl_track_stats = ctk.CTkLabel(card_act, text="Ready to track.", font=ctk.CTkFont(size=12), text_color="#94a3b8")
        self.lbl_track_stats.pack(anchor="w", padx=15, pady=(0, 10))

        # Video Player Preview
        self.card_track_prev = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        self.card_track_prev.pack(fill="both", expand=True, pady=5, padx=5)
        ctk.CTkLabel(self.card_track_prev, text="16:9 BROADCAST VIDEO PLAYER", font=ctk.CTkFont(size=13, weight="bold"), text_color="#94a3b8").pack(anchor="w", padx=15, pady=5)
        
        self.lbl_track_img = ctk.CTkLabel(self.card_track_prev, text="No video loaded", height=240, fg_color="#0f172a", corner_radius=6)
        self.lbl_track_img.pack(fill="both", expand=True, padx=15, pady=(0, 5))

        ctrl_row = ctk.CTkFrame(self.card_track_prev, fg_color="transparent")
        ctrl_row.pack(fill="x", padx=15, pady=(0, 15))

        self.btn_play_prev = ctk.CTkButton(ctrl_row, text="▶ Play Preview", width=120, command=self.toggle_preview_playback)
        self.btn_play_prev.pack(side="left", padx=(0, 10))

    # ─── TAB 3: LIVE BROADCAST STUDIO ──────────────────────────────────
    def build_ptz_tab(self):
        container = ctk.CTkScrollableFrame(self.tab_ptz, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Card 1: Source & Trajectory Import
        card_src = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_src.pack(fill="x", pady=5, padx=5)

        lbl_sec = ctk.CTkLabel(card_src, text="LIVE BROADCAST SOURCE & TRAJECTORY IMPORT", font=ctk.CTkFont(size=14, weight="bold"), text_color="#38bdf8")
        lbl_sec.grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(10, 5))

        # Panorama source
        ctk.CTkLabel(card_src, text="Panorama Video:").grid(row=1, column=0, sticky="w", padx=15, pady=5)
        self.entry_live_pano = ctk.CTkEntry(card_src, width=450)
        self.entry_live_pano.insert(0, self.config_data.get("live_pano_source", "stitched_panorama_full.mp4"))
        self.entry_live_pano.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_src, text="Browse...", width=90, command=lambda: self.browse_file(self.entry_live_pano)).grid(row=1, column=2, padx=15, pady=5)

        # Start Time & Seek Bar
        time_bar = ctk.CTkFrame(card_src, fg_color="transparent")
        time_bar.grid(row=3, column=0, columnspan=3, sticky="ew", padx=15, pady=5)

        ctk.CTkLabel(time_bar, text="Start Video From (HH:MM:SS):").pack(side="left", padx=(0, 5))
        self.entry_live_start = ctk.CTkEntry(time_bar, width=90)
        self.entry_live_start.insert(0, "00:00:00")
        self.entry_live_start.pack(side="left", padx=(0, 15))

        self.btn_seek_live = ctk.CTkButton(time_bar, text="⏩ Jump to Time", width=110, height=28, command=self.seek_live_start_time)
        self.btn_seek_live.pack(side="left", padx=(0, 20))

        self.lbl_live_time_display = ctk.CTkLabel(time_bar, text="Position: 00:00:00 (Frame 0)", font=ctk.CTkFont(size=12), text_color="#94a3b8")
        self.lbl_live_time_display.pack(side="right")

        card_src.columnconfigure(1, weight=1)

        # Card 2: Dual-View Studio
        card_views = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_views.pack(fill="both", expand=True, pady=10, padx=5)

        lbl_pano_title = ctk.CTkLabel(card_views, text="1. FULL FIELD PANORAMA (GREEN 16:9 BOX AUTO-FOLLOWS BALL TRAJECTORY)", font=ctk.CTkFont(size=13, weight="bold"), text_color="#10b981")
        lbl_pano_title.pack(anchor="w", padx=15, pady=(10, 2))

        self.ptz_canvas = tk.Canvas(card_views, height=180, bg="#0f172a", highlightthickness=1, highlightbackground="#334155")
        self.ptz_canvas.pack(fill="x", padx=15, pady=5)
        self.ptz_canvas.bind("<B1-Motion>", self.on_ptz_drag)
        self.ptz_canvas.bind("<Button-1>", self.on_ptz_drag)

        lbl_out_title = ctk.CTkLabel(card_views, text="2. LIVE 16:9 BROADCAST OUTPUT (DIRECT CROPPED & ZOOMED STREAM)", font=ctk.CTkFont(size=13, weight="bold"), text_color="#38bdf8")
        lbl_out_title.pack(anchor="w", padx=15, pady=(10, 2))

        self.lbl_live_16_9 = ctk.CTkLabel(card_views, text="Live Broadcast Viewport", height=240, fg_color="#090d16", corner_radius=6)
        self.lbl_live_16_9.pack(fill="both", expand=True, padx=15, pady=5)

        # Controls Bar
        ctrl_bar = ctk.CTkFrame(card_views, fg_color="transparent")
        ctrl_bar.pack(fill="x", padx=15, pady=10)

        self.btn_live_play = ctk.CTkButton(
            ctrl_bar,
            text="▶ START LIVE BROADCAST",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10b981",
            hover_color="#059669",
            width=190,
            height=36,
            command=self.toggle_live_playback
        )
        self.btn_live_play.pack(side="left", padx=(0, 10))

        ctk.CTkButton(ctrl_bar, text="◀ Left Goal", width=95, height=36, command=lambda: self.set_ptz_pos(0.15)).pack(side="left", padx=3)
        ctk.CTkButton(ctrl_bar, text="⚽ Center", width=95, height=36, command=lambda: self.set_ptz_pos(0.50)).pack(side="left", padx=3)
        ctk.CTkButton(ctrl_bar, text="Right Goal ▶", width=95, height=36, command=lambda: self.set_ptz_pos(0.85)).pack(side="left", padx=3)

        self.switch_auto_ptz = ctk.CTkSwitch(ctrl_bar, text="Trajectory Auto Pan/Zoom", font=ctk.CTkFont(size=12))
        self.switch_auto_ptz.select()
        self.switch_auto_ptz.pack(side="right", padx=10)

        # Mobile LAN Stream Server
        card_stream = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_stream.pack(fill="x", pady=5, padx=5)

        ctk.CTkLabel(card_stream, text="LOCAL LAN MOBILE BROADCAST SERVER", font=ctk.CTkFont(size=14, weight="bold"), text_color="#10b981").pack(anchor="w", padx=15, pady=(10, 5))

        stream_row = ctk.CTkFrame(card_stream, fg_color="transparent")
        stream_row.pack(fill="x", padx=15, pady=10)

        self.btn_toggle_stream = ctk.CTkButton(
            stream_row,
            text="📡 START MOBILE STREAM SERVER",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10b981",
            hover_color="#059669",
            height=36,
            command=self.toggle_stream_server
        )
        self.btn_toggle_stream.pack(side="left", padx=(0, 15))

        self.lbl_stream_url = ctk.CTkLabel(
            stream_row,
            text="Server: OFF (Click to broadcast live on local Wi-Fi)",
            font=ctk.CTkFont(size=13),
            text_color="#94a3b8"
        )
        self.lbl_stream_url.pack(side="left")

        self.load_initial_live_frame()
        self.load_live_trajectory_file(silent=True)

    def load_live_trajectory_file(self, silent=False):
        if not silent:
            f = filedialog.askopenfilename(filetypes=[("JSONL / JSON Trajectory Files", "*.jsonl *.json"), ("All Files", "*.*")])
            if f:
                self.entry_live_coords.delete(0, "end")
                self.entry_live_coords.insert(0, f)
        
        path = self.entry_live_coords.get().strip()
        if os.path.exists(path):
            self.live_trajectory = parse_trajectory_file(path)
            self.log_message(f"Loaded {len(self.live_trajectory)} frame coordinates from {path}")
        else:
            self.live_trajectory = {}

    def load_initial_live_frame(self):
        pano_file = self.entry_live_pano.get().strip()
        if not os.path.exists(pano_file):
            pano_file = "check_natural_3200x1080.jpg"
        
        if os.path.exists(pano_file):
            if pano_file.endswith(('.mp4', '.mov', '.mkv')):
                cap = cv2.VideoCapture(pano_file)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.live_current_frame = frame
            else:
                self.live_current_frame = cv2.imread(pano_file)
            self.render_live_studio_views()

    def render_live_studio_views(self):
        if self.live_current_frame is None:
            return

        frame = self.live_current_frame
        h, w = frame.shape[:2]

        cw = self.ptz_canvas.winfo_width() or 800
        ch = 180
        resized_pano = cv2.resize(frame, (cw, ch))
        rgb_pano = cv2.cvtColor(resized_pano, cv2.COLOR_BGR2RGB)
        self.ptz_photo = ImageTk.PhotoImage(Image.fromarray(rgb_pano))

        self.ptz_canvas.delete("all")
        self.ptz_canvas.create_image(0, 0, anchor="nw", image=self.ptz_photo)

        # Outer Tracking Window
        outer_crop_w = float(self.live_ptz_w)
        outer_crop_h = outer_crop_w * 9.0 / 16.0
        if outer_crop_w > w:
            outer_crop_w = float(w)
            outer_crop_h = outer_crop_w * 9.0 / 16.0

        outer_box_w = int(cw * (outer_crop_w / float(w)))
        outer_box_h = int(ch * (outer_crop_h / float(h)))
        center_x = int(self.live_ptz_x * cw)
        
        ox1 = max(5, min(cw - outer_box_w - 5, center_x - outer_box_w // 2))
        ox2 = ox1 + outer_box_w
        oy1 = max(4, (ch - outer_box_h) // 2)
        oy2 = oy1 + outer_box_h

        # Draw Outer Tracking Frame (Cyan)
        self.ptz_canvas.create_rectangle(ox1, oy1, ox2, oy2, outline="#38bdf8", width=1, dash=(4, 4))
        self.ptz_canvas.create_text(ox1 + 10, oy1 + 10, anchor="nw", text="TRACKING DEADBAND (16:9)", fill="#38bdf8", font=("Segoe UI", 9, "bold"))

        # Inner Broadcast Viewport (85% scale - Stabilized Output)
        inner_scale = 0.85
        inner_box_w = int(outer_box_w * inner_scale)
        inner_box_h = int(outer_box_h * inner_scale)
        
        ix1 = ox1 + (outer_box_w - inner_box_w) // 2
        ix2 = ix1 + inner_box_w
        iy1 = oy1 + (outer_box_h - inner_box_h) // 2
        iy2 = iy1 + inner_box_h

        # Draw Inner Broadcast Frame (Emerald Green)
        self.ptz_canvas.create_rectangle(ix1, iy1, ix2, iy2, outline="#10b981", width=3)
        self.ptz_canvas.create_text((ix1 + ix2)//2, iy1 + 14, text="★ 16:9 STABILIZED BROADCAST", fill="#10b981", font=("Segoe UI", 10, "bold"))

        # Extract the inner stabilized broadcast crop
        inner_crop_w = outer_crop_w * inner_scale
        inner_crop_h = inner_crop_w * 9.0 / 16.0

        actual_cx = int(self.live_ptz_x * w)
        cx1 = int(max(0, min(w - inner_crop_w, actual_cx - inner_crop_w // 2)))
        cx2 = int(cx1 + inner_crop_w)
        cy1 = int(max(0, (h - inner_crop_h) // 2))
        cy2 = int(cy1 + inner_crop_h)

        broadcast_crop = frame[cy1:cy2, cx1:cx2]
        
        disp_w = 540
        disp_h = int(disp_w * 9.0 / 16.0)
        resized_broad = cv2.resize(broadcast_crop, (disp_w, disp_h))
        rgb_broad = cv2.cvtColor(resized_broad, cv2.COLOR_BGR2RGB)
        pil_broad = Image.fromarray(rgb_broad)
        ctk_broad = ctk.CTkImage(light_image=pil_broad, dark_image=pil_broad, size=(disp_w, disp_h))
        self.lbl_live_16_9.configure(image=ctk_broad, text="")
        self.lbl_live_16_9.image = ctk_broad

        if self.stream_server and self.stream_server.running:
            self.stream_server.update_frame(broadcast_crop)

    def on_ptz_drag(self, event):
        cw = self.ptz_canvas.winfo_width() or 800
        self.live_ptz_x = max(0.15, min(0.85, event.x / float(cw)))
        self.render_live_studio_views()

    def set_ptz_pos(self, pos):
        self.live_ptz_x = pos
        self.render_live_studio_views()

    def toggle_live_playback(self):
        if self.live_playing:
            self.live_playing = False
            self.btn_live_play.configure(text="▶ START LIVE BROADCAST", fg_color="#10b981", hover_color="#059669")
        else:
            pano_file = self.entry_live_pano.get().strip()
            if not os.path.exists(pano_file):
                messagebox.showerror("File Error", f"Panoramic video not found:\n{pano_file}")
                return
            self.load_live_trajectory_file(silent=True)
            self.live_playing = True
            self.btn_live_play.configure(text="⏸ PAUSE LIVE BROADCAST", fg_color="#f59e0b", hover_color="#d97706")
            t = threading.Thread(target=self.live_playback_loop, args=(pano_file,), daemon=True)
            t.start()

    def parse_time_to_seconds(self, time_str):
        if not time_str: return 0.0
        parts = time_str.strip().split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            else:
                return float(parts[0])
        except Exception:
            return 0.0

    def seek_live_start_time(self):
        time_str = self.entry_live_start.get().strip()
        secs = self.parse_time_to_seconds(time_str)
        pano_file = self.entry_live_pano.get().strip()
        if not os.path.exists(pano_file):
            return

        cap = cv2.VideoCapture(pano_file)
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        target_frame = int(secs * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if ret:
            self.live_current_frame = frame
            # Update trajectory position if available
            if self.live_trajectory and target_frame in self.live_trajectory:
                cx, cw = self.live_trajectory[target_frame]
                self.live_ptz_x = cx / 3200.0
                self.live_ptz_w = cw
            self.render_live_studio_views()
            self.lbl_live_time_display.configure(text=f"Position: {time_str} (Frame {target_frame})")
            self.log_message(f"Jumped to video position {time_str} (Frame {target_frame})")

    def live_playback_loop(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 3200
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

        # Seek to start time if specified
        start_secs = self.parse_time_to_seconds(self.entry_live_start.get().strip())
        frame_idx = int(start_secs * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        curr_cx = self.live_ptz_x * w
        curr_cw = self.live_ptz_w
        curr_vel_x = 0.0

        while self.live_playing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            
            self.live_current_frame = frame
            
            # Replay imported trajectory if enabled with Ultra-Stable Damping
            if self.switch_auto_ptz.get():
                if self.live_trajectory and frame_idx in self.live_trajectory:
                    target_cx, target_cw = self.live_trajectory[frame_idx]
                    
                    # Cosine Deadband Damping
                    deadband = (curr_cw * 0.18)
                    dist = target_cx - curr_cx
                    if abs(dist) < deadband:
                        factor = 0.5 * (1.0 - np.cos(np.pi * (abs(dist) / deadband)))
                        pull = dist * factor
                    else:
                        pull = dist - np.sign(dist) * deadband

                    # Velocity accumulation & clamp
                    curr_vel_x = curr_vel_x * 0.78 + pull * 0.035
                    curr_vel_x = np.clip(curr_vel_x, -14.0, 14.0)

                    curr_cx += curr_vel_x
                    curr_cw = curr_cw * 0.96 + target_cw * 0.04

                    self.live_ptz_x = np.clip(curr_cx / float(w), 0.15, 0.85)
                    self.live_ptz_w = np.clip(curr_cw, 1400, w)
                else:
                    # Fallback auto sway
                    t = time.time() * 0.4
                    self.live_ptz_x = 0.5 + 0.20 * np.sin(t)
                    self.live_ptz_w = float(h * 16.0 / 9.0)

            self.render_live_studio_views()

            # Update time display every 30 frames
            if frame_idx % 30 == 0:
                cur_sec = int(frame_idx / fps)
                mins = cur_sec // 60
                secs = cur_sec % 60
                self.lbl_live_time_display.configure(text=f"Position: {mins:02d}:{secs:02d} (Frame {frame_idx})")

            frame_idx += 1
            time.sleep(0.033) # 30 fps
        cap.release()

    def toggle_stream_server(self):
        from engines.stream_server import LiveStreamServer
        if self.stream_server is None or not self.stream_server.running:
            self.stream_server = LiveStreamServer(port=8080)
            self.stream_server.start()
            ip = self.stream_server.get_local_ip()
            url = f"http://{ip}:8080"
            self.lbl_stream_url.configure(text=f"Live on LAN: {url} (Open in mobile browser)", text_color="#38bdf8")
            self.btn_toggle_stream.configure(text="⏹ STOP STREAM SERVER", fg_color="#ef4444", hover_color="#dc2626")
            self.log_message(f"Mobile live stream server started at {url}")
        else:
            self.stream_server.stop()
            self.lbl_stream_url.configure(text="Server: OFF", text_color="#94a3b8")
            self.btn_toggle_stream.configure(text="📡 START MOBILE STREAM SERVER", fg_color="#10b981", hover_color="#059669")
            self.log_message("Mobile live stream server stopped.")

    # ─── COLLAPSIBLE LOG CONSOLE ───────────────────────────────────────
    def build_log_console(self):
        self.console_frame = ctk.CTkFrame(self, height=140, corner_radius=0, fg_color="#090d16")
        self.console_frame.pack(fill="x", side="bottom")

        bar = ctk.CTkFrame(self.console_frame, height=26, fg_color="#0f172a")
        bar.pack(fill="x")

        ctk.CTkLabel(bar, text="💻 TERMINAL & SYSTEM LOGS", font=ctk.CTkFont(size=11, weight="bold"), text_color="#64748b").pack(side="left", padx=15)
        ctk.CTkButton(bar, text="Clear", width=50, height=20, font=ctk.CTkFont(size=10), command=self.clear_logs).pack(side="right", padx=10)

        self.log_text = ctk.CTkTextbox(self.console_frame, height=110, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#090d16", text_color="#e2e8f0")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 5))
        self.log_message("Zentropy AI Broadcast Control Center Initialized.")

    def log_message(self, text):
        formatted = f"[{time.strftime('%H:%M:%S')}] {text}\n"
        if hasattr(self, 'log_text') and self.log_text:
            try:
                self.log_text.insert("end", formatted)
                self.log_text.see("end")
            except Exception:
                pass
        print(formatted, end="")

    def clear_logs(self):
        self.log_text.delete("1.0", "end")

    def set_status(self, text, color="#94a3b8"):
        self.status_badge.configure(text=f"● {text}", text_color=color)

    # ─── FILE PICKER HELPERS ───────────────────────────────────────────
    def browse_file(self, entry_widget):
        f = filedialog.askopenfilename(filetypes=[("Video / JSON Files", "*.mp4 *.mov *.mkv *.avi *.jsonl *.json"), ("All Files", "*.*")])
        if f:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, f)

    def browse_save_file(self, entry_widget):
        f = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if f:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, f)

    # ─── JOBS & THREAD EXECUTION ───────────────────────────────────────
    def start_stitch_job(self):
        lhs = self.entry_lhs.get().strip()
        rhs = self.entry_rhs.get().strip()
        out = self.entry_pano_out.get().strip()
        st = self.entry_stitch_start.get().strip()
        dur = self.entry_stitch_dur.get().strip()
        dur_val = float(dur) if dur else None

        if not os.path.exists(lhs) or not os.path.exists(rhs):
            messagebox.showerror("File Error", f"Source videos not found:\nLHS: {lhs}\nRHS: {rhs}")
            return

        self.set_status("PROCESSING STITCH", "#10b981")
        self.btn_run_stitch.configure(state="disabled")
        self.btn_abort_stitch.configure(state="normal")
        self.cancel_requested = False

        self.log_message(f"Starting Panoramic Stitching: {lhs} + {rhs} -> {out}")

        def worker():
            from engines.stitch_engine import run_stitching
            try:
                def cb(c, tot, fps, el, eta):
                    pct = c / tot if tot > 0 else 0
                    self.stitch_prog_bar.set(pct)
                    self.lbl_stitch_stats.configure(text=f"Frame: {c}/{tot} ({pct*100:.1f}%) | Speed: {fps:.1f} FPS | Elapsed: {el:.1f}s | ETA: {eta:.1f}s")

                run_stitching(lhs, rhs, out, start_time=st, duration=dur_val, progress_callback=cb, cancel_flag=lambda: self.cancel_requested)
                self.log_message(f"Panoramic Stitching Complete! Saved to {out}")
                self.set_status("COMPLETED", "#10b981")
                self.entry_pano_in.delete(0, "end")
                self.entry_pano_in.insert(0, out)
                self.entry_live_pano.delete(0, "end")
                self.entry_live_pano.insert(0, out)
            except Exception as e:
                self.log_message(f"Error in stitching: {e}")
                self.set_status("ERROR", "#ef4444")
            finally:
                self.btn_run_stitch.configure(state="normal")
                self.btn_abort_stitch.configure(state="disabled")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def start_track_job(self):
        pano = self.entry_pano_in.get().strip()
        out = self.entry_broad_out.get().strip()
        model = self.entry_model.get().strip()
        coords = self.entry_coords.get().strip() or None
        st = self.entry_track_start.get().strip()
        dur = self.entry_track_dur.get().strip()
        dur_val = float(dur) if dur else None
        smooth = float(self.slider_smooth.get())

        if not os.path.exists(pano):
            messagebox.showerror("File Error", f"Panoramic video not found: {pano}")
            return

        self.set_status("AI TRACKING IN PROGRESS", "#3b82f6")
        self.btn_run_track.configure(state="disabled")
        self.btn_run_ball_feed.configure(state="disabled")
        self.btn_export_json.configure(state="disabled")
        self.btn_abort_track.configure(state="normal")
        self.cancel_requested = False

        self.log_message(f"Starting Tracking Broadcast: {pano} -> {out} (Start: {st}, Duration: {dur or 'Full'}, Source: {coords if coords else 'YOLO: ' + model})")

        def worker():
            from engines.tracker_engine import run_tracker_broadcast
            try:
                def cb(c, tot, fps, el, eta):
                    pct = c / tot if tot > 0 else 0
                    self.track_prog_bar.set(pct)
                    self.lbl_track_stats.configure(text=f"Tracked: {c}/{tot} ({pct*100:.1f}%) | Speed: {fps:.1f} FPS | ETA: {eta:.1f}s")

                run_tracker_broadcast(pano, out, model_name=model, coordinates_file=coords,
                                      smoothing=smooth, start_time=st, duration=dur_val,
                                      progress_callback=cb, cancel_flag=lambda: self.cancel_requested)
                self.log_message(f"Broadcast 16:9 Video Complete! Saved to {out}")
                self.set_status("COMPLETED", "#10b981")
                self.load_preview_video(out)
            except Exception as e:
                self.log_message(f"Error in AI tracking: {e}")
                self.set_status("ERROR", "#ef4444")
            finally:
                self.btn_run_track.configure(state="normal")
                self.btn_run_ball_feed.configure(state="normal")
                self.btn_export_json.configure(state="normal")
                self.btn_abort_track.configure(state="disabled")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def start_ball_feed_job(self):
        pano = self.entry_pano_in.get().strip()
        out = "ball_tracking_feed.mp4"
        model = self.entry_model.get().strip()

        if not os.path.exists(pano):
            messagebox.showerror("File Error", f"Panoramic video not found: {pano}")
            return

        self.set_status("GENERATING BALL TRACKING FEED", "#f59e0b")
        self.btn_run_track.configure(state="disabled")
        self.btn_run_ball_feed.configure(state="disabled")
        self.btn_export_json.configure(state="disabled")
        self.btn_abort_track.configure(state="normal")
        self.cancel_requested = False

        self.log_message(f"Starting Annotated Ball Tracking Feed: {pano} -> {out}")

        def worker():
            from generate_ball_tracking_feed import generate_ball_tracking_feed
            try:
                generate_ball_tracking_feed(pano, out, model_name=model)
                self.log_message(f"Ball Tracking Feed Complete! Saved to {out}")
                self.set_status("COMPLETED", "#10b981")
                self.load_preview_video(out)
            except Exception as e:
                self.log_message(f"Error in ball tracking feed: {e}")
                self.set_status("ERROR", "#ef4444")
            finally:
                self.btn_run_track.configure(state="normal")
                self.btn_run_ball_feed.configure(state="normal")
                self.btn_export_json.configure(state="normal")
                self.btn_abort_track.configure(state="disabled")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def start_export_json_job(self):
        pano = self.entry_pano_in.get().strip()
        model = self.entry_model.get().strip()
        out_jsonl = "ball_trajectory_events.jsonl"
        out_json = "ball_trajectory.json"

        if not os.path.exists(pano):
            messagebox.showerror("File Error", f"Panoramic video not found: {pano}")
            return

        self.set_status("EXPORTING TRAJECTORY JSON", "#8b5cf6")
        self.btn_run_track.configure(state="disabled")
        self.btn_run_ball_feed.configure(state="disabled")
        self.btn_export_json.configure(state="disabled")
        self.btn_abort_track.configure(state="normal")
        self.cancel_requested = False

        self.log_message(f"Exporting Ball & Player Trajectory Coordinates: {pano} -> {out_jsonl}")

        def worker():
            from export_ball_trajectory_json import export_tracking_coordinates
            try:
                def cb(c, tot, fps, el, eta):
                    pct = c / tot if tot > 0 else 0
                    self.track_prog_bar.set(pct)
                    self.lbl_track_stats.configure(text=f"Exporting Coordinates: {c}/{tot} ({pct*100:.1f}%) | Speed: {fps:.1f} FPS | ETA: {eta:.1f}s")

                export_tracking_coordinates(pano, out_jsonl, out_json, model_name=model, progress_callback=cb, cancel_flag=lambda: self.cancel_requested)
                self.log_message(f"Trajectory Coordinates Exported Successfully:\n1. {out_jsonl}\n2. {out_json}")
                self.set_status("COMPLETED", "#10b981")
                self.entry_coords.delete(0, "end")
                self.entry_coords.insert(0, out_jsonl)
                self.entry_live_coords.delete(0, "end")
                self.entry_live_coords.insert(0, out_jsonl)
                self.load_live_trajectory_file(silent=True)
            except Exception as e:
                self.log_message(f"Error exporting coordinates: {e}")
                self.set_status("ERROR", "#ef4444")
            finally:
                self.btn_run_track.configure(state="normal")
                self.btn_run_ball_feed.configure(state="normal")
                self.btn_export_json.configure(state="normal")
                self.btn_abort_track.configure(state="disabled")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def abort_job(self):
        self.cancel_requested = True
        self.log_message("Abort requested by user.")
        self.set_status("ABORTING", "#f59e0b")

    # ─── TAB 2 PREVIEW PLAYER ──────────────────────────────────────────
    def load_preview_video(self, video_path):
        if os.path.exists(video_path):
            self.preview_video_path = video_path
            self.toggle_preview_playback()

    def toggle_preview_playback(self):
        if self.preview_playing:
            self.preview_playing = False
            self.btn_play_prev.configure(text="▶ Play Preview")
        else:
            path = getattr(self, 'preview_video_path', self.entry_broad_out.get().strip())
            if not os.path.exists(path):
                path = self.entry_pano_in.get().strip()
            if not os.path.exists(path):
                return
            self.preview_playing = True
            self.btn_play_prev.configure(text="⏸ Pause Preview")
            t = threading.Thread(target=self.preview_loop, args=(path,), daemon=True)
            t.start()

    def preview_loop(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while self.preview_playing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            target_w = 480
            target_h = int(target_w * h / w)
            resized = cv2.resize(rgb, (target_w, target_h))
            pil_prev = Image.fromarray(resized)
            ctk_prev = ctk.CTkImage(light_image=pil_prev, dark_image=pil_prev, size=(target_w, target_h))
            self.lbl_track_img.configure(image=ctk_prev, text="")
            self.lbl_track_img.image = ctk_prev

            time.sleep(0.033)
        cap.release()

    def on_close(self):
        self.config_data = {
            "lhs_video": self.entry_lhs.get(),
            "rhs_video": self.entry_rhs.get(),
            "pano_output": self.entry_pano_out.get(),
            "pano_input": self.entry_pano_in.get(),
            "broadcast_output": self.entry_broad_out.get(),
            "yolo_model": self.entry_model.get(),
            "smoothing": self.slider_smooth.get(),
            "stitch_start": self.entry_stitch_start.get(),
            "stitch_duration": self.entry_stitch_dur.get(),
            "live_pano_source": self.entry_live_pano.get(),
            "live_coords_source": self.entry_live_coords.get()
        }
        self.save_config()
        self.live_playing = False
        self.preview_playing = False
        if self.stream_server:
            self.stream_server.stop()
        self.destroy()

if __name__ == "__main__":
    app = ZentropyControlCenter()
    app.mainloop()
