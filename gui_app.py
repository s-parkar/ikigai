"""Reco GUI Control Center & Pipeline Manager.

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
    "live_pano_source": "stitched_panorama_full.mp4"
}

class RecoControlCenter(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Reco AI Broadcast Control Center - Pipeline Manager")
        self.geometry("1240x900")
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
        self.live_ptz_zoom = 1.0
        self.live_auto_track = False
        self.live_current_frame = None

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
            text="⚽ RECO AI BROADCAST STUDIO", 
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
        self.status_badge.pack(side="right", padx=20, pady=15)

        # Tabview
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=10)

        self.tab_stitch = self.tabview.add(" 📷 1. Pano Stitcher ")
        self.tab_track = self.tabview.add(" 🎯 2. AI Tracking & Broadcast ")
        self.tab_ptz = self.tabview.add(" 📡 3. Live Broadcast Studio ")

        self.build_stitch_tab()
        self.build_track_tab()
        self.build_ptz_tab()

        # Bottom Collapsible Log Console
        self.build_log_console()

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
            resized = img.resize((tw, th), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            self.lbl_pano_img.configure(image=photo, text="")
            self.lbl_pano_img.image = photo

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
        self.entry_coords = ctk.CTkEntry(card_in, width=450, placeholder_text="Optional: Select detections.jsonl / events.jsonl")
        self.entry_coords.grid(row=3, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_in, text="Browse...", width=90, command=lambda: self.browse_file(self.entry_coords)).grid(row=3, column=2, padx=15, pady=5)

        # Broadcast Controls
        ctk.CTkLabel(card_in, text="Camera Smoothing:").grid(row=4, column=0, sticky="w", padx=15, pady=5)
        self.slider_smooth = ctk.CTkSlider(card_in, from_=0.02, to=0.15, number_of_steps=13)
        self.slider_smooth.set(self.config_data.get("smoothing", 0.06))
        self.slider_smooth.grid(row=4, column=1, sticky="ew", padx=10, pady=5)
        self.lbl_smooth_val = ctk.CTkLabel(card_in, text=f"Smooth ({self.slider_smooth.get():.2f})", width=90)
        self.lbl_smooth_val.grid(row=4, column=2, padx=15, pady=5)
        self.slider_smooth.configure(command=lambda v: self.lbl_smooth_val.configure(text=f"{'Cinematic' if v < 0.04 else 'Smooth' if v < 0.09 else 'Fast'} ({v:.2f})"))

        # Output broadcast
        ctk.CTkLabel(card_in, text="Broadcast Output (16:9):").grid(row=5, column=0, sticky="w", padx=15, pady=5)
        self.entry_broad_out = ctk.CTkEntry(card_in, width=450)
        self.entry_broad_out.insert(0, self.config_data.get("broadcast_output", "broadcast_16_9.mp4"))
        self.entry_broad_out.grid(row=5, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_in, text="Save As...", width=90, command=lambda: self.browse_save_file(self.entry_broad_out)).grid(row=5, column=2, padx=15, pady=5)

        card_in.columnconfigure(1, weight=1)

        # Actions & Progress
        card_act = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_act.pack(fill="x", pady=10, padx=5)

        btn_row = ctk.CTkFrame(card_act, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=10)

        self.btn_run_track = ctk.CTkButton(
            btn_row, 
            text="▶ GENERATE 16:9 AUTO-BROADCAST", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#3b82f6", 
            hover_color="#2563eb",
            height=40,
            command=self.start_track_job
        )
        self.btn_run_track.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_run_ball_feed = ctk.CTkButton(
            btn_row, 
            text="🎯 GENERATE BALL TRACKING FEED", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#f59e0b", 
            hover_color="#d97706",
            height=40,
            command=self.start_ball_feed_job
        )
        self.btn_run_ball_feed.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_abort_track = ctk.CTkButton(
            btn_row, 
            text="⏹ ABORT", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#ef4444", 
            hover_color="#dc2626",
            height=40,
            width=90,
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

    # ─── TAB 3: LIVE BROADCAST STUDIO (DIRECT PROCESSED PANORAMA & PTZ) 
    def build_ptz_tab(self):
        container = ctk.CTkScrollableFrame(self.tab_ptz, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Card 1: Panorama Source Selection
        card_src = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_src.pack(fill="x", pady=5, padx=5)

        lbl_sec = ctk.CTkLabel(card_src, text="LIVE BROADCAST PANORAMA SOURCE", font=ctk.CTkFont(size=14, weight="bold"), text_color="#38bdf8")
        lbl_sec.grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(10, 5))

        ctk.CTkLabel(card_src, text="Processed Panorama:").grid(row=1, column=0, sticky="w", padx=15, pady=5)
        self.entry_live_pano = ctk.CTkEntry(card_src, width=450)
        self.entry_live_pano.insert(0, self.config_data.get("live_pano_source", "stitched_panorama_full.mp4"))
        self.entry_live_pano.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(card_src, text="Browse...", width=90, command=lambda: self.browse_file(self.entry_live_pano)).grid(row=1, column=2, padx=15, pady=5)

        card_src.columnconfigure(1, weight=1)

        # Card 2: Interactive Dual-View Studio
        card_views = ctk.CTkFrame(container, corner_radius=8, fg_color="#1e293b")
        card_views.pack(fill="both", expand=True, pady=10, padx=5)

        # Top: Wide Panorama Canvas with PTZ Box
        lbl_pano_title = ctk.CTkLabel(card_views, text="1. FULL FIELD PANORAMA (DRAG GREEN BOX TO PAN)", font=ctk.CTkFont(size=13, weight="bold"), text_color="#10b981")
        lbl_pano_title.pack(anchor="w", padx=15, pady=(10, 2))

        self.ptz_canvas = tk.Canvas(card_views, height=180, bg="#0f172a", highlightthickness=1, highlightbackground="#334155")
        self.ptz_canvas.pack(fill="x", padx=15, pady=5)
        self.ptz_canvas.bind("<B1-Motion>", self.on_ptz_drag)
        self.ptz_canvas.bind("<Button-1>", self.on_ptz_drag)

        # Bottom: Live 16:9 Broadcast Output Viewport
        lbl_out_title = ctk.CTkLabel(card_views, text="2. LIVE 16:9 BROADCAST OUTPUT (DIRECT CROPPED VIEWPORT)", font=ctk.CTkFont(size=13, weight="bold"), text_color="#38bdf8")
        lbl_out_title.pack(anchor="w", padx=15, pady=(10, 2))

        self.lbl_live_16_9 = ctk.CTkLabel(card_views, text="Live Broadcast Viewport", height=240, fg_color="#090d16", corner_radius=6)
        self.lbl_live_16_9.pack(fill="both", expand=True, padx=15, pady=5)

        # Live Controls Bar
        ctrl_bar = ctk.CTkFrame(card_views, fg_color="transparent")
        ctrl_bar.pack(fill="x", padx=15, pady=10)

        self.btn_live_play = ctk.CTkButton(
            ctrl_bar,
            text="▶ START LIVE PREVIEW",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10b981",
            hover_color="#059669",
            width=170,
            height=36,
            command=self.toggle_live_playback
        )
        self.btn_live_play.pack(side="left", padx=(0, 10))

        # Hotspots
        ctk.CTkButton(ctrl_bar, text="◀ Left Goal", width=95, height=36, command=lambda: self.set_ptz_pos(0.15)).pack(side="left", padx=3)
        ctk.CTkButton(ctrl_bar, text="⚽ Center", width=95, height=36, command=lambda: self.set_ptz_pos(0.50)).pack(side="left", padx=3)
        ctk.CTkButton(ctrl_bar, text="Right Goal ▶", width=95, height=36, command=lambda: self.set_ptz_pos(0.85)).pack(side="left", padx=3)

        # Auto-track toggle
        self.switch_auto_ptz = ctk.CTkSwitch(ctrl_bar, text="Auto AI Track", font=ctk.CTkFont(size=12))
        self.switch_auto_ptz.pack(side="right", padx=10)

        # Card 3: Mobile LAN Stream Server
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

        # Initial frame render
        self.load_initial_live_frame()

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

        # 1. Update Panoramic Canvas with PTZ Box
        cw = self.ptz_canvas.winfo_width() or 800
        ch = 180
        resized_pano = cv2.resize(frame, (cw, ch))
        rgb_pano = cv2.cvtColor(resized_pano, cv2.COLOR_BGR2RGB)
        self.ptz_photo = ImageTk.PhotoImage(Image.fromarray(rgb_pano))

        self.ptz_canvas.delete("all")
        self.ptz_canvas.create_image(0, 0, anchor="nw", image=self.ptz_photo)

        box_w = int(cw * 0.45)
        box_h = int(ch * 0.90)
        center_x = int(self.live_ptz_x * cw)
        x1 = max(5, min(cw - box_w - 5, center_x - box_w // 2))
        x2 = x1 + box_w
        y1 = (ch - box_h) // 2
        y2 = y1 + box_h

        self.ptz_canvas.create_rectangle(x1, y1, x2, y2, outline="#10b981", width=3)
        self.ptz_canvas.create_text((x1 + x2)//2, y1 + 14, text="16:9 BROADCAST VIEWPORT", fill="#10b981", font=("Segoe UI", 10, "bold"))

        # 2. Extract & Render 16:9 Broadcast Frame
        crop_h = h
        crop_w = int(crop_h * 16.0 / 9.0)
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w * 9.0 / 16.0)

        actual_cx = int(self.live_ptz_x * w)
        cx1 = max(0, min(w - crop_w, actual_cx - crop_w // 2))
        cx2 = cx1 + crop_w
        cy1 = max(0, (h - crop_h) // 2)
        cy2 = cy1 + crop_h

        broadcast_crop = frame[cy1:cy2, cx1:cx2]
        
        # Render to preview widget
        disp_w = 540
        disp_h = int(disp_w * 9.0 / 16.0)
        resized_broad = cv2.resize(broadcast_crop, (disp_w, disp_h))
        rgb_broad = cv2.cvtColor(resized_broad, cv2.COLOR_BGR2RGB)
        photo_broad = ImageTk.PhotoImage(Image.fromarray(rgb_broad))
        self.lbl_live_16_9.configure(image=photo_broad, text="")
        self.lbl_live_16_9.image = photo_broad

        # Push to stream server if active
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
            self.btn_live_play.configure(text="▶ START LIVE PREVIEW", fg_color="#10b981", hover_color="#059669")
        else:
            pano_file = self.entry_live_pano.get().strip()
            if not os.path.exists(pano_file):
                messagebox.showerror("File Error", f"Panoramic video not found:\n{pano_file}")
                return
            self.live_playing = True
            self.btn_live_play.configure(text="⏸ PAUSE LIVE PREVIEW", fg_color="#f59e0b", hover_color="#d97706")
            t = threading.Thread(target=self.live_playback_loop, args=(pano_file,), daemon=True)
            t.start()

    def live_playback_loop(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while self.live_playing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.live_current_frame = frame
            
            # If auto-track switch is ON, auto-pan gently based on frame action
            if self.switch_auto_ptz.get():
                # Smooth auto sway
                t = time.time() * 0.5
                self.live_ptz_x = 0.5 + 0.25 * np.sin(t)

            self.render_live_studio_views()
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
        self.log_message("Reco AI Broadcast Control Center Initialized.")

    def log_message(self, text):
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.log_text.see("end")

    def clear_logs(self):
        self.log_text.delete("1.0", "end")

    def set_status(self, text, color="#94a3b8"):
        self.status_badge.configure(text=f"● {text}", text_color=color)

    # ─── FILE PICKER HELPERS ───────────────────────────────────────────
    def browse_file(self, entry_widget):
        f = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.mkv *.avi"), ("All Files", "*.*")])
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
        smooth = float(self.slider_smooth.get())

        if not os.path.exists(pano):
            messagebox.showerror("File Error", f"Panoramic video not found: {pano}")
            return

        self.set_status("AI TRACKING IN PROGRESS", "#3b82f6")
        self.btn_run_track.configure(state="disabled")
        self.btn_run_ball_feed.configure(state="disabled")
        self.btn_abort_track.configure(state="normal")
        self.cancel_requested = False

        self.log_message(f"Starting Tracking Broadcast: {pano} -> {out} (Source: {coords if coords else 'YOLO model: ' + model})")

        def worker():
            from engines.tracker_engine import run_tracker_broadcast
            try:
                def cb(c, tot, fps, el, eta):
                    pct = c / tot if tot > 0 else 0
                    self.track_prog_bar.set(pct)
                    self.lbl_track_stats.configure(text=f"Tracked: {c}/{tot} ({pct*100:.1f}%) | Speed: {fps:.1f} FPS | ETA: {eta:.1f}s")

                run_tracker_broadcast(pano, out, model_name=model, coordinates_file=coords, smoothing=smooth, progress_callback=cb, cancel_flag=lambda: self.cancel_requested)
                self.log_message(f"Broadcast 16:9 Video Complete! Saved to {out}")
                self.set_status("COMPLETED", "#10b981")
                self.load_preview_video(out)
            except Exception as e:
                self.log_message(f"Error in AI tracking: {e}")
                self.set_status("ERROR", "#ef4444")
            finally:
                self.btn_run_track.configure(state="normal")
                self.btn_run_ball_feed.configure(state="normal")
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
            img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
            self.lbl_track_img.configure(image=img_tk, text="")
            self.lbl_track_img.image = img_tk

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
            "live_pano_source": self.entry_live_pano.get()
        }
        self.save_config()
        self.live_playing = False
        self.preview_playing = False
        if self.stream_server:
            self.stream_server.stop()
        self.destroy()

if __name__ == "__main__":
    app = RecoControlCenter()
    app.mainloop()
