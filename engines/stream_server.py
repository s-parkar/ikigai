"""Lightweight HTTP MJPEG Streaming Server for LAN & Mobile Broadcast Preview."""

import socket
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2

class StreamingHandler(BaseHTTPRequestHandler):
    current_frame_bytes = None
    lock = threading.Lock()

    @classmethod
    def set_frame(cls, frame_bgr):
        if frame_bgr is None:
            return
        ret, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with cls.lock:
                cls.current_frame_bytes = jpeg.tobytes()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Zentropy Live Broadcast Preview</title>
                <style>
                    body { background: #0f172a; color: #f8fafc; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
                    h2 { color: #10b981; margin-bottom: 8px; }
                    .badge { background: #1e293b; padding: 4px 12px; border-radius: 999px; font-size: 14px; border: 1px solid #334155; }
                    img { max-width: 100%; height: auto; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); margin-top: 15px; }
                </style>
            </head>
            <body>
                <h2>Zentropy AI Broadcast Live Stream</h2>
                <span class="badge">Live 16:9 Virtual PTZ Feed</span><br>
                <img src="/stream.mjpg" />
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with StreamingHandler.lock:
                        frame = StreamingHandler.current_frame_bytes
                    if frame is not None:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.04) # ~25 fps
            except Exception:
                pass
        else:
            self.send_error(404)

class LiveStreamServer:
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.running = False

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def start(self):
        if self.running:
            return
        self.server = HTTPServer((self.host, self.port), StreamingHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.running = True
        print(f"Streaming server running at http://{self.get_local_ip()}:{self.port}/")

    def update_frame(self, frame_bgr):
        StreamingHandler.set_frame(frame_bgr)

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
