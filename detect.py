"""
============================================================
 Smart Object Detection — ESP32-CAM + YOLOv8
============================================================
 Connects to ESP32-CAM MJPEG stream, runs YOLOv8 detection,
 draws bounding boxes, and sends detected object names to
 the ESP32 OLED display via HTTP.

 Usage:
   python detect.py --ip <ESP32_IP>
   python detect.py --ip 192.168.1.100 --model yolov8s.pt
   python detect.py --ip 192.168.1.100 --no-send

 Controls:
   ESC — quit
   S   — screenshot
   P   — pause/resume
============================================================
"""

import cv2
import numpy as np
import argparse
import time
import requests
import http.client
import sys
import os
from threading import Thread, Lock
from queue import Queue, Empty
from datetime import datetime
from urllib.parse import quote

# Fix Windows console encoding for emoji/unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ─── CLI Arguments ───────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ESP32-CAM YOLOv8 Object Detection")
    p.add_argument("--ip", type=str, default="10.87.197.177",
                   help="ESP32-CAM IP address")
    p.add_argument("--stream-port", type=int, default=81,
                   help="Stream port (default: 81)")
    p.add_argument("--oled-port", type=int, default=82,
                   help="OLED object endpoint port (default: 82)")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="YOLOv8 model file (default: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.45,
                   help="Confidence threshold (default: 0.45)")
    p.add_argument("--width", type=int, default=640,
                   help="Processing width")
    p.add_argument("--height", type=int, default=480,
                   help="Processing height")
    p.add_argument("--no-send", action="store_true",
                   help="Don't send object names to ESP32 OLED")
    p.add_argument("--save-dir", type=str, default="screenshots",
                   help="Screenshot save directory")
    return p.parse_args()


# ─── MJPEG Stream Reader (using http.client) ─────────────────
class MJPEGStreamReader(Thread):
    """
    Reads MJPEG stream from ESP32-CAM using http.client.HTTPConnection.
    
    Why http.client instead of urllib?
    urllib.request.urlopen(timeout=N) permanently corrupts the socket after
    a single timeout — all future reads fail with 'cannot read from timed 
    out object'. http.client gives direct socket control.
    """
    
    def __init__(self, ip, port=81, path="/stream"):
        super().__init__(daemon=True)
        self.ip = ip
        self.port = port
        self.path = path
        self.frame = None
        self.lock = Lock()
        self.running = True
        self.connected = False
        self.frame_count = 0
        self.errors = 0
    
    def get_frame(self):
        with self.lock:
            return self.frame
    
    def run(self):
        while self.running:
            try:
                self._read_stream()
            except Exception as e:
                self.connected = False
                self.errors += 1
                err_name = type(e).__name__
                print(f"⚠ Stream error: {err_name}: {e}")
                if self.running:
                    wait = min(2 * self.errors, 10)
                    print(f"  Reconnecting in {wait}s... (attempt {self.errors})")
                    time.sleep(wait)
    
    def _read_stream(self):
        print(f"📡 Connecting to http://{self.ip}:{self.port}{self.path}")
        
        conn = http.client.HTTPConnection(self.ip, self.port, timeout=30)
        conn.request("GET", self.path)
        resp = conn.getresponse()
        
        if resp.status != 200:
            print(f"❌ HTTP {resp.status} {resp.reason}")
            conn.close()
            return
        
        print("✅ Stream connected!")
        self.connected = True
        self.errors = 0
        
        buf = b""
        
        while self.running:
            try:
                chunk = resp.read(4096)
            except http.client.IncompleteRead as e:
                chunk = e.partial
            except Exception:
                break
            
            if not chunk:
                print("⚠ Stream ended (empty read)")
                break
            
            buf += chunk
            
            # Extract JPEG frames using SOI (FFD8) and EOI (FFD9) markers
            while True:
                soi = buf.find(b'\xff\xd8')
                eoi = buf.find(b'\xff\xd9', soi + 2) if soi != -1 else -1
                
                if soi == -1 or eoi == -1:
                    # Keep only data from last SOI marker to avoid unbounded growth
                    if soi != -1:
                        buf = buf[soi:]
                    elif len(buf) > 100000:
                        buf = buf[-50000:]
                    break
                
                jpeg_data = buf[soi:eoi + 2]
                buf = buf[eoi + 2:]
                
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                
                if frame is not None:
                    with self.lock:
                        self.frame = frame
                    self.frame_count += 1
        
        self.connected = False
        try:
            conn.close()
        except Exception:
            pass
    
    def stop(self):
        self.running = False


# ─── OLED Sender ─────────────────────────────────────────────
class OLEDSender:
    """Sends detected object names to ESP32 OLED via HTTP GET."""
    
    def __init__(self, ip, port=82):
        self.ip = ip
        self.port = port
        self.last_sent = ""
        self.last_time = 0
        self.min_interval = 1.0  # Don't spam — max 1 update/sec
    
    def send(self, object_names):
        """Send list of object names to ESP32 OLED."""
        now = time.time()
        names_str = ",".join(object_names)
        
        # Skip if same objects and within interval
        if names_str == self.last_sent and (now - self.last_time) < self.min_interval:
            return
        
        try:
            url = f"http://{self.ip}:{self.port}/object?name={quote(names_str)}"
            requests.get(url, timeout=2)
            self.last_sent = names_str
            self.last_time = now
        except Exception:
            pass  # Non-critical — don't crash detection over OLED


# ─── Pretty Bounding Boxes ──────────────────────────────────
# Color palette for different classes
COLORS = [
    (255, 107, 107), (78, 205, 196), (255, 230, 109),
    (162, 155, 254), (0, 210, 211), (255, 154, 162),
    (181, 234, 215), (199, 206, 234), (255, 183, 178),
    (245, 205, 121), (169, 222, 249), (255, 179, 186),
]

def draw_detection(frame, box, label, conf, color):
    """Draw a styled bounding box with label."""
    x1, y1, x2, y2 = map(int, box)
    
    # Main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Corner accents (top-left and bottom-right)
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)
    
    # Label background
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, text, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_hud(frame, fps, obj_count, paused, sending):
    """Draw heads-up display overlay."""
    h, w = frame.shape[:2]
    
    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Object count
    cv2.putText(frame, f"Objects: {obj_count}", (w // 2 - 50, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Status indicators
    status = "PAUSED" if paused else ("OLED" if sending else "LOCAL")
    color = (0, 165, 255) if paused else ((0, 255, 0) if sending else (200, 200, 200))
    cv2.putText(frame, status, (w - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_splash(width, height, msg="Connecting to ESP32-CAM..."):
    """Create a splash screen while waiting for stream."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Dark gradient background
    for y in range(height):
        val = int(30 + 20 * (y / height))
        frame[y, :] = (val, val // 2, val // 3)
    
    # Title
    cv2.putText(frame, "SMART OBJECT DETECTION", (width // 2 - 200, height // 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    
    # Status message
    dots = "." * (int(time.time() * 2) % 4)
    cv2.putText(frame, msg + dots, (width // 2 - 180, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    # Controls hint
    cv2.putText(frame, "ESC=Quit  S=Screenshot  P=Pause", (width // 2 - 180, height * 3 // 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return frame


# ─── Main Detection Loop ────────────────────────────────────
def main():
    args = parse_args()
    
    stream_url = f"http://{args.ip}:{args.stream_port}/stream"
    
    print("=" * 50)
    print("  Smart Object Detection — ESP32-CAM + YOLOv8")
    print("=" * 50)
    print(f"  Stream URL : {stream_url}")
    print(f"  Model      : {args.model}")
    print(f"  Confidence : {args.conf}")
    print(f"  Resolution : {args.width}x{args.height}")
    print(f"  Send OLED  : {'No' if args.no_send else 'Yes'}")
    print("=" * 50)
    print()
    
    # Load YOLO model
    print(f"📦 Loading model: {args.model}")
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        class_names = model.names
        print(f"✅ Model loaded — {len(class_names)} classes")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Check device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using {device.upper()} for inference")
    
    # Start stream reader
    reader = MJPEGStreamReader(args.ip, args.stream_port)
    reader.start()
    
    # OLED sender
    oled = None if args.no_send else OLEDSender(args.ip, args.oled_port)
    
    # Create screenshot directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Wait for first frame
    print(f"\n⏳ Waiting for first frame from ESP32-CAM...")
    
    window_name = "Smart Object Detection — ESP32-CAM + YOLOv8"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)
    
    paused = False
    fps = 0.0
    fps_start = time.time()
    fps_frames = 0
    last_objects = []
    
    while True:
        frame = reader.get_frame()
        
        if frame is None:
            # Show splash screen while waiting
            splash = draw_splash(args.width, args.height)
            cv2.imshow(window_name, splash)
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC
                break
            continue
        
        if fps_frames == 0:
            print("✅ Receiving frames! Starting detection...\n")
        
        # Resize for processing
        frame = cv2.resize(frame, (args.width, args.height))
        
        if not paused:
            # Run YOLO detection
            results = model(frame, conf=args.conf, verbose=False)
            
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    label = class_names[cls_id]
                    coords = box.xyxy[0].cpu().numpy()
                    color = COLORS[cls_id % len(COLORS)]
                    
                    draw_detection(frame, coords, label, conf_val, color)
                    detected_objects.append(label)
            
            # Send to OLED (unique objects only)
            unique_objects = list(dict.fromkeys(detected_objects))
            if oled and unique_objects:
                oled.send(unique_objects)
            last_objects = unique_objects
        
        # Calculate FPS
        fps_frames += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = fps_frames / elapsed
            fps_frames = 0
            fps_start = time.time()
        
        # Draw HUD
        draw_hud(frame, fps, len(last_objects), paused, oled is not None)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\n👋 Exiting...")
            break
        elif key == ord('s') or key == ord('S'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(args.save_dir, f"detection_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"📸 Screenshot saved: {path}")
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print(f"{'⏸ Paused' if paused else '▶ Resumed'}")
    
    # Cleanup
    reader.stop()
    cv2.destroyAllWindows()
    print("✅ Done!")


if __name__ == "__main__":
    main()
