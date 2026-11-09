import time
from pathlib import Path
import cv2
import numpy as np
import config
from tracker.centroid_tracker import CentroidTracker


class ParticleCounter:
    def __init__(self, video_path, scale=None, show_preview=None):
        self.video_path = video_path
        self.scale = config.SCALE_FACTOR if scale is None else scale
        self.show_preview = config.SHOW_PREVIEW if show_preview is None else show_preview
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.orig_w * self.scale)
        self.h = int(self.orig_h * self.scale)
        self.count_soybean = 0
        self.count_black_bean = 0
        self.frame_count = 0
        self.process_times = []
        self.tracker = CentroidTracker(config.TRACK_MAX_DISAPPEARED,
                                       config.TRACK_MAX_DISTANCE)
        self.writer = None
        self.bg_gray = None
        self.roi = (0, 0, self.w, self.h)
        self.counting_line_y = int(self.h * config.COUNTING_LINE_Y)
        self.last_counted_frame = {}

    def _resize(self, frame):
        if self.scale != 1.0:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return frame

    def _init_background(self):
        print(f"Initializing background from {config.INIT_FRAMES} frames...")
        samples = []
        for _ in range(config.INIT_FRAMES):
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = self._resize(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append(gray)
        if not samples:
            raise RuntimeError("Could not read frames for background initialization.")
        self.bg_gray = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
        self.frame_count = len(samples)
        print(f"Background ready: {self.w}x{self.h}, {self.fps:.1f} FPS")

    def _init_writer(self):
        if not config.SAVE_OUTPUT_VIDEO:
            return
        out = Path(config.OUTPUT_VIDEO_FILE)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                      self.fps, (self.w, self.h))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create output video: {out}")

    def _foreground_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.bg_gray, gray)
        _, mask = cv2.threshold(diff, config.BG_DIFF_THRESHOLD,
                                config.THRESH_BINARY, cv2.THRESH_BINARY)
        k = max(1, int(config.MORPH_KERNEL_SIZE))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                iterations=config.MORPH_CLOSE_ITERATIONS)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                iterations=config.MORPH_OPEN_ITERATIONS)
        return mask

    def _classify_particle(self, frame, x, y, w, h):
        pad = 2
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(self.w, x+w+pad), min(self.h, y+h+pad)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return "unknown"
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mean_l, mean_a, mean_b = np.mean(lab, axis=(0, 1))
        dark_ratio = float(np.mean(gray < config.BLACK_BEAN_DARK_PIXEL_THRESHOLD))

        # Yellow/brown soybean color has a relatively high LAB b channel.
        if mean_b >= config.SOYBEAN_LAB_B_MIN and mean_l >= config.SOYBEAN_LAB_L_MIN:
            return "soybean"
        if (mean_l <= config.BLACK_BEAN_LAB_L_MAX or
                np.mean(gray) <= config.BLACK_BEAN_MEAN_GRAY_MAX or
                dark_ratio >= config.BLACK_BEAN_DARK_PIXEL_RATIO):
            return "black_bean"
        return "unknown"

    def detect_particles(self, frame):
        mask = self._foreground_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.MIN_CONTOUR_AREA or area > config.MAX_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / max(1, min(w, h))
            if aspect < config.MIN_ASPECT_RATIO or aspect > config.MAX_ASPECT_RATIO:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-9)
            if circularity < config.MIN_CIRCULARITY:
                continue
            cx, cy = x + w // 2, y + h // 2
            label = self._classify_particle(frame, x, y, w, h)
            detections.append({"centroid": (cx, cy), "bbox": (x, y, w, h),
                               "class": label, "area": area})
        return detections

    def draw_results(self, frame, tracks):
        overlay = frame.copy()
        cv2.line(overlay, (0, self.counting_line_y), (self.w, self.counting_line_y),
                 config.COLOR_COUNT_LINE, 2)
        cv2.putText(overlay, "COUNTING LINE", (8, self.counting_line_y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_COUNT_LINE, 1)
        for track in tracks:
            if track.disappeared > 0:
                continue
            x, y, bw, bh = track.bbox
            if track.stable_class == "soybean":
                color, label = config.COLOR_SOYBEAN, "Soy"
            elif track.stable_class == "black_bean":
                color, label = config.COLOR_BLACK_BEAN, "Black"
            else:
                color, label = config.COLOR_UNKNOWN, "?"
            cv2.rectangle(overlay, (x, y), (x+bw, y+bh), color, 2)
            text = f"{label} #{track.object_id}"
            cv2.putText(overlay, text, (x, max(15, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        self._draw_hud(frame)
        return frame

    def _draw_hud(self, frame):
        x1, y1, x2, y2 = 5, 5, self.w-5, 52
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_HUD_BG, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_HUD_BORDER, 1)
        cv2.putText(frame, f"Soybean: {self.count_soybean}",
                    (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    config.COLOR_SOYBEAN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Black Bean: {self.count_black_bean}",
                    (max(140, self.w//3), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    config.COLOR_BLACK_BEAN, 1, cv2.LINE_AA)
        total = self.count_soybean + self.count_black_bean
        cv2.putText(frame, f"Total: {total}",
                    (max(260, 2*self.w//3), 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self.fps:.1f} | Frame: {self.frame_count}",
                    (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (220, 220, 220), 1, cv2.LINE_AA)

    def _count_crossings(self, tracks):
        for track in tracks:
            if track.disappeared > 0 or track.counted:
                continue
            old_y, new_y = track.previous_y, track.centroid[1]
            crossed = ((old_y < self.counting_line_y <= new_y) or
                       (old_y > self.counting_line_y >= new_y))
            if not crossed:
                continue
            last = self.last_counted_frame.get(track.object_id, -10**9)
            if self.frame_count - last < config.COUNTING_COOLDOWN_FRAMES:
                continue
            if track.stable_class == "soybean":
                self.count_soybean += 1
            elif track.stable_class == "black_bean":
                self.count_black_bean += 1
            else:
                continue
            track.counted = True
            self.last_counted_frame[track.object_id] = self.frame_count

    def run(self):
        try:
            self._init_background()
            self._init_writer()
            print("Starting particle counting & classification...")
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                t0 = time.time()
                frame = self._resize(frame)
                self.frame_count += 1

                detections = self.detect_particles(frame)
                tracks = self.tracker.update(detections)
                self._count_crossings(tracks)
                display = self.draw_results(frame.copy(), tracks)

                elapsed = time.time() - t0
                self.process_times.append(elapsed)
                if self.writer is not None:
                    self.writer.write(display)
                if self.show_preview:
                    cv2.imshow("Particle Counting & Classification", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("p"):
                        cv2.waitKey(0)

                if self.frame_count % 50 == 0:
                    total = self.count_soybean + self.count_black_bean
                    print(f"Frame {self.frame_count}/{self.total_frames} | "
                          f"Soybean={self.count_soybean} BlackBean={self.count_black_bean} "
                          f"Total={total}")
        finally:
            self.cap.release()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()
        self.print_summary()

    def print_summary(self):
        total = self.count_soybean + self.count_black_bean
        avg = float(np.mean(self.process_times)) if self.process_times else 0.0
        avg_fps = 1.0 / avg if avg > 0 else 0.0
        print("\n" + "=" * 38)
        print("PARTICLE COUNTING RESULTS")
        print("=" * 38)
        print(f"Soybean count : {self.count_soybean}")
        print(f"Black bean count: {self.count_black_bean}")
        print(f"Total counted : {total}")
        print(f"Processed frames: {self.frame_count}")
        print(f"Average processing FPS: {avg_fps:.2f}")
        if self.writer is not None:
            print(f"Output video: {config.OUTPUT_VIDEO_FILE}")
