"""
Virtual Air Canvas - Hand Gesture Drawing App (Pro UI + Shape/Text Snap Edition)
---------------------------------------------------------------------------------
Draw on screen using your index finger in the air, tracked via webcam.

Controls (Gestures):
- Only INDEX finger up      -> Draw mode
- INDEX + MIDDLE finger up  -> Selection mode (move without drawing, pick color)
- All fingers down (fist)   -> Idle

Keyboard:
- '1' -> Freehand mode (normal drawing, no auto-correction)
- '2' -> Shape mode (rough shapes snap to perfect line / rectangle / triangle / circle)
- '3' -> Text mode (finish a word, press 'w' to OCR it into clean printed text)
- 'w' -> (Text mode only) recognize the last-drawn word and replace it with printed text
- 'c' -> Clear canvas
- 's' -> Save drawing as image
- 'q' -> Quit

Requirements:
    pip install opencv-python mediapipe numpy
    (Optional, for Text mode) pip install pytesseract
    (Optional, for Text mode) Install the Tesseract-OCR engine itself:
        Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math

# Optional OCR support -----------------------------------------------------
try:
    import pytesseract
    OCR_AVAILABLE = True
    # If Tesseract isn't on PATH on Windows, uncomment and set the path below:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    OCR_AVAILABLE = False

# ---------------------------
# 1. Setup
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

CAM_WIDTH, CAM_HEIGHT = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

canvas = None          # persistent, committed drawing
stroke_canvas = None   # temporary layer for the stroke currently being drawn

# ---------------------------
# Theme
# ---------------------------
THEME_BG        = (35, 25, 20)
THEME_ACCENT    = (255, 170, 60)
THEME_TEXT      = (235, 235, 235)
THEME_MUTED     = (150, 150, 150)

TOPBAR_HEIGHT = 110
SWATCH_RADIUS = 26

PALETTE_COLORS = [
    ("Red",    (0, 0, 255)),
    ("Green",  (0, 255, 0)),
    ("Blue",   (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Purple", (255, 0, 255)),
    ("Cyan",   (255, 255, 0)),
    ("Orange", (0, 140, 255)),
    ("White",  (255, 255, 255)),
    ("Eraser", (90, 90, 90)),
]

current_color = (0, 0, 255)
brush_thickness = 8
eraser_thickness = 40

prev_x, prev_y = 0, 0
smooth_x, smooth_y = 0, 0
missed_frames = 0
MAX_MISSED_FRAMES = 5
SMOOTHING = 0.7

prev_time = 0

APP_MODE = "FREEHAND"      # FREEHAND | SHAPE | TEXT
stroke_points = []          # points of the stroke currently being drawn
was_drawing = False          # tracks draw-mode transitions to detect stroke end
text_region_pts = []          # accumulated bounding points for the current word (TEXT mode)


# ---------------------------
# 2. UI helpers
# ---------------------------
def draw_glass_panel(img, pt1, pt2, color, alpha=0.55, radius=18):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def get_palette_layout(width):
    n = len(PALETTE_COLORS)
    slot_w = width / n
    layout = []
    for i, (name, color) in enumerate(PALETTE_COLORS):
        cx = int(slot_w * i + slot_w / 2)
        cy = TOPBAR_HEIGHT // 2
        layout.append((name, color, cx, cy, slot_w))
    return layout


# ---------------------------
# 3. Hand landmark helper
# ---------------------------
def fingers_up(hand_landmarks, img_shape):
    h, w, _ = img_shape
    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    fingers = []
    fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)

    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip, pip in zip(tip_ids, pip_ids):
        fingers.append(1 if lm_list[tip][1] < lm_list[pip][1] else 0)

    return fingers, lm_list


# ---------------------------
# 4. Shape recognition ("Shape Mode")
# ---------------------------
def snap_to_perfect_shape(points, color, thickness, target_canvas):
    """Given the raw points of one finished stroke, detect the closest
    basic shape (line / rectangle / triangle / circle) and draw a clean
    version of it onto target_canvas. Falls back to the original freehand
    stroke if nothing matches confidently."""
    if len(points) < 6:
        return False  # too short to be a meaningful shape

    pts = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    if perimeter == 0:
        return False

    # --- Check for a straight line first (very low area relative to length) ---
    x, y, w, h = cv2.boundingRect(pts)
    diag = math.hypot(w, h)
    if area < 0.04 * (diag ** 2) and diag > 40:
        vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        # project extreme points onto the fitted line direction
        projections = [((p[0] - cx) * vx + (p[1] - cy) * vy) for p in points]
        t_min, t_max = min(projections), max(projections)
        p1 = (int(cx + vx * t_min), int(cy + vy * t_min))
        p2 = (int(cx + vx * t_max), int(cy + vy * t_max))
        cv2.line(target_canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)
        return True

    # --- Approximate polygon to count corners ---
    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)
    circularity = 4 * math.pi * area / (perimeter ** 2)

    if circularity > 0.75:
        (ccx, ccy), radius = cv2.minEnclosingCircle(pts)
        cv2.circle(target_canvas, (int(ccx), int(ccy)), int(radius), color, thickness, lineType=cv2.LINE_AA)
        return True

    if len(approx) == 3:
        pts3 = approx.reshape(3, 2)
        cv2.polylines(target_canvas, [pts3], True, color, thickness, lineType=cv2.LINE_AA)
        return True

    if len(approx) == 4:
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.polylines(target_canvas, [box], True, color, thickness, lineType=cv2.LINE_AA)
        return True

    return False  # not a recognizable shape -> caller keeps the freehand version


# ---------------------------
# 5. Text recognition ("Text Mode")
# ---------------------------
def recognize_and_render_text(region_canvas, bbox, color, target_canvas):
    """Run OCR on the drawn region and replace it with clean printed text."""
    if not OCR_AVAILABLE:
        return False

    x, y, w, h = bbox
    if w < 10 or h < 10:
        return False

    pad = 20
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = x + w + pad, y + h + pad
    crop = region_canvas[y0:y1, x0:x1]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)  # OCR expects dark text on light background

    text = pytesseract.image_to_string(binary, config="--psm 7").strip()
    if not text:
        return False

    # Clear the messy freehand region, then draw clean printed text in its place
    cv2.rectangle(target_canvas, (x0, y0), (x1, y1), (0, 0, 0), -1)
    font_scale = max(0.6, h / 60)
    cv2.putText(target_canvas, text, (x0 + 5, y1 - pad), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 2, cv2.LINE_AA)
    return True


# ---------------------------
# 6. Main loop
# ---------------------------
def main():
    global canvas, stroke_canvas, current_color, prev_x, prev_y, smooth_x, smooth_y
    global missed_frames, prev_time, APP_MODE, stroke_points, was_drawing, text_region_pts

    mode_label = "IDLE"

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera not accessible. Check webcam connection.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            stroke_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        layout = get_palette_layout(w)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        x1 = y1 = 0
        is_drawing_now = False

        if result.multi_hand_landmarks:
            missed_frames = 0
            hand_landmarks = result.multi_hand_landmarks[0]
            fingers, lm_list = fingers_up(hand_landmarks, frame.shape)

            raw_x, raw_y = lm_list[8]
            mx, my = lm_list[12]

            if smooth_x == 0 and smooth_y == 0:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = int(SMOOTHING * smooth_x + (1 - SMOOTHING) * raw_x)
                smooth_y = int(SMOOTHING * smooth_y + (1 - SMOOTHING) * raw_y)

            x1, y1 = smooth_x, smooth_y
            index_up = fingers[1] == 1
            middle_up = fingers[2] == 1

            if index_up and middle_up:
                mode_label = "SELECT"
                prev_x, prev_y = 0, 0

                if y1 < TOPBAR_HEIGHT:
                    for name, color, cx, cy, slot_w in layout:
                        if abs(x1 - cx) < slot_w / 2:
                            current_color = color

            elif index_up and not middle_up:
                mode_label = "DRAW"
                is_drawing_now = True

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1

                thickness = eraser_thickness if current_color == (90, 90, 90) else brush_thickness
                draw_color = (0, 0, 0) if current_color == (90, 90, 90) else current_color

                # Freehand mode (or eraser, in any mode) draws straight to the
                # permanent canvas. Shape/Text mode draw to a temp layer first
                # so the raw stroke can be swapped out once it's finished.
                if APP_MODE == "FREEHAND" or current_color == (90, 90, 90):
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), draw_color, thickness, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (x1, y1), thickness // 2, draw_color, cv2.FILLED, lineType=cv2.LINE_AA)
                else:
                    cv2.line(stroke_canvas, (prev_x, prev_y), (x1, y1), draw_color, thickness, lineType=cv2.LINE_AA)
                    cv2.circle(stroke_canvas, (x1, y1), thickness // 2, draw_color, cv2.FILLED, lineType=cv2.LINE_AA)
                    stroke_points.append((x1, y1))
                    text_region_pts.append((x1, y1))

                prev_x, prev_y = x1, y1

            else:
                mode_label = "IDLE"
                prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=THEME_ACCENT, thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1)
            )
        else:
            mode_label = "NO HAND"
            missed_frames += 1
            if missed_frames > MAX_MISSED_FRAMES:
                prev_x, prev_y = 0, 0
                smooth_x, smooth_y = 0, 0

        # ---- Stroke just ended: commit it (Shape mode snaps automatically) ----
        if was_drawing and not is_drawing_now and current_color != (90, 90, 90):
            if APP_MODE == "SHAPE" and stroke_points:
                snapped = snap_to_perfect_shape(stroke_points, current_color, brush_thickness, canvas)
                if not snapped:
                    canvas = cv2.bitwise_or(canvas, stroke_canvas)  # keep freehand if no shape matched
                stroke_canvas[:] = 0
                stroke_points = []
            elif APP_MODE == "TEXT" and stroke_points:
                # Just merge the stroke for now; OCR conversion happens on 'w' key
                canvas = cv2.bitwise_or(canvas, stroke_canvas)
                stroke_canvas[:] = 0
                stroke_points = []
        was_drawing = is_drawing_now

        # ---- Merge layers: canvas + active stroke + webcam frame ----
        combined_drawing = cv2.bitwise_or(canvas, stroke_canvas)
        gray_canvas = cv2.cvtColor(combined_drawing, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame_bg = cv2.bitwise_and(frame, mask)
        combined = cv2.bitwise_or(frame_bg, combined_drawing)

        # ---- Top bar ----
        draw_glass_panel(combined, (0, 0), (w, TOPBAR_HEIGHT), THEME_BG, alpha=0.6, radius=0)
        cv2.putText(combined, "AIR CANVAS", (18, 34), cv2.FONT_HERSHEY_DUPLEX,
                    0.85, THEME_ACCENT, 2, cv2.LINE_AA)
        cv2.putText(combined, f"Mode: {APP_MODE}  (1 Freehand / 2 Shape / 3 Text)", (18, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, THEME_MUTED, 1, cv2.LINE_AA)

        for name, color, cx, cy, slot_w in layout:
            is_selected = (color == current_color)
            radius = SWATCH_RADIUS + (5 if is_selected else 0)
            if is_selected:
                cv2.circle(combined, (cx, cy), radius + 6, THEME_ACCENT, 2, lineType=cv2.LINE_AA)
            swatch_display = (40, 40, 40) if name == "Eraser" else color
            cv2.circle(combined, (cx, cy), radius, swatch_display, cv2.FILLED, lineType=cv2.LINE_AA)
            cv2.circle(combined, (cx, cy), radius, (15, 15, 15), 2, lineType=cv2.LINE_AA)
            if name == "Eraser":
                cv2.putText(combined, "E", (cx - 7, cy + 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (230, 230, 230), 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(combined, name, (int(cx - tw / 2), TOPBAR_HEIGHT - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, THEME_MUTED, 1, cv2.LINE_AA)

        # ---- Fingertip cursor ----
        if x1 or y1:
            cursor_color = (40, 40, 40) if current_color == (90, 90, 90) else current_color
            cv2.circle(combined, (x1, y1), 12, cursor_color, cv2.FILLED, lineType=cv2.LINE_AA)
            cv2.circle(combined, (x1, y1), 12, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        # ---- Bottom status bar ----
        bar_h = 42
        draw_glass_panel(combined, (0, h - bar_h), (w, h), THEME_BG, alpha=0.55, radius=0)

        now = time.time()
        fps = 1 / (now - prev_time) if prev_time else 0
        prev_time = now

        mode_colors = {"DRAW": (0, 220, 0), "SELECT": (0, 200, 255),
                        "IDLE": THEME_MUTED, "NO HAND": (0, 0, 220)}
        cv2.circle(combined, (24, h - bar_h // 2), 7, mode_colors.get(mode_label, THEME_MUTED), -1, lineType=cv2.LINE_AA)
        cv2.putText(combined, f"Hand: {mode_label}", (40, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, THEME_TEXT, 1, cv2.LINE_AA)

        if APP_MODE == "TEXT" and not OCR_AVAILABLE:
            cv2.putText(combined, "OCR not installed - see comments at top of file",
                        (250, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(combined, f"FPS: {int(fps)}", (w - 330, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, THEME_TEXT, 1, cv2.LINE_AA)
        cv2.putText(combined, "C:Clear S:Save W:OCR Q:Quit", (w - 250, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, THEME_MUTED, 1, cv2.LINE_AA)

        cv2.imshow("Air Canvas", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            stroke_canvas[:] = 0
            text_region_pts = []
        elif key == ord('s'):
            os.makedirs("saved_drawings", exist_ok=True)
            filename = f"saved_drawings/drawing_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f"Saved: {filename}")
        elif key == ord('1'):
            APP_MODE = "FREEHAND"
        elif key == ord('2'):
            APP_MODE = "SHAPE"
        elif key == ord('3'):
            APP_MODE = "TEXT"
        elif key == ord('w') and APP_MODE == "TEXT":
            if text_region_pts:
                pts = np.array(text_region_pts)
                x, y, bw, bh = cv2.boundingRect(pts)
                recognize_and_render_text(canvas, (x, y, bw, bh), current_color, canvas)
                text_region_pts = []
            else:
                print("Nothing written yet to recognize.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()