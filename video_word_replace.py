import cv2
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from moviepy import *
import shutil

# --- Configuration ---
video_path = "data/p1.mp4"
excel_path = "data.xlsx"
output_video = "data/output_video1.mp4"
font_path = "font/NotoSansDevanagari-Bold.ttf"
temp_dir = "temp_frames"
replacement_index = 0
target_font_color = (59, 180, 255)  # Accurate color
duration_to_modify = 2  # seconds

# --- Load name from Excel ---
df = pd.read_excel(excel_path)
replacement_word = df.iloc[replacement_index]["Name"]

# --- Temp frame folder ---
shutil.rmtree(temp_dir, ignore_errors=True)
os.makedirs(temp_dir, exist_ok=True)

# --- Video setup ---
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, first_frame = cap.read()
if not ret:
    raise Exception("❌ Failed to read first frame")

clone = first_frame.copy()
bbox = []


def click_and_crop(event, x, y, flags, param):
    global bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        bbox.append((x, y))
        cv2.rectangle(clone, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.imshow("Draw", clone)


cv2.namedWindow("Draw")
cv2.setMouseCallback("Draw", click_and_crop)
print("📍 Draw a box around the original name and press any key")

while True:
    cv2.imshow("Draw", clone)
    if cv2.waitKey(1) & 0xFF != 255:
        break
cv2.destroyAllWindows()

if len(bbox) != 2:
    raise Exception("❌ Bounding box not selected properly.")

(x1, y1), (x2, y2) = bbox
box_w, box_h = x2 - x1, y2 - y1

# --- Reset capture ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_list = []
modify_frame_count = int(fps * duration_to_modify)

for i in range(frame_total):
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Frame {i} @ {i / fps:.2f}s", end=" - ")

    if i < modify_frame_count:
        # Blur background using OpenCV for accuracy
        region = frame[y1:y2, x1:x2].copy()
        blurred = cv2.GaussianBlur(region, (45, 45), 0)  # Strong blur
        frame[y1:y2, x1:x2] = blurred

        # Overlay slight dark transparent shadow to match text contrast
        shadow = frame.copy()
        cv2.rectangle(shadow, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(shadow, 0.4, frame, 0.6, 0)

        # Convert to PIL for text drawing
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        # Font auto-size
        font_size = 10
        while True:
            test_font = ImageFont.truetype(font_path, font_size)
            bbox_text = test_font.getbbox(replacement_word)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            if tw > box_w - 10 or th > box_h - 10:
                break
            font_size += 1
        font = ImageFont.truetype(font_path, font_size - 1)

        # Final position fine-tuned
        bbox_text = font.getbbox(replacement_word)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        tx = x1 + (box_w - tw) // 2
        ty = y1 + (box_h - th) // 2 - 3  # Shift up slightly

        draw.text((tx, ty), replacement_word, font=font, fill=target_font_color)

        final_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        print("✓ Modified")
    else:
        final_frame = frame
        print("✓ Unchanged")

    out_path = os.path.join(temp_dir, f"frame_{i:04}.png")
    cv2.imwrite(out_path, final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    frame_list.append(out_path)

cap.release()

# --- Audio ---
temp_audio = "temp_audio.aac"
os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec copy "{temp_audio}"')

# --- Assemble ---
clip = ImageSequenceClip(frame_list, fps=fps).with_audio(AudioFileClip(temp_audio))
clip.write_videofile(
    output_video,
    codec="libx264",
    audio_codec="aac",
    ffmpeg_params=["-crf", "18", "-preset", "slow", "-x264opts", "keyint=1"],
)

print("✅ Final output saved:", output_video)
