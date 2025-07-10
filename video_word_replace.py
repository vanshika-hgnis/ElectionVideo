import cv2
import pandas as pd
from moviepy import *
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

# ---------- CONFIG ----------
video_path = "data/p1.mp4"
excel_path = "data.xlsx"
output_path = "data/output_video.mp4"
frame_time = 0  # in seconds
replacement_index = 0
font_path = "font/NotoSansDevanagari-Regular.ttf"  # Use any Hindi TTF font

# ---------- STEP 1: Read Excel ----------
df = pd.read_excel(excel_path)
replacement_word = df.iloc[replacement_index]["Name"]

# ---------- STEP 2: Capture frame ----------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_time * fps))
ret, frame = cap.read()
cap.release()

if not ret:
    raise Exception("Could not read frame from video.")

# ---------- STEP 3: Draw bounding box ----------
clone = frame.copy()
bbox = []


def click_and_crop(event, x, y, flags, param):
    global bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        bbox.append((x, y))
        cv2.rectangle(clone, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.imshow("Select Word", clone)


cv2.namedWindow("Select Word")
cv2.setMouseCallback("Select Word", click_and_crop)
print("Draw a box around the word you want to replace... then press any key.")

while True:
    cv2.imshow("Select Word", clone)
    if cv2.waitKey(1) & 0xFF != 255:
        break

cv2.destroyAllWindows()

if len(bbox) != 2:
    raise Exception("Bounding box not selected.")

# ---------- STEP 4: Replace Text using PIL for Hindi ----------
(x1, y1), (x2, y2) = bbox

# Convert OpenCV to PIL
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(frame_rgb)
draw = ImageDraw.Draw(image_pil)

# Load Hindi font
try:
    font = ImageFont.truetype(font_path, 40)
except Exception as e:
    raise Exception(f"Font load error: {e}")

# Draw white box and Hindi text
draw.rectangle([x1, y1, x2, y2], fill="white")
draw.text((x1 + 5, y1 + 5), replacement_word, font=font, fill="black")

# Convert back to OpenCV
frame_edited = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
cv2.imwrite("modified_frame.jpg", frame_edited)
print("✅ Modified frame saved as 'modified_frame.jpg'")

# ---------- STEP 5: Replace frame in video ----------
edited_frame_path = "modified_frame.jpg"
full_clip = VideoFileClip(video_path)

# Extend slightly beyond 1s to avoid original flicker
replace_end = 1 + (1 / fps)
size = full_clip.size

edited_clip = (
    ImageClip(edited_frame_path)
    .with_duration(replace_end)
    .with_fps(fps)
    .resized(size)
    .with_audio(full_clip.subclipped(0, replace_end).audio)
)

after_clip = full_clip.subclipped(replace_end)
final_clip = concatenate_videoclips([edited_clip, after_clip], method="compose")
final_clip.write_videofile(
    output_path, codec="libx264", audio_codec="aac", bitrate="3000k"
)
print(f"🎬 Final video ready at: {output_path}")
