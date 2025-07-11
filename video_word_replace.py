import cv2
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from moviepy import *
import shutil

# Config
video_path = "data/p1.mp4"
excel_path = "data.xlsx"
output_video = "data/output_video1.mp4"
font_path = "font/NotoSansDevanagari-Regular.ttf"
temp_dir = "temp_frames"
replacement_index = 0

# Load name from Excel
df = pd.read_excel(excel_path)
replacement_word = df.iloc[replacement_index]["Name"]

# Setup temp folder
shutil.rmtree(temp_dir, ignore_errors=True)
os.makedirs(temp_dir, exist_ok=True)

# Read video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get first frame for box selection
ret, first_frame = cap.read()
if not ret:
    raise Exception("Failed to read first frame")

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
print("Draw a box on the word to replace, then press any key.")

while True:
    cv2.imshow("Draw", clone)
    if cv2.waitKey(1) & 0xFF != 255:
        break

cv2.destroyAllWindows()

if len(bbox) != 2:
    raise Exception("Bounding box not selected")

(x1, y1), (x2, y2) = bbox
font = ImageFont.truetype(font_path, 40)

# Rewind to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_list = []
modify_frame_count = int(fps * 2)
for i in range(frame_total):
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Frame {i} @ time: {i/fps:.3f}s", end=" - ")

    # Replace only first 1s (fps frames)
    # if i <= fps + 2:
    if i < modify_frame_count:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(image_pil)

        draw.rectangle([x1, y1, x2, y2], fill="white")
        draw.text((x1 + 5, y1 + 5), replacement_word, font=font, fill="black")
        final_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        print(f"Modifying frame {i}")
    else:
        final_frame = frame
        print(f"Keeping frame {i} as is")

    out_path = os.path.join(temp_dir, f"frame_{i:04}.png")
    # cv2.imwrite(out_path, final_frame)
    cv2.imwrite(out_path, final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    frame_list.append(out_path)

cap.release()

# Extract original audio
temp_audio = "temp_audio.aac"
os.system(f"ffmpeg -y -i {video_path} -vn -acodec copy {temp_audio}")

# Build video from frames
clip = ImageSequenceClip(frame_list, fps=fps)
clip = clip.with_audio(AudioFileClip(temp_audio))
clip.write_videofile(
    output_video,
    codec="libx264",
    audio_codec="aac",
    # bitrate="3000k",
    # ffmpeg_params=["-x264opts", "keyint=1"],
    ffmpeg_params=["-crf", "18", "-preset", "slow", "-x264opts", "keyint=1"],
)


print("✅ Flicker-free video saved at:", output_video)
