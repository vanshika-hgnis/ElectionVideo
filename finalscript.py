import os
import cv2
import pandas as pd
import numpy as np
import shutil
import asyncio
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from moviepy import *

# --- Configuration ---
video_path = "data/p1.mp4"
excel_path = "data.xlsx"
output_video = "data/final_output_video.mp4"
font_path = "font/NotoSansDevanagari-Bold.ttf"
temp_dir = "temp_frames"
replacement_index = 0
target_font_color = (59, 180, 255)
duration_to_modify = 2  # seconds

# Audio config
original_audio_path = "temp_audio.aac"
modified_audio_path = "modified_audio.m4a"
tts_audio_mp3 = "tts_audio.mp3"
tts_audio_wav = "tts_audio.wav"
replacement_text_template = "{} जी नमस्कार"


# --- Load name from Excel ---
df = pd.read_excel(excel_path)
replacement_word = df.iloc[replacement_index]["Name"]
replacement_text = replacement_text_template.format(replacement_word)

# --- Setup frame temp folder ---
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

    if i < modify_frame_count:
        region = frame[y1:y2, x1:x2].copy()
        blurred = cv2.GaussianBlur(region, (45, 45), 0)
        frame[y1:y2, x1:x2] = blurred

        shadow = frame.copy()
        cv2.rectangle(shadow, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(shadow, 0.4, frame, 0.6, 0)

        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

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

        bbox_text = font.getbbox(replacement_word)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        tx = x1 + (box_w - tw) // 2
        ty = y1 + (box_h - th) // 2 - 3

        draw.text((tx, ty), replacement_word, font=font, fill=target_font_color)
        final_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    else:
        final_frame = frame

    out_path = os.path.join(temp_dir, f"frame_{i:04}.png")
    cv2.imwrite(out_path, final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    frame_list.append(out_path)

cap.release()

# --- Extract original audio ---
os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec copy "{original_audio_path}"')


# --- Generate TTS with Edge ---
async def generate_tts():
    communicate = edge_tts.Communicate(
        text=replacement_text,
        voice="hi-IN-MadhurNeural",
        rate="+0%",
    )
    await communicate.save(tts_audio_mp3)


asyncio.run(generate_tts())

# Convert MP3 to WAV
os.system(f'ffmpeg -y -i "{tts_audio_mp3}" "{tts_audio_wav}"')

# Replace audio at 1s mark
original_audio = AudioSegment.from_file(original_audio_path)
replacement_audio = AudioSegment.from_wav(tts_audio_wav).set_frame_rate(44100)
start_ms = 1000
end_ms = start_ms + len(replacement_audio)

final_audio = original_audio[:start_ms] + replacement_audio + original_audio[end_ms:]
final_audio.export(modified_audio_path, format="ipod")

# --- Combine video and audio ---
clip = ImageSequenceClip(frame_list, fps=fps).with_audio(
    AudioFileClip(modified_audio_path)
)
clip.write_videofile(
    output_video,
    codec="libx264",
    audio_codec="aac",
    ffmpeg_params=["-crf", "18", "-preset", "slow", "-x264opts", "keyint=1"],
)

print("✅ Final cohesive video saved:", output_video)
