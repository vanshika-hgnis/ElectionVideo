import os
import cv2
import ffmpeg
import asyncio
import shutil
import pandas as pd
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFont
from moviepy import *
from pydub import AudioSegment
import edge_tts

# Configuration
ffmpeg_path = r"D:/C-Drive/ffmpeg-7.1.1-full_build/bin/ffmpeg.exe"
input_video = "Example/input.mp4"
temp_dir = "temp_frames"
font_path = "font/NotoSansDevanagari-Bold.ttf"
excel_path = "Example/data.xlsx"
replacement_text_template = "{} जी नमस्कार"
target_font_color = (59, 180, 255)
duration_to_modify = 2  # in seconds
split_time = 3  # in seconds


# Utility: Cleanup temporary files
def cleanup_temp():
    for f in [
        "p1.mp4",
        "p2.mp4",
        "temp_audio.aac",
        "tts_audio.mp3",
        "tts_audio.wav",
        "modified_audio.m4a",
        "files.txt",
    ]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    shutil.rmtree(temp_dir, ignore_errors=True)


# Step 1: Generate TTS
async def generate_tts(text):
    communicate = edge_tts.Communicate(text=text, voice="hi-IN-MadhurNeural")
    await communicate.save("tts_audio.mp3")


# Step 2: Split original video
def split_video(tts_wav_path):
    # First part
    cmd1 = (
        ffmpeg.input(input_video, ss=0, t=split_time)
        .output("p1.mp4", c="copy")
        .compile(cmd=ffmpeg_path)
    )
    subprocess.run(cmd1)

    # Second part (start after TTS)
    replacement_audio = AudioSegment.from_wav(tts_wav_path)
    tts_duration_sec = len(replacement_audio) / 1000.0
    split_time_for_part2 = 1 + tts_duration_sec + 0.5

    cmd2 = f'"{ffmpeg_path}" -y -ss {split_time_for_part2:.2f} -i "{input_video}" -c:v libx264 -preset fast -crf 23 -c:a aac "p2.mp4"'
    subprocess.run(cmd2)


# Step 3: Modify first part (overlay name and TTS audio)
def modify_p1(name, output_folder):
    replacement_text = replacement_text_template.format(name)

    # Setup temp frame directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture("p1.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get user bounding box once
    ret, first_frame = cap.read()
    if not ret:
        raise Exception("❌ Failed to read first frame")
    clone = first_frame.copy()
    bbox = []

    def click_and_crop(event, x, y, flags, param):
        nonlocal bbox
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
                bbox_text = test_font.getbbox(name)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
                if tw > box_w - 10 or th > box_h - 10:
                    break
                font_size += 1
            font = ImageFont.truetype(font_path, font_size - 1)
            bbox_text = font.getbbox(name)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            tx = x1 + (box_w - tw) // 2
            ty = y1 + (box_h - th) // 2 - 3
            draw.text((tx, ty), name, font=font, fill=target_font_color)
            final_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        else:
            final_frame = frame

        out_path = os.path.join(temp_dir, f"frame_{i:04}.png")
        cv2.imwrite(out_path, final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        frame_list.append(out_path)

    cap.release()

    # Audio Extraction
    os.system(f'ffmpeg -y -i "p1.mp4" -vn -acodec copy "temp_audio.aac"')

    # Generate TTS
    asyncio.run(generate_tts(replacement_text))
    os.system(f'ffmpeg -y -i "tts_audio.mp3" "tts_audio.wav"')

    # Audio Replacement
    original = AudioSegment.from_file("temp_audio.aac")
    replacement = AudioSegment.from_wav("tts_audio.wav").set_frame_rate(44100)
    final_audio = original[:1000] + replacement + original[1000 + len(replacement) :]
    final_audio.export("modified_audio.m4a", format="ipod")

    # Write modified video
    clip = ImageSequenceClip(frame_list, fps=fps).with_audio(
        AudioFileClip("modified_audio.m4a")
    )
    clip.write_videofile("p1.mp4", codec="libx264", audio_codec="aac")


# Step 4: Merge p1 + p2
def merge_videos(output_path):
    with open("files.txt", "w") as f:
        f.write("file 'p1.mp4'\n")
        f.write("file 'p2.mp4'\n")

    merge_cmd = [
        ffmpeg_path,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "files.txt",
        "-c",
        "copy",
        output_path,
    ]
    subprocess.run(merge_cmd)


# --- MAIN ---
if __name__ == "__main__":
    df = pd.read_excel(excel_path)
    for idx, row in df.iterrows():
        name = str(row["Name"]).strip()
        mobile = str(row["Mobile"]).strip()
        output_folder = os.path.join("output", f"{name}_{mobile}")
        os.makedirs(output_folder, exist_ok=True)

        print(f"\n🔧 Generating video for: {name} ({mobile})")
        print("🎬 Step 1: Generate TTS + split...")
        asyncio.run(generate_tts(replacement_text_template.format(name)))
        os.system('ffmpeg -y -i "tts_audio.mp3" "tts_audio.wav"')
        split_video("tts_audio.wav")

        print("🎨 Step 2: Modify visual/audio...")
        modify_p1(name, output_folder)

        print("🔗 Step 3: Merge parts...")
        final_output = os.path.join(output_folder, "final.mp4")
        merge_videos(final_output)

        print(f"🧹 Cleaning up...")
        cleanup_temp()

    print("\n✅ All videos generated in /output/")
