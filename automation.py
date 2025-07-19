import os
import json
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

# ---------- Load config.json ----------
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    default_config = {
        "ffmpeg_path": "D:/C-Drive/ffmpeg-7.1.1-full_build/bin/ffmpeg.exe",
        "input_video": "Example/input.mp4",
        "output_dir": "Example/",
        "temp_frames_dir": "temp_frames",
        "excel_path": "Example/data.xlsx",
        "replacement_index": 0,
        "replacement_text_template": "{} जी नमस्कार",
        "font_path": "font/NotoSansDevanagari-Bold.ttf",
        "target_font_color": [59, 180, 255],
        "split_part1_start": 0,
        "split_part1_end": 3,
        "tts_insert_time": 1.0,
        "extra_buffer_after_tts": 0.5,
        "blur_duration": 2,
        "video_crf": 18,
        "video_preset": "slow",
        "tts_voice": "hi-IN-MadhurNeural",
        "tts_rate": "+0%",
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(default_config, f, indent=2)
    print("🛠️ 'config.json' created. Please configure it and re-run the script.")
    exit()

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

# ---------- Config Parameters ----------
ffmpeg_path = config["ffmpeg_path"]
input_video = config["input_video"]
output_dir = config["output_dir"]
temp_dir = config["temp_frames_dir"]
excel_path = config["excel_path"]
replacement_template = config["replacement_text_template"]
font_path = config["font_path"]
font_color = tuple(config["target_font_color"])
split_start = config["split_part1_start"]
split_end = config["split_part1_end"]
tts_insert_time = config["tts_insert_time"]
extra_buffer = config["extra_buffer_after_tts"]
blur_duration = config["blur_duration"]
video_crf = config["video_crf"]
video_preset = config["video_preset"]
tts_voice = config["tts_voice"]
tts_rate = config["tts_rate"]

global_bbox = None  # For reuse


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
        except:
            pass
    shutil.rmtree(temp_dir, ignore_errors=True)


async def generate_tts(text):
    communicate = edge_tts.Communicate(text=text, voice=tts_voice, rate=tts_rate)
    await communicate.save("tts_audio.mp3")


def split_video(tts_wav_path):
    # Part 1
    cmd1 = (
        ffmpeg.input(input_video, ss=split_start, t=split_end - split_start)
        .output("p1.mp4", c="copy")
        .compile(cmd=ffmpeg_path)
    )
    subprocess.run(cmd1)

    # Part 2
    replacement_audio = AudioSegment.from_wav(tts_wav_path)
    tts_duration = len(replacement_audio) / 1000.0
    part2_start = tts_insert_time + tts_duration + extra_buffer

    cmd2 = f'"{ffmpeg_path}" -y -ss {part2_start:.2f} -i "{input_video}" -c:v libx264 -preset {video_preset} -crf {video_crf} -c:a aac "p2.mp4"'
    subprocess.run(cmd2)


def get_bbox_from_user(frame):
    clone = frame.copy()
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
    print("📍 Draw a box around the name and press any key...")
    while True:
        cv2.imshow("Draw", clone)
        if cv2.waitKey(1) & 0xFF != 255:
            break
    cv2.destroyAllWindows()
    if len(bbox) != 2:
        raise Exception("❌ Bounding box selection failed.")
    return bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]


def modify_p1(name, output_folder):
    global global_bbox

    replacement_text = replacement_template.format(name)
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture("p1.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        raise Exception("❌ Failed to read video")
    if global_bbox is None:
        global_bbox = get_bbox_from_user(first_frame)
    x1, y1, x2, y2 = global_bbox
    box_w, box_h = x2 - x1, y2 - y1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    modify_frame_count = int(fps * blur_duration)
    frame_list = []

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
            tx = x1 + (box_w - tw) // 2
            ty = y1 + (box_h - th) // 2 - 3
            draw.text((tx, ty), name, font=font, fill=font_color)
            final_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        else:
            final_frame = frame

        out_path = os.path.join(temp_dir, f"frame_{i:04}.png")
        cv2.imwrite(out_path, final_frame)
        frame_list.append(out_path)
    cap.release()

    os.system(f'ffmpeg -y -i "p1.mp4" -vn -acodec copy "temp_audio.aac"')
    asyncio.run(generate_tts(replacement_text))
    os.system(f'ffmpeg -y -i "tts_audio.mp3" "tts_audio.wav"')

    original = AudioSegment.from_file("temp_audio.aac")
    replacement = AudioSegment.from_wav("tts_audio.wav").set_frame_rate(44100)
    final_audio = (
        original[: int(tts_insert_time * 1000)]
        + replacement
        + original[int(tts_insert_time * 1000) + len(replacement) :]
    )
    final_audio.export("modified_audio.m4a", format="ipod")

    clip = ImageSequenceClip(frame_list, fps=fps).with_audio(
        AudioFileClip("modified_audio.m4a")
    )
    clip.write_videofile("p1.mp4", codec="libx264", audio_codec="aac")


def merge_videos(output_path):
    with open("files.txt", "w") as f:
        f.write("file 'p1.mp4'\n")
        f.write("file 'p2.mp4'\n")
    subprocess.run(
        [
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
    )


# ---------- MAIN ----------
if __name__ == "__main__":
    df = pd.read_excel(excel_path)
    for idx, row in df.iterrows():
        name = str(row["Name"]).strip()
        mobile = str(row["Mobile"]).strip()
        output_folder = os.path.join(output_dir, f"{name}_{mobile}")
        os.makedirs(output_folder, exist_ok=True)

        print(f"\n🎯 Generating for: {name} ({mobile})")
        asyncio.run(generate_tts(replacement_template.format(name)))
        os.system('ffmpeg -y -i "tts_audio.mp3" "tts_audio.wav"')
        split_video("tts_audio.wav")

        modify_p1(name, output_folder)
        final_path = os.path.join(output_folder, "final.mp4")
        merge_videos(final_path)

        print(f"✅ Saved: {final_path}")
        cleanup_temp()

    print("\n✅ All done.")
