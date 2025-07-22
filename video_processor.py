# File: video_processor.py

import os
import cv2
import asyncio
import shutil
import pandas as pd
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFont
from moviepy import *
from pydub import AudioSegment
import edge_tts


class VideoProcessor:
    def __init__(self, config, log_callback):
        self.config = config
        self.log = log_callback
        self.global_bbox = config.get("bbox", None)

    def log_msg(self, msg):
        if self.log:
            self.log(msg)

    def cleanup_temp(self):
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
        shutil.rmtree("temp_frames", ignore_errors=True)

    async def generate_tts(self, text):
        communicate = edge_tts.Communicate(text=text, voice=self.config["tts_voice"])
        await communicate.save("tts_audio.mp3")

    def split_video(self, tts_wav_path):
        ffmpeg_path = self.config["ffmpeg_path"]
        input_video = self.config["input_video"]
        split_time = self.config["split_part1_end"]

        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-ss",
                "0",
                "-t",
                str(split_time),
                "-i",
                input_video,
                "-c",
                "copy",
                "p1.mp4",
            ]
        )

        replacement_audio = AudioSegment.from_wav(tts_wav_path)
        tts_duration_sec = len(replacement_audio) / 1000.0
        split_time_for_part2 = 1 + tts_duration_sec + 0.5

        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-ss",
                f"{split_time_for_part2:.2f}",
                "-i",
                input_video,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "p2.mp4",
            ]
        )

    def modify_p1(self, name, output_folder):
        replacement_text = self.config["replacement_text_template"].format(name)
        font_path = self.config["font_path"]
        target_font_color = tuple(self.config["target_font_color"])
        duration_to_modify = float(self.config["blur_duration"])

        os.makedirs("temp_frames", exist_ok=True)

        cap = cv2.VideoCapture("p1.mp4")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        x1, y1, x2, y2 = self.global_bbox
        box_w, box_h = x2 - x1, y2 - y1

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

            out_path = os.path.join("temp_frames", f"frame_{i:04}.png")
            cv2.imwrite(out_path, final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            frame_list.append(out_path)

        cap.release()

        subprocess.run(
            [
                self.config["ffmpeg_path"],
                "-y",
                "-i",
                "p1.mp4",
                "-vn",
                "-acodec",
                "copy",
                "temp_audio.aac",
            ]
        )
        asyncio.run(self.generate_tts(replacement_text))
        subprocess.run(
            [self.config["ffmpeg_path"], "-y", "-i", "tts_audio.mp3", "tts_audio.wav"]
        )

        original = AudioSegment.from_file("temp_audio.aac")
        replacement = AudioSegment.from_wav("tts_audio.wav").set_frame_rate(44100)
        final_audio = (
            original[:1000] + replacement + original[1000 + len(replacement) :]
        )
        final_audio.export("modified_audio.m4a", format="ipod")

        clip = ImageSequenceClip(frame_list, fps=fps).with_audio(
            AudioFileClip("modified_audio.m4a")
        )
        clip.write_videofile("p1.mp4", codec="libx264", audio_codec="aac")

    def merge_videos(self, output_path):
        with open("files.txt", "w") as f:
            f.write("file 'p1.mp4'\n")
            f.write("file 'p2.mp4'\n")
        subprocess.run(
            [
                self.config["ffmpeg_path"],
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

    def process_all(self):
        df = pd.read_excel(self.config["excel_path"])
        for idx, row in df.iterrows():
            name = str(row["Name"]).strip()
            mobile = str(row["Mobile"]).strip()
            output_folder = os.path.join("output", f"{name}_{mobile}")
            os.makedirs(output_folder, exist_ok=True)

            self.log_msg(f"\n🔧 Generating video for: {name} ({mobile})")
            self.log_msg("🎬 Step 1: Generate TTS + split...")
            asyncio.run(
                self.generate_tts(self.config["replacement_text_template"].format(name))
            )
            subprocess.run(
                [
                    self.config["ffmpeg_path"],
                    "-y",
                    "-i",
                    "tts_audio.mp3",
                    "tts_audio.wav",
                ]
            )
            self.split_video("tts_audio.wav")

            self.log_msg("🎨 Step 2: Modify visual/audio...")
            self.modify_p1(name, output_folder)

            self.log_msg("🔗 Step 3: Merge parts...")
            final_output = os.path.join(output_folder, "final.mp4")
            self.merge_videos(final_output)

            self.log_msg("🧹 Cleaning up...")
            self.cleanup_temp()

        self.log_msg("\n✅ All videos generated in /output/")
