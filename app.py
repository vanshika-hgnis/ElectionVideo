# integrated_app.py

import os
import json
import shutil
import asyncio
import subprocess
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QLineEdit,
    QTextEdit,
    QColorDialog,
    QMessageBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QRectF, QPointF
from moviepy import *
from pydub import AudioSegment
import edge_tts
import cv2

CONFIG_FILE = "config.json"


class CropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.start_point = None
        self.end_point = None
        self.rect_color = QColor(0, 255, 0)
        self.rect = None

    def mousePressEvent(self, event):
        self.start_point = event.pos()

    def mouseReleaseEvent(self, event):
        self.end_point = event.pos()
        self.rect = QRectF(self.start_point, self.end_point).normalized()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.rect:
            painter = QPainter(self)
            painter.setPen(QPen(self.rect_color, 2))
            painter.drawRect(self.rect)

    def get_rect(self):
        if self.rect:
            return [
                int(self.rect.left()),
                int(self.rect.top()),
                int(self.rect.right()),
                int(self.rect.bottom()),
            ]
        return None


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Election Video Generator")
        self.setGeometry(100, 100, 900, 700)
        self.config = self.load_config()

        # UI elements
        self.excel_path = QLineEdit(self.config.get("excel_path", ""))
        self.video_path = QLineEdit(self.config.get("input_video", ""))
        self.font_path = QLineEdit(self.config.get("font_path", ""))
        self.template_text = QLineEdit(
            self.config.get("replacement_text_template", "{} जी नमस्कार")
        )
        self.voice_name = QLineEdit(self.config.get("tts_voice", "hi-IN-MadhurNeural"))
        self.font_color_btn = QPushButton("Choose Font Color")
        self.font_color = tuple(self.config.get("target_font_color", [59, 180, 255]))
        self.log = QTextEdit()
        self.image_view = CropLabel(self)
        self.image_view.setFixedHeight(300)

        layout = QVBoxLayout()

        layout.addLayout(self.row("Excel File", self.excel_path, self.pick_excel))
        layout.addLayout(self.row("Input Video", self.video_path, self.pick_video))
        layout.addLayout(self.row("Font File", self.font_path, self.pick_font))
        layout.addLayout(self.row("Text Template", self.template_text))
        layout.addLayout(self.row("TTS Voice", self.voice_name))

        layout.addWidget(self.font_color_btn)
        self.font_color_btn.clicked.connect(self.pick_color)
        layout.addWidget(QLabel("Draw Bounding Box on First Frame:"))
        layout.addWidget(self.image_view)

        btns = QHBoxLayout()
        save_btn = QPushButton("💾 Save Config")
        run_btn = QPushButton("▶️ Generate Videos")
        save_btn.clicked.connect(self.save_config)
        run_btn.clicked.connect(self.generate_all)
        btns.addWidget(save_btn)
        btns.addWidget(run_btn)
        layout.addLayout(btns)

        layout.addWidget(QLabel("Logs"))
        layout.addWidget(self.log)

        self.setLayout(layout)

        self.load_first_frame()

    def row(self, label, field, browse_func=None):
        h = QHBoxLayout()
        h.addWidget(QLabel(label))
        h.addWidget(field)
        if browse_func:
            btn = QPushButton("Browse")
            btn.clicked.connect(browse_func)
            h.addWidget(btn)
        return h

    def pick_excel(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel File", "", "Excel Files (*.xlsx)"
        )
        if path:
            self.excel_path.setText(path)
            self.load_first_frame()

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.mov *.avi)"
        )
        if path:
            self.video_path.setText(path)
            self.load_first_frame()

    def pick_font(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Font", "", "Fonts (*.ttf)")
        if path:
            self.font_path.setText(path)

    def pick_color(self):
        color = QColorDialog.getColor(QColor(*self.font_color))
        if color.isValid():
            self.font_color = (color.red(), color.green(), color.blue())
            self.font_color_btn.setStyleSheet(
                f"background-color: rgb{self.font_color};"
            )

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_config(self):
        self.config["excel_path"] = self.excel_path.text()
        self.config["input_video"] = self.video_path.text()
        self.config["font_path"] = self.font_path.text()
        self.config["replacement_text_template"] = self.template_text.text()
        self.config["tts_voice"] = self.voice_name.text()
        self.config["target_font_color"] = list(self.font_color)
        self.config["bbox"] = self.image_view.get_rect()

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        self.log.append("✅ Config saved.")

    def load_first_frame(self):
        video = self.video_path.text()
        if not video or not os.path.exists(video):
            return
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.image_view.setPixmap(
                pix.scaled(self.image_view.size(), Qt.KeepAspectRatio)
            )
        cap.release()

    async def generate_tts(self, text):
        communicate = edge_tts.Communicate(text=text, voice=self.voice_name.text())
        await communicate.save("tts_audio.mp3")

    def generate_all(self):
        self.save_config()
        try:
            df = pd.read_excel(self.excel_path.text())
            bbox = self.config.get("bbox")
            if not bbox:
                self.log.append("❌ Bounding box not set.")
                return

            for idx, row in df.iterrows():
                name = str(row["Name"]).strip()
                mobile = str(row["Mobile"]).strip()
                out_dir = os.path.join("output", f"{name}_{mobile}")
                os.makedirs(out_dir, exist_ok=True)

                self.log.append(f"🔧 Generating for {name} ({mobile})")
                text = self.template_text.text().format(name)
                asyncio.run(self.generate_tts(text))
                os.system('ffmpeg -y -i "tts_audio.mp3" "tts_audio.wav"')

                self.generate_video(name, bbox, out_dir)

                self.log.append(f"✅ Done: {out_dir}/final.mp4")
        except Exception as e:
            self.log.append(f"❌ Error: {str(e)}")

    def generate_video(self, name, bbox, out_dir):
        input_video = self.video_path.text()
        font_path = self.font_path.text()
        font_color = tuple(self.font_color)

        shutil.rmtree("temp_frames", ignore_errors=True)
        os.makedirs("temp_frames", exist_ok=True)

        cap = cv2.VideoCapture(input_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        modify_frame_count = int(fps * 2)

        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1
        frame_list = []

        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
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

            out_path = os.path.join("temp_frames", f"frame_{i:04}.png")
            cv2.imwrite(out_path, final_frame)
            frame_list.append(out_path)
        cap.release()

        os.system(
            'ffmpeg -y -i "{}" -vn -acodec copy "temp_audio.aac"'.format(input_video)
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
        clip.write_videofile(
            os.path.join(out_dir, "final.mp4"), codec="libx264", audio_codec="aac"
        )


if __name__ == "__main__":
    app = QApplication([])
    win = VideoApp()
    win.show()
    app.exec_()
