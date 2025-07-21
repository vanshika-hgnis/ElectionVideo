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
    QScrollArea,
)
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QRectF
import cv2
import edge_tts
from moviepy import *
from pydub import AudioSegment

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
        self.setGeometry(100, 100, 1200, 800)
        self.config = self.load_config()

        self.excel_path = QLineEdit(self.config.get("excel_path", ""))
        self.video_path = QLineEdit(self.config.get("input_video", ""))
        self.font_path = QLineEdit(self.config.get("font_path", ""))
        self.template_text = QLineEdit(
            self.config.get("replacement_text_template", "{} जी नमस्कार")
        )
        self.voice_name = QLineEdit(self.config.get("tts_voice", "hi-IN-MadhurNeural"))
        self.font_color_btn = QPushButton("Choose Font Color")
        self.font_color = tuple(self.config.get("target_font_color", [59, 180, 255]))

        self.split_start = QLineEdit(str(self.config.get("split_part1_start", 0)))
        self.split_end = QLineEdit(str(self.config.get("split_part1_end", 3)))
        self.insert_time = QLineEdit(str(self.config.get("tts_insert_time", 1)))
        self.blur_duration = QLineEdit(str(self.config.get("blur_duration", 2)))
        self.ffmpeg_path = QLineEdit(str(self.config.get("ffmpeg_path", "ffmpeg")))

        self.image_view = CropLabel(self)
        self.image_view.setFixedHeight(400)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addLayout(self.row("Excel File", self.excel_path, self.pick_excel))
        layout.addLayout(self.row("Input Video", self.video_path, self.pick_video))
        layout.addLayout(self.row("Font File", self.font_path, self.pick_font))
        layout.addLayout(self.row("Text Template", self.template_text))
        layout.addLayout(self.row("TTS Voice", self.voice_name))
        layout.addWidget(self.font_color_btn)
        layout.addWidget(QLabel("Draw Bounding Box on First Frame:"))
        layout.addWidget(self.image_view)
        layout.addLayout(self.row("Split Start Time", self.split_start))
        layout.addLayout(self.row("Split End Time", self.split_end))
        layout.addLayout(self.row("TTS Insert Time", self.insert_time))
        layout.addLayout(self.row("Blur Duration (sec)", self.blur_duration))
        layout.addLayout(self.row("FFmpeg Path", self.ffmpeg_path))

        self.font_color_btn.clicked.connect(self.pick_color)

        btns = QHBoxLayout()
        save_btn = QPushButton("📂 Save Config")
        run_btn = QPushButton("▶️ Generate Videos")
        save_btn.clicked.connect(self.save_config)
        run_btn.clicked.connect(self.generate_all)
        btns.addWidget(save_btn)
        btns.addWidget(run_btn)
        layout.addLayout(btns)

        layout.addWidget(QLabel("Logs"))
        layout.addWidget(self.log)

        # Scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setLayout(layout)
        scroll.setWidget(scroll_content)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

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
                pix.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        cap.release()

    def load_config(self):
        default_config = {
            "excel_path": "",
            "input_video": "",
            "font_path": "",
            "replacement_text_template": "{} जी नमस्कार",
            "tts_voice": "hi-IN-MadhurNeural",
            "target_font_color": [59, 180, 255],
            "bbox": None,
            "split_part1_start": 0,
            "split_part1_end": 3,
            "tts_insert_time": 1,
            "blur_duration": 2,
            "ffmpeg_path": "ffmpeg",
        }
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            return default_config

    def save_config(self):
        self.config = {
            "excel_path": self.excel_path.text(),
            "input_video": self.video_path.text(),
            "font_path": self.font_path.text(),
            "replacement_text_template": self.template_text.text(),
            "tts_voice": self.voice_name.text(),
            "target_font_color": list(self.font_color),
            "bbox": self.image_view.get_rect(),
            "split_part1_start": float(self.split_start.text()),
            "split_part1_end": float(self.split_end.text()),
            "tts_insert_time": float(self.insert_time.text()),
            "blur_duration": float(self.blur_duration.text()),
            "ffmpeg_path": self.ffmpeg_path.text(),
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        self.log.append("<span style='color:green;'>✅ Config saved.</span>")

    def generate_all(self):
        if not os.path.exists(self.excel_path.text()):
            self.log.append("<span style='color:red;'>❌ Excel file not found.</span>")
            return
        if not os.path.exists(self.video_path.text()):
            self.log.append("<span style='color:red;'>❌ Video file not found.</span>")
            return
        if not os.path.exists(self.font_path.text()):
            self.log.append("<span style='color:red;'>❌ Font file not found.</span>")
            return
        self.save_config()
        self.log.append("🚀 Starting generation...")
        self.run_final_script()

    def run_final_script(self):
        self.log.append("🔄 Running finalscript.py...")
        self.log.repaint()
        process = subprocess.Popen(
            ["python", "finalscript.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        for line in process.stdout:
            self.log.append(line.strip())
            QApplication.processEvents()
        process.wait()
        self.log.append("<span style='color:green;'>✅ Done.</span>")


if __name__ == "__main__":
    app = QApplication([])
    win = VideoApp()
    win.show()
    app.exec_()
