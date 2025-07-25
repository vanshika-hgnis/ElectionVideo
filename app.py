import os
import json
import shutil
import asyncio
import subprocess
import sys
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
    QProgressBar,
)
from PyQt5.QtGui import QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QThread, QSettings, pyqtSignal
import cv2
import edge_tts
from moviepy import *
from pydub import AudioSegment
from video_processor import VideoProcessor

CONFIG_FILE = "config.json"


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Election Video Generator")
        self.setGeometry(100, 100, 1200, 800)
        self.config = self.load_config()

        self.settings = QSettings("Dynode Corp", "VideoGenerator")
        self.resize(self.settings.value("size", self.size()))
        self.move(self.settings.value("pos", self.pos()))

        self.excel_path = QLineEdit(self.config.get("excel_path", ""))
        self.video_path = QLineEdit(self.config.get("input_video", ""))
        self.output_folder = QLineEdit(self.config.get("output_folder", "output"))
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
        self.bbox_time = QLineEdit(str(self.config.get("bbox_time", 0)))

        self.bbox_time.textChanged.connect(self.update_frame_preview_from_time)

        self.image_view = QLabel(self)
        self.image_view.setFixedHeight(400)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.progress = QProgressBar()
        self.progress.setValue(0)

        layout = QVBoxLayout()
        layout.addLayout(self.row("Excel File", self.excel_path, self.pick_excel))
        layout.addLayout(self.row("Input Video", self.video_path, self.pick_video))
        layout.addLayout(
            self.row("Output Folder", self.output_folder, self.pick_output_folder)
        )
        layout.addLayout(self.row("Font File", self.font_path, self.pick_font))
        layout.addLayout(self.row("Text Template", self.template_text))
        layout.addLayout(self.row("TTS Voice", self.voice_name))
        layout.addWidget(self.font_color_btn)
        layout.addLayout(self.row("Bounding Box Time (s)", self.bbox_time))
        layout.addWidget(QLabel("Draw Bounding Box on First Frame:"))
        layout.addWidget(self.image_view)
        layout.addLayout(self.row("Split Start Time", self.split_start))
        layout.addLayout(self.row("Split End Time", self.split_end))
        layout.addLayout(self.row("TTS Insert Time", self.insert_time))
        layout.addLayout(self.row("Blur Duration (sec)", self.blur_duration))
        layout.addLayout(self.row("FFmpeg Path", self.ffmpeg_path))
        layout.addWidget(self.progress)
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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setLayout(layout)
        scroll.setWidget(scroll_content)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self.load_first_frame()

    def update_progress(self, value):
        self.progress.setValue(value)

    def closeEvent(self, event):
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())
        super().closeEvent(event)

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

    def pick_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_folder.setText(path)

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

    def update_frame_preview_from_time(self):
        video_path = self.video_path.text()
        try:
            time_sec = float(self.bbox_time.text())
        except ValueError:
            return

        if not os.path.exists(video_path):
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * time_sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_view.setPixmap(pixmap)

    def load_first_frame(self):
        video = self.video_path.text()
        if not video or not os.path.exists(video):
            return
        try:
            bbox_time_sec = float(self.bbox_time.text())
        except ValueError:
            bbox_time_sec = 0
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_MSEC, bbox_time_sec * 1000)
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

    def get_bbox_from_opencv(self):
        video = self.video_path.text()
        try:
            bbox_time = float(self.bbox_time.text())
        except ValueError:
            bbox_time = 0.0

        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_MSEC, bbox_time * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.critical(self, "Error", "Could not load frame from video.")
            return None

        clone = frame.copy()
        bbox = []

        def click_and_crop(event, x, y, flags, param):
            nonlocal bbox
            if event == cv2.EVENT_LBUTTONDOWN:
                bbox = [(x, y)]
            elif event == cv2.EVENT_LBUTTONUP:
                bbox.append((x, y))
                cv2.rectangle(clone, bbox[0], bbox[1], (0, 255, 0), 2)
                cv2.imshow("Draw BBox", clone)

        cv2.namedWindow("Draw BBox")
        cv2.setMouseCallback("Draw BBox", click_and_crop)

        while True:
            cv2.imshow("Draw BBox", clone)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                break

        cv2.destroyAllWindows()

        if len(bbox) == 2:
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        else:
            return None

    def load_config(self):
        default_config = {
            "excel_path": "",
            "input_video": "",
            "output_folder": "output",
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
            "bbox_time": 0,
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
            "output_folder": self.output_folder.text(),
            "font_path": self.font_path.text(),
            "replacement_text_template": self.template_text.text(),
            "tts_voice": self.voice_name.text(),
            "target_font_color": list(self.font_color),
            "bbox": self.get_bbox_from_opencv(),
            "split_part1_start": float(self.split_start.text()),
            "split_part1_end": float(self.split_end.text()),
            "tts_insert_time": float(self.insert_time.text()),
            "blur_duration": float(self.blur_duration.text()),
            "ffmpeg_path": self.ffmpeg_path.text(),
            "bbox_time": float(self.bbox_time.text()),
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
        self.log.append("🛠 Starting integrated video generation...")

        class VideoWorker(QThread):
            progress = pyqtSignal(int)
            log = pyqtSignal(str)
            finished = pyqtSignal()

            def __init__(self, config):
                super().__init__()
                self.config = config

            def run(self):
                processor = VideoProcessor(
                    self.config, self.log.emit, self.progress.emit
                )
                processor.process_all()
                self.finished.emit()

        # processor = VideoProcessor(self.config, self.log.append)
        # processor.process_all()
        self.thread = QThread()
        self.worker = VideoWorker(self.config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.log.append)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(
            lambda: self.log.append("<span style='color:green;'>✅ All done!</span>")
        )
        self.progress.setValue(0)
        self.thread.start()


if __name__ == "__main__":
    app = QApplication([])
    win = VideoApp()
    win.show()
    app.exec_()
