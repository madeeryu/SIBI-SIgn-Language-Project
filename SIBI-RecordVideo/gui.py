"""
Komponen GUI PyQt5 utama.
"""
import os
import time
import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QTime, QObject, Qt, QElapsedTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QComboBox, QSlider, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QStatusBar,
    QLineEdit, QCheckBox, QFileDialog
)

from recorder import VideoRecorder
from pose import PoseDrawer
from utils import ensure_dir, get_next_filename, append_metadata

# --- audio beep helper (Windows-first, fallback to Qt beep) ---
try:
    import winsound  # type: ignore
except Exception:
    winsound = None


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LSTM Sign-Language Dataset Recorder")
        self.resize(1100, 720)

        # --- atribut utama ---
        self.cap = None
        self.recorder = None
        self.pose_drawer = PoseDrawer()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.label_class = "wave"
        self.notes_text = ""
        self.start_time = None
        self.last_saved_path = None
        self.pose_enabled = True
        self.dataset_dir = "dataset"  # default directory
        self.current_camera_index = 0

        # Durasi maksimal rekaman (detik)
        self.duration_limit = 10.0

        # Camera adjustment values
        self.brightness_val = 0.0
        self.contrast_val = 1.0
        self.saturation_val = 1.0

        # Countdown & auto takes
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_value = 0
        self.current_take = 0
        self.total_takes = 1
        self.auto_takes_running = False

        # --- precise record control (frame-aligned) ---
        self.pending_start_recording = False
        self.pending_start_at = None  # monotonic timestamp when countdown ended
        self.record_elapsed = QElapsedTimer()

        # --- init UI ---
        self.init_ui()

        # --- detect available cameras ---
        self.detect_cameras()

        # --- mulai preview ---
        self.open_camera(self.current_camera_index)

    # ---------- UI ----------
    def init_ui(self):
        # Layout utama
        main_layout = QHBoxLayout(self)

        # Kiri: preview
        left = QVBoxLayout()
        self.video_label = QLabel("Loading camera...")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color:white;")
        left.addWidget(self.video_label)

        # Overlay countdown
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(QtCore.Qt.AlignCenter)
        self.countdown_label.setStyleSheet(
            "font-size: 72px; color: yellow; background-color: rgba(0,0,0,0);"
        )
        self.countdown_label.setVisible(False)
        left.addWidget(self.countdown_label)

        # Kanan: kontrol
        right = QVBoxLayout()
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_frame.setLayout(right)
        right_frame.setMaximumWidth(420)

        # --- pilih kamera ---
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        camera_layout.addWidget(self.camera_combo)

        self.refresh_camera_btn = QPushButton("🔄")
        self.refresh_camera_btn.setMaximumWidth(40)
        self.refresh_camera_btn.setToolTip("Refresh camera list")
        self.refresh_camera_btn.clicked.connect(self.detect_cameras)
        camera_layout.addWidget(self.refresh_camera_btn)
        right.addLayout(camera_layout)

        # --- pilih resolusi ---
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        for (w, h), label in self.get_resolution_presets().items():
            self.resolution_combo.addItem(label, (w, h))
        # default: 1920x1080 jika ada, kalau tidak 1280x720, else pertama
        default_idx = self.resolution_combo.findData((1920, 1080))
        if default_idx < 0:
            default_idx = self.resolution_combo.findData((1280, 720))
        if default_idx >= 0:
            self.resolution_combo.setCurrentIndex(default_idx)
        self.resolution_combo.currentIndexChanged.connect(self.on_resolution_changed)
        res_layout.addWidget(self.resolution_combo)
        right.addLayout(res_layout)

        # --- pilih directory ---
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Save Directory:"))
        self.dir_btn = QPushButton("Browse...")
        self.dir_btn.clicked.connect(self.choose_directory)
        dir_layout.addWidget(self.dir_btn)
        right.addLayout(dir_layout)

        # --- orientasi ---
        orient_layout = QHBoxLayout()
        orient_layout.addWidget(QLabel("Orientasi:"))
        self.orient_combo = QComboBox()
        self.orient_combo.addItems(["Landscape", "Potrait"])
        orient_layout.addWidget(self.orient_combo)
        right.addLayout(orient_layout)

        self.dir_label = QLabel(f"📁 {self.dataset_dir}")
        self.dir_label.setStyleSheet("color: gray; font-size: 10px;")
        right.addWidget(self.dir_label)

        # --- input kelas (text input) ---
        right.addWidget(QLabel("Label Kelas:"))
        self.class_input = QLineEdit()
        self.class_input.setText("wave")
        self.class_input.setPlaceholderText("Masukkan nama kelas...")
        self.class_input.textChanged.connect(self.on_class_changed)
        right.addWidget(self.class_input)

        # --- input durasi maksimal ---
        right.addWidget(QLabel("Durasi Maksimal (detik):"))
        self.duration_limit_input = QLineEdit()
        self.duration_limit_input.setText("3")
        self.duration_limit_input.setPlaceholderText("Masukkan durasi maksimal...")
        self.duration_limit_input.textChanged.connect(self.on_duration_limit_changed)
        right.addWidget(self.duration_limit_input)

        # --- input jumlah take ---
        right.addWidget(QLabel("Jumlah Take Otomatis:"))
        self.take_count_input = QLineEdit()
        self.take_count_input.setText("1")
        self.take_count_input.setPlaceholderText("Masukkan jumlah take...")
        right.addWidget(self.take_count_input)

        # --- checkbox pose detection ---
        self.pose_checkbox = QCheckBox("Enable Pose Detection (Hand + Body)")
        self.pose_checkbox.setChecked(True)
        self.pose_checkbox.stateChanged.connect(self.on_pose_toggle)
        right.addWidget(self.pose_checkbox)

        # --- tombol rekaman ---
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Recording (Home)")
        self.stop_btn = QPushButton("Stop Recording (Home)")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_auto_takes)
        self.stop_btn.clicked.connect(self.stop_recording)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        right.addLayout(btn_layout)

        # --- durasi ---
        self.duration_label = QLabel("Durasi: 0.0 detik")
        right.addWidget(self.duration_label)

        # --- notes ---
        right.addWidget(QLabel("Notes:"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(60)
        self.notes_edit.textChanged.connect(self.on_notes_changed)
        right.addWidget(self.notes_edit)

        # --- slider pengaturan kamera ---
        right.addWidget(QLabel("Brightness"))
        self.brightness_slider = QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        right.addWidget(self.brightness_slider)

        right.addWidget(QLabel("Contrast"))
        self.contrast_slider = QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        right.addWidget(self.contrast_slider)

        right.addWidget(QLabel("Saturation"))
        self.saturation_slider = QSlider(QtCore.Qt.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(100)
        self.saturation_slider.valueChanged.connect(self.on_saturation_changed)
        right.addWidget(self.saturation_slider)

        # --- preview last video ---
        self.preview_btn = QPushButton("Preview Last Video")
        self.preview_btn.clicked.connect(self.preview_last)
        right.addWidget(self.preview_btn)

        # --- status bar ---
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Idle")
        right.addWidget(self.status_bar)

        main_layout.addLayout(left, 1)
        main_layout.addWidget(right_frame)

        # --- timer durasi label ---
        self.duration_timer = QTimer()
        self.duration_timer.timeout.connect(self.update_duration)
        self.duration_timer.start(100)

        # --- shortcut keyboard ---
        QtWidgets.QShortcut(Qt.Key_Home, self, self.toggle_recording)
        QtWidgets.QShortcut(QtGui.QKeySequence("N"), self, self.new_take)

    @property
    def is_potrait(self):
        return self.orient_combo.currentText() == "Potrait"

    # ---------- RESOLUSI ----------
    def get_resolution_presets(self):
        """
        Preset resolusi yang umum, hingga 2K (QHD 2560x1440).
        Kamu bisa tambah preset lain kalau webcam mendukung.
        """
        # ordered dict-like behavior: Python 3.7+ keeps insertion order
        return {
            (640, 480): "640×480 (VGA)",
            (800, 600): "800×600 (SVGA)",
            (1280, 720): "1280×720 (HD)",
            (1920, 1080): "1920×1080 (Full HD)",
            (2048, 1080): "2048×1080 (2K DCI)",
            (2560, 1440): "2560×1440 (2K QHD)",
        }

    def current_resolution(self):
        data = self.resolution_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return int(data[0]), int(data[1])
        return 640, 480

    def on_resolution_changed(self, _index):
        # Re-open camera to apply reliably across drivers/backends
        self.open_camera(self.current_camera_index)

    # ---------- AUDIO ----------
    def beep(self, freq=880, duration_ms=80):
        """
        Suara blip sederhana untuk countdown/start.
        - Windows: winsound.Beep
        - fallback: QApplication.beep()
        """
        try:
            if winsound is not None:
                winsound.Beep(int(freq), int(duration_ms))
            else:
                QtWidgets.QApplication.beep()
        except Exception:
            # jangan ganggu proses utama kalau audio gagal
            pass

    # ---------- KAMERA ----------
    def detect_cameras(self):
        """Deteksi kamera yang tersedia"""
        self.camera_combo.clear()
        available_cameras = []

        # Cek hingga 10 kamera
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        # Jika tidak ada kamera dengan DSHOW, coba tanpa backend
        if not available_cameras:
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()

        # Tambahkan ke combo box
        if available_cameras:
            for cam_idx in available_cameras:
                self.camera_combo.addItem(f"Camera {cam_idx}", cam_idx)
            self.status_bar.showMessage(f"Found {len(available_cameras)} camera(s)")
        else:
            self.camera_combo.addItem("No camera found", -1)
            self.status_bar.showMessage("No camera detected")

    def open_camera(self, camera_index):
        """Buka kamera dengan index tertentu + terapkan resolusi yang dipilih."""
        # Tutup kamera sebelumnya jika ada
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.timer.stop()

        if camera_index < 0:
            self.video_label.setText("No camera available")
            return

        # Coba buka dengan DSHOW backend (lebih stabil di Windows)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        # Terapkan resolusi
        w, h = self.current_resolution()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        # Target FPS preview (driver bisa abaikan)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Jika gagal, coba tanpa backend
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Kamera {camera_index} tidak dapat diakses.")
            self.video_label.setText(f"Camera {camera_index} failed to open")
            return

        self.current_camera_index = camera_index

        # Ambil resolusi aktual (set bisa berbeda dari hasil)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.timer.start(30)  # ~33 FPS UI refresh
        self.status_bar.showMessage(
            f"Camera {camera_index} opened | {actual_w}×{actual_h}"
        )

    def on_camera_changed(self, index):
        """Event ketika user memilih kamera lain"""
        if index < 0:
            return
        camera_index = self.camera_combo.itemData(index)
        if camera_index is not None and camera_index >= 0:
            self.open_camera(camera_index)

    def on_brightness_changed(self, value):
        self.brightness_val = value

    def on_contrast_changed(self, value):
        self.contrast_val = value / 100.0

    def on_saturation_changed(self, value):
        self.saturation_val = value / 100.0

    def adjust_frame(self, frame):
        """Apply brightness, contrast, and saturation adjustments"""
        # Brightness
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness_val)

        # Contrast
        frame = cv2.convertScaleAbs(frame, alpha=self.contrast_val, beta=0)

        # Saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * self.saturation_val
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return frame

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # --- rotasi jika potrait ---
        if self.is_potrait:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame = self.adjust_frame(frame)

        if self.pose_enabled:
            frame = self.pose_drawer.draw(frame)

        # --- START recording persis pada frame pertama setelah countdown selesai ---
        if self.pending_start_recording and (self.recorder is None or not self.recorder.is_recording()):
            # start recorder menggunakan ukuran frame yang *sebenarnya* (setelah rotasi)
            h, w = frame.shape[:2]
            self.start_recording(frame_size_override=(w, h), first_frame=frame)
            self.pending_start_recording = False

        # --- tulis frame kalau sedang recording ---
        if self.recorder and self.recorder.is_recording():
            self.recorder.write(frame)

            # Stop presisi berbasis elapsed time (mengurangi drift timer)
            if self.duration_limit > 0 and self.record_elapsed.isValid():
                if self.record_elapsed.elapsed() >= int(self.duration_limit * 1000):
                    # untuk auto-takes: stop lalu lanjut
                    if self.auto_takes_running:
                        self.stop_recording_and_continue()
                    else:
                        self.stop_recording()

        # tampilkan
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                self.video_label.width(),
                self.video_label.height(),
                QtCore.Qt.KeepAspectRatio
            )
        )

    # ---------- REKAMAN ----------
    def start_recording(self, frame_size_override=None, first_frame=None):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Warning", "Kamera tidak aktif!")
            return

        self.label_class = self.class_input.text().strip()
        if not self.label_class:
            QMessageBox.warning(self, "Warning", "Label kelas tidak boleh kosong!")
            return

        folder = os.path.join(self.dataset_dir, self.label_class)
        ensure_dir(folder)
        filename = get_next_filename(folder, self.label_class, ext="mp4")
        filepath = os.path.join(folder, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # FPS target untuk writer (akan kamu koreksi via re-encode)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 120:
            fps = 30.0

        if frame_size_override is not None:
            frame_size = (int(frame_size_override[0]), int(frame_size_override[1]))
        else:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.is_potrait:
                w, h = h, w
            frame_size = (w, h)

        self.recorder = VideoRecorder(filepath, fourcc, fps, frame_size)
        self.recorder.start()

        # catat waktu (label) + elapsed timer (kontrol presisi stop)
        self.start_time = QtCore.QTime.currentTime()
        self.record_elapsed.restart()

        # tulis frame pertama agar benar-benar "langsung" mulai pada frame saat countdown selesai
        if first_frame is not None:
            try:
                self.recorder.write(first_frame)
            except Exception:
                pass

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.resolution_combo.setEnabled(False)
        self.status_bar.showMessage(f"Recording take {self.current_take}/{self.total_takes}...")

    def stop_recording(self):
        if not self.recorder:
            return

        self.recorder.stop()

        duration = 0.0
        if self.start_time:
            duration = self.start_time.msecsTo(QtCore.QTime.currentTime()) / 1000.0

        # Estimasi real fps dari timestamp write()
        if len(self.recorder.timestamps) > 1:
            real_duration = self.recorder.timestamps[-1] - self.recorder.timestamps[0]
            real_fps = len(self.recorder.timestamps) / max(real_duration, 1e-6)
        else:
            real_fps = 30.0

        self.last_saved_path = self.recorder.filepath
        self.recorder = None

        # Koreksi fps agar playback sesuai dengan real timestamp
        self.reencode_with_real_fps(self.last_saved_path, real_fps)

        # simpan metadata
        append_metadata(
            os.path.basename(self.last_saved_path),
            self.label_class,
            duration,
            self.notes_text,
            self.dataset_dir
        )

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.resolution_combo.setEnabled(True)
        self.status_bar.showMessage(f"Saved: {os.path.basename(self.last_saved_path)}")

        # reset auto flag jika stop manual
        if not self.auto_takes_running:
            self.current_take = 0
            self.total_takes = 1

    def start_auto_takes(self):
        try:
            self.total_takes = int(self.take_count_input.text())
        except ValueError:
            self.total_takes = 1

        self.auto_takes_running = True
        self.current_take = 0
        self.start_next_take()

    def start_next_take(self):
        if self.current_take >= self.total_takes:
            self.auto_takes_running = False
            self.status_bar.showMessage("Semua take selesai.")
            return

        self.current_take += 1

        # countdown 3..2..1 (default 2 detik pada versi lama)
        self.countdown_value = 3
        self.countdown_label.setText(str(self.countdown_value))
        self.countdown_label.setVisible(True)

        # blip pertama (langsung saat tampil)
        self.beep(freq=880, duration_ms=70)

        self.countdown_timer.start(1000)

    def update_countdown(self):
        self.countdown_value -= 1
        if self.countdown_value > 0:
            self.countdown_label.setText(str(self.countdown_value))
            # blip tiap detik
            self.beep(freq=880, duration_ms=70)
        else:
            self.countdown_timer.stop()
            self.countdown_label.setVisible(False)

            # suara "start"
            self.beep(freq=1200, duration_ms=160)

            # tandai agar recording dimulai tepat pada frame berikutnya (frame-aligned)
            self.pending_start_recording = True
            self.pending_start_at = time.monotonic()

    def stop_recording_and_continue(self):
        # stop current
        self.stop_recording()
        # jeda 2 detik sebelum take berikutnya
        QtCore.QTimer.singleShot(2000, self.start_next_take)

    def reencode_with_real_fps(self, filepath, real_fps):
        tmp = filepath.replace(".mp4", "_tmp.mp4")
        cap = cv2.VideoCapture(filepath)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(tmp, fourcc, float(real_fps), (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        os.replace(tmp, filepath)  # ganti file lama

    def toggle_recording(self):
        if self.stop_btn.isEnabled():
            # stop manual
            self.auto_takes_running = False
            self.stop_recording()
        else:
            # start mode auto-takes minimal 1 take
            self.start_auto_takes()

    def new_take(self):
        self.notes_edit.clear()
        self.status_bar.showMessage("New take – notes cleared")

    def update_duration(self):
        # label durasi: gunakan elapsed timer saat recording agar akurat
        if self.recorder and self.recorder.is_recording() and self.record_elapsed.isValid():
            elapsed = self.record_elapsed.elapsed() / 1000.0
            self.duration_label.setText(f"Durasi: {elapsed:.1f} detik")
        else:
            self.duration_label.setText("Durasi: 0.0 detik")

    # ---------- EVENT ----------
    def on_class_changed(self, txt):
        self.label_class = txt.strip()

    def on_duration_limit_changed(self):
        try:
            self.duration_limit = float(self.duration_limit_input.text())
        except ValueError:
            self.duration_limit = 0.0

    def on_notes_changed(self):
        self.notes_text = self.notes_edit.toPlainText()

    def on_pose_toggle(self, state):
        self.pose_enabled = (state == QtCore.Qt.Checked)
        status = "enabled" if self.pose_enabled else "disabled"
        self.status_bar.showMessage(f"Pose detection {status}")

    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Pilih Folder Penyimpanan Dataset",
            self.dataset_dir
        )
        if directory:
            self.dataset_dir = directory
            self.dir_label.setText(f"📁 {self.dataset_dir}")
            self.status_bar.showMessage(f"Dataset directory: {self.dataset_dir}")

    def preview_last(self):
        if not self.last_saved_path or not os.path.isfile(self.last_saved_path):
            QMessageBox.information(self, "Info", "Belum ada rekaman terakhir.")
            return
        cap = cv2.VideoCapture(self.last_saved_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Preview Last Video", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyWindow("Preview Last Video")

    def closeEvent(self, event):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.recorder:
                self.recorder.stop()
        except Exception:
            pass
        event.accept()
