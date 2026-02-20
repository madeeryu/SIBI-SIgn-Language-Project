import sys
import cv2
import numpy as np
import time
import threading
import queue
import requests
import json
from gtts import gTTS
import pygame
import tempfile
import os
from datetime import datetime
import csv
import threading
from collections import deque
from pathlib import Path

# ============================================================================
# KONFIGURASI - SESUAIKAN DI SINI
# ============================================================================

# Model Configuration
# Model Configuration
MODEL_PATH = "sign_v5_10kata_e5.keras"  # Path ke model Keras Anda
# CLASS_LABELS = ["aku", "mau", "makan", "diam","kamu","dia","nama","kita"]  # Label kelas dalam model
CLASS_LABELS = ["saya", "mau", "makan", "diam","kamu", "siapa", "tolong", "apa", "kenalkan","nama"] 

# Sequence Configuration (sesuaikan dengan training model Anda)
SEQUENCE_LENGTH = 30  # Jumlah frame untuk satu prediksi
# NUM_LANDMARKS = 21 * 2 + 33 + 21 * 2  # hands(21*2) + pose(33) + hands(21*2)
NUM_LANDMARKS = 1662  # hands(21*2) + pose(33) + hands(21*2)

# AI Configuration (Ollama + DeepSeek)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
AI_MODEL_NAME = "deepseek-r1:1.7b"  # Model yang ringan untuk lokal
# AI_MODEL_NAME = "qwen2.5:1.5b"

# Detection Configuration
# CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence untuk prediksi valid
CONFIDENCE_THRESHOLD = 0.95  # Minimum confidence untuk prediksi valid
SILENT_DURATION = 2.0       # Durasi "diam" untuk menganggap selesai (detik)
PREDICTION_COOLDOWN = 1   # Cooldown antar prediksi (detik)      # Minimal frame stabil sebelum diterima
MIN_STABLE_FRAMES = 50       # Minimal frame stabil sebelum diterima       # Minimal panjang kata valid
MIN_WORD_LENGTH = 10

# GUI Configuration
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
FPS_TARGET = 30

# Pose Alignment Configuration
ALIGNMENT_ENABLED = True  # Enable/disable alignment check
ALIGNMENT_OVERLAP_THRESHOLD = 0.7  # Minimum overlap ratio (0.0-1.0)
ALIGNMENT_SHOULDER_TOLERANCE = 0.15  # Shoulder position tolerance

# TTS Configuration
TTS_ENABLED = True  # Enable/disable Text-to-Speech
TTS_LANGUAGE = 'id'  # Bahasa Indonesia
TTS_WORD_DELAY = 1.0  # Jeda antar kata (detik) - dapat diubah via GUI
TTS_FINAL_ENABLED = True  # TTS kalimat final hasil AI - dapat diubah via GUI

# Logging Configuration
LOGGING_ENABLED = True
LOG_FOLDER = "logs"


# Trigger Gesture Configuration
TRIGGER_GESTURE_ENABLED = True  # Enable/disable trigger gesture
COUNTDOWN_DURATION = 3           # Countdown duration in seconds
TRIGGER_COOLDOWN = 1.0          # Cooldown after processing before accepting new trigger

# ============================================================================
# IMPORTS - GUI Libraries
# ============================================================================

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QFileDialog, QTextEdit,
        QFrame, QSplitter, QGroupBox, QProgressBar, QSlider,
        QMessageBox, QSizePolicy, QScrollArea
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
    from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QFileDialog, QTextEdit,
        QFrame, QSplitter, QGroupBox, QProgressBar, QSlider,
        QMessageBox, QSizePolicy, QScrollArea
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
    from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
    PYQT_VERSION = 5

# MediaPipe for landmark detection
import mediapipe as mp

# TensorFlow/Keras for model
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available. Model prediction disabled.")

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ============================================================================
# AI PROCESSOR CLASS
# ============================================================================

class AIProcessor:
    """Handles communication with local Ollama/DeepSeek AI"""
    
    def __init__(self):
        self.api_url = OLLAMA_API_URL
        self.model_name = AI_MODEL_NAME
        self.is_available = self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def process_predictions(self, raw_predictions):
        """Process raw predictions using AI to form coherent sentences"""
        if not self.is_available:
            return self._fallback_process(raw_predictions)
        
        # ✅ PROMPT BARU: Sangat spesifik, dengan contoh
        prompt = f"""Kamu adalah modul pasca-pemrosesan bahasa untuk aplikasi penerjemah bahasa isyarat.

            TUGAS UTAMA:
            Ubah daftar kata hasil deteksi bahasa isyarat menjadi SATU kalimat bahasa Indonesia yang:
            - alami untuk komunikasi sehari-hari
            - tetap setia pada makna input
            - tidak kaku
            - tidak menambahkan makna baru

            ========================
            INPUT
            ========================
            Daftar kata hasil deteksi (berurutan):
            {raw_predictions}

            ========================
            ATURAN KERAS (WAJIB)
            ========================
            1. Kata "diam" adalah PEMISAH → HAPUS sepenuhnya
            2. DILARANG menambahkan kata yang mengubah makna utama
            3. DILARANG menambahkan informasi baru
            4. Output hanya SATU kalimat pendek (maksimal 8 kata)
            5. Gunakan huruf kapital di awal kalimat saja
            6. TIDAK BOLEH ada typo, kata asing, atau simbol
            7. Jika ragu, pilih versi kalimat PALING NETRAL

            ========================
            ATURAN PEMBERSIHAN (WAJIB)
            ========================
            - Hapus duplikasi kata berurutan
            - Hapus kata yang jelas tidak nyambung secara konteks
            (contoh: kata tanya muncul di tengah kalimat pernyataan)
            - Jika ada noise satu kata di antara dua kata relevan, abaikan noise tersebut

            Contoh pembersihan:
            Input: ['aku', 'apa', 'mau', 'makan']
            Makna utama: Aku mau makan
            Output: Aku mau makan

            Input: ['diam', 'aku', 'diam', 'apa', 'diam', 'mau', 'diam', 'makan']
            Makna utama: Aku mau makan
            Output: Aku mau makan

            ========================
            ATURAN STRUKTUR KALIMAT
            ========================
            - Susun ulang kata HANYA jika diperlukan agar sesuai EYD
            - Pola umum:
            • Subjek – Predikat
            • Subjek – Predikat – Objek
            • Kalimat tanya sederhana

            ========================
            NATURALISASI RINGAN (OPSIONAL)
            ========================
            - Kamu BOLEH menambahkan MAKSIMAL SATU partikel percakapan ringan
            seperti: "dong", "ya" , "nih", "deh", dll.
            - HANYA jika kalimat terdengar lebih natural
            - JANGAN gunakan jika kalimat sudah jelas tanpa partikel

            Contoh:
            Input: ['tolong', 'kenalkan', 'nama', 'kamu']
            Output: Tolong kenalkan nama kamu dong

            Input: ['saya', 'mau', 'makan']
            Output: Saya mau makan

            ========================
            CONTOH FINAL
            ========================
            Input: ['aku', 'diam', 'mau', 'diam', 'makan']
            Output: Aku mau makan

            Input: ['tolong', 'diam', 'kenalkan', 'diam', 'nama', 'diam', 'kamu']
            Output: Tolong kenalkan nama kamu dong

            Input: ['kamu', 'diam', 'siapa']
            Output: Kamu siapa

            ========================
            INSTRUKSI AKHIR
            ========================
            Sekarang proses input di atas.
            Output HANYA kalimat akhirnya saja, tanpa penjelasan apa pun :"""

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1, 
                    "top_p": 0.8,
                    "top_k": 10,         # 
                    "repeat_penalty": 1.2 #
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result.get("response", "").strip()
                
                # ✅ Bersihkan output
                output = output.replace("```", "").replace("**", "").replace("*", "")
                output = output.split('\n')[0]  # Ambil baris pertama
                output = output.strip()
                
                # ✅ VALIDASI: Pastikan tidak ada kata aneh
                if output and len(output) < 100:  # Kalimat wajar
                    return output
                else:
                    print(f"⚠️ AI output terlalu panjang atau aneh: {output}")
                    return self._fallback_process(raw_predictions)
            else:
                return self._fallback_process(raw_predictions)
        except Exception as e:
            print(f"AI Error: {e}")
            return self._fallback_process(raw_predictions)
        
    def _fallback_process(self, raw_predictions):
        """Fallback processing without AI - improved logic"""
        if not raw_predictions:
            return "Tidak ada kata terdeteksi"
        
        print(f"\n🔧 Fallback processing...")
        print(f"   Input: {raw_predictions}")
        
        words = []
        prev_word = None
        
        for word in raw_predictions:
            # Skip "diam"
            if word.lower() == "diam":
                prev_word = None
                continue
            
            # Skip kata yang terlalu pendek (noise)
            if len(word) < 2:
                continue
            
            # Skip duplikasi berurutan
            if word != prev_word:
                words.append(word.lower())
                prev_word = word
        
        print(f"   After dedup: {words}")
        
        # Hapus kata berulang yang tidak berurutan
        unique_words = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        
        print(f"   After unique: {unique_words}")
        
        # Format menjadi kalimat
        if unique_words:
            sentence = " ".join(unique_words)
            result = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            print(f"   Result: {result}\n")
            return result
        
        return "Tidak ada kata terdeteksi"


# ============================================================================
# GESTURE TRIGGER DETECTOR
# ============================================================================

class GestureTriggerDetector:
    """Detects trigger gesture (peace sign with both hands)"""
    
    def __init__(self):
        self.trigger_frames_count = 0
        self.min_trigger_frames = 15  # Minimal 15 frame konsisten untuk trigger
        
    def detect_trigger(self, results):
        """
        Detect peace sign gesture (index and middle fingers up, others down)
        Returns True if both hands show peace sign
        """
        if not results.left_hand_landmarks or not results.right_hand_landmarks:
            self.trigger_frames_count = 0
            return False
        
        # Check both hands
        left_is_peace = self._is_peace_sign(results.left_hand_landmarks.landmark)
        right_is_peace = self._is_peace_sign(results.right_hand_landmarks.landmark)
        
        if left_is_peace and right_is_peace:
            self.trigger_frames_count += 1
        else:
            self.trigger_frames_count = 0
        
        # Return True if gesture stable for required frames
        return self.trigger_frames_count >= self.min_trigger_frames
    
    def _is_peace_sign(self, landmarks):
        """
        Check if hand shows peace sign:
        - Index finger (8) and middle finger (12) extended
        - Ring finger (16) and pinky (20) folded
        - Thumb can be any position
        """
        # Get landmark positions
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        wrist = landmarks[0]
        
        # Index and middle fingers should be extended (tip higher than pip)
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        
        # Ring and pinky should be folded (tip lower than pip)
        ring_folded = ring_tip.y > ring_pip.y
        pinky_folded = pinky_tip.y > pinky_pip.y
        
        return index_extended and middle_extended and ring_folded and pinky_folded
    
    def reset(self):
        """Reset trigger counter"""
        self.trigger_frames_count = 0
        

# ============================================================================
# POSE ALIGNMENT GUIDE
# ============================================================================

class PoseAlignmentGuide:
    """Guide for body pose alignment"""
    
    def __init__(self):
        # Define ROI guide (center area)
        self.roi_x_min = 0.25  # 25% from left
        self.roi_x_max = 0.75  # 75% from left (50% width)
        self.roi_y_min = 0.15  # 15% from top
        self.roi_y_max = 0.85  # 85% from top (70% height)
        
        self.is_aligned = False
        self.alignment_frames = 0
        self.min_alignment_frames = 10  # Harus aligned minimal 10 frame
        
    def check_alignment(self, results, frame_width, frame_height):
        """Check if pose is aligned with guide"""
        if not results.pose_landmarks:
            self.alignment_frames = 0
            self.is_aligned = False
            return False
        
        landmarks = results.pose_landmarks.landmark
        
        # Get shoulder landmarks (11: left shoulder, 12: right shoulder)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Check if shoulders are visible
        if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            self.alignment_frames = 0
            self.is_aligned = False
            return False
        
        # Calculate bounding box of detected pose
        x_coords = [lm.x for lm in landmarks if lm.visibility > 0.5]
        y_coords = [lm.y for lm in landmarks if lm.visibility > 0.5]
        
        if not x_coords or not y_coords:
            self.alignment_frames = 0
            self.is_aligned = False
            return False
        
        pose_x_min = min(x_coords)
        pose_x_max = max(x_coords)
        pose_y_min = min(y_coords)
        pose_y_max = max(y_coords)
        
        # Calculate overlap with ROI
        overlap_x_min = max(pose_x_min, self.roi_x_min)
        overlap_x_max = min(pose_x_max, self.roi_x_max)
        overlap_y_min = max(pose_y_min, self.roi_y_min)
        overlap_y_max = min(pose_y_max, self.roi_y_max)
        
        if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
            overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
            pose_area = (pose_x_max - pose_x_min) * (pose_y_max - pose_y_min)
            
            overlap_ratio = overlap_area / pose_area if pose_area > 0 else 0
        else:
            overlap_ratio = 0
        
        # Check shoulder alignment (should be in center horizontally)
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        is_shoulder_centered = abs(shoulder_center_x - 0.5) < ALIGNMENT_SHOULDER_TOLERANCE
        
        # Check if aligned
        aligned = (overlap_ratio >= ALIGNMENT_OVERLAP_THRESHOLD and is_shoulder_centered)
        
        if aligned:
            self.alignment_frames += 1
        else:
            self.alignment_frames = 0
        
        self.is_aligned = self.alignment_frames >= self.min_alignment_frames
        
        return self.is_aligned
    
    def draw_guide(self, frame):
        """Draw alignment guide on frame"""
        h, w = frame.shape[:2]
        
        # Calculate ROI coordinates
        x1 = int(self.roi_x_min * w)
        x2 = int(self.roi_x_max * w)
        y1 = int(self.roi_y_min * h)
        y2 = int(self.roi_y_max * h)
        
        # Draw ROI rectangle
        color = (0, 255, 0) if self.is_aligned else (0, 0, 255)
        thickness = 3 if self.is_aligned else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner markers
        marker_len = 30
        for corner in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            cx, cy = corner
            # Horizontal line
            cv2.line(frame, (cx - marker_len, cy), (cx + marker_len, cy), color, thickness)
            # Vertical line
            cv2.line(frame, (cx, cy - marker_len), (cx, cy + marker_len), color, thickness)
        
        # Draw simple silhouette guide (shoulders and body outline)
        center_x = w // 2
        shoulder_y = int(0.3 * h)
        hip_y = int(0.6 * h)
        shoulder_width = int(0.2 * w)
        
        # Shoulders
        cv2.circle(frame, (center_x - shoulder_width, shoulder_y), 8, color, -1)
        cv2.circle(frame, (center_x + shoulder_width, shoulder_y), 8, color, -1)
        
        # Body line
        cv2.line(frame, (center_x, shoulder_y), (center_x, hip_y), color, 2)
        
        # Status text
        status = "ALIGNED ✓" if self.is_aligned else "NOT ALIGNED - Adjust Position"
        status_color = (0, 255, 0) if self.is_aligned else (0, 0, 255)
        cv2.putText(frame, status, (x1 - 50, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        # Instruction text
        if not self.is_aligned:
            cv2.putText(frame, "Stand in the guide box", (x1, y2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    def reset(self):
        """Reset alignment state"""
        self.is_aligned = False
        self.alignment_frames = 0


# ============================================================================
# TTS MANAGER
# ============================================================================

class TTSManager:
    """Manages Text-to-Speech using gTTS and pygame"""
    
    def __init__(self):
        self.enabled = TTS_ENABLED
        self.language = TTS_LANGUAGE
        self.is_playing = False
        self.current_file = None
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init()
            self.available = True
            print("✅ TTS Manager initialized")
        except Exception as e:
            print(f"⚠️ TTS initialization failed: {e}")
            self.available = False
    
    def speak(self, text, callback=None):
        """Convert text to speech and play (non-blocking)"""
        if not self.enabled or not self.available or not text:
            return
        
        def _speak_thread():
            try:
                self.is_playing = True
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.close()
                self.current_file = temp_file.name
                
                print(f"\n🔊 TTS: Generating speech for '{text}'")
                
                # Generate speech
                tts = gTTS(text=text, lang=self.language, slow=False)
                tts.save(self.current_file)
                
                print(f"🔊 TTS: Playing audio...")
                
                # Play audio
                pygame.mixer.music.load(self.current_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                print(f"🔊 TTS: Playback complete\n")
                
                # Cleanup
                try:
                    os.remove(self.current_file)
                except:
                    pass
                
                self.is_playing = False
                
                if callback:
                    callback()
                    
            except Exception as e:
                print(f"❌ TTS Error: {e}")
                self.is_playing = False
                if self.current_file and os.path.exists(self.current_file):
                    try:
                        os.remove(self.current_file)
                    except:
                        pass
        
        # Run in separate thread
        thread = threading.Thread(target=_speak_thread, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop current playback"""
        try:
            pygame.mixer.music.stop()
            self.is_playing = False
        except:
            pass
    
    def set_enabled(self, enabled):
        """Enable/disable TTS"""
        self.enabled = enabled
        if not enabled:
            self.stop()


# ============================================================================
# CSV LOGGER
# ============================================================================

class CsvLogger:
    """Logger for testing and performance evaluation"""
    
    def __init__(self):
        self.enabled = LOGGING_ENABLED
        self.log_folder = Path(LOG_FOLDER)
        self.log_file = None
        self.lock = threading.Lock()
        
        if self.enabled:
            self._create_log_file()
    
    def _create_log_file(self):
        """Create new log file with timestamp"""
        self.log_folder.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_folder / f"test_{timestamp}.csv"
        
        # Write header
        header = [
            'timestamp',
            'mode',
            'aligned_status',
            'trigger_used',
            'buffer_raw',
            'buffer_filtered',
            'ai_sentence',
            'latency_buffer_to_ollama_ms',
            'latency_total_processing_ms',
            'avg_model_predict_ms',
            'confidence_last',
            'frame_fps_estimate',
            'notes',
            'word_events_json',
            'word_events_summary'
        ]
        
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        print(f"📝 Log file created: {self.log_file}")
    
    def log_inference(self, data):
        """Log a single inference event (thread-safe)"""
        if not self.enabled or not self.log_file:
            return
        
        with self.lock:
            try:
                row = [
                    data.get('timestamp', datetime.now().isoformat()),
                    data.get('mode', 'unknown'),
                    data.get('aligned_status', 'N/A'),
                    data.get('trigger_used', False),
                    str(data.get('buffer_raw', [])),
                    str(data.get('buffer_filtered', [])),
                    data.get('ai_sentence', ''),
                    data.get('latency_buffer_to_ollama_ms', 0),
                    data.get('latency_total_processing_ms', 0),
                    data.get('avg_model_predict_ms', 0),
                    data.get('confidence_last', 0),
                    data.get('frame_fps_estimate', 0),
                    data.get('notes', ''),
                    data.get('word_events_json', '[]'),
                    data.get('word_events_summary', '')
                ]
                
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    
            except Exception as e:
                print(f"❌ Logging error: {e}")
    
    def get_log_folder(self):
        """Get log folder path"""
        return str(self.log_folder.absolute())

# ============================================================================
# SIGN LANGUAGE DETECTOR CLASS
# ============================================================================

class SignLanguageDetector:
    """Main detector class for sign language recognition"""
    
    def __init__(self, use_trigger=True):
            self.model = None
            self.holistic = None
            self.sequence = []
            self.predictions_history = []
            self.sentence = []
            self.threshold = 0.5
            
            # Buffer untuk AI processing
            self.predictions_buffer = []
            self.last_prediction = None
            self.last_prediction_time = 0
            self.is_collecting = True
            self.last_diam_time = None
            self.diam_duration = 0
            
            self.stable_count = 0
            self.last_raw_prediction = None
            self.min_stable_frames = 5
            
            # Performance tracking
            self.total_predictions = 0
            self.prediction_times = []
            self.frame_counter = 0
           
            # Gesture trigger state
            self.use_trigger = use_trigger 
            self.trigger_detector = GestureTriggerDetector()
            self.is_active = not use_trigger  
            self.waiting_for_trigger = use_trigger 
            self.is_countdown = False
            self.countdown_start_time = None
            self.countdown_value = 0
            
            # Pose alignment
            self.alignment_guide = PoseAlignmentGuide()
            self.alignment_enabled = ALIGNMENT_ENABLED and use_trigger  # Only for camera mode
            
            # Performance tracking untuk logging
            self.last_confidence = 0.0
            self.buffer_finalized_time = None
            
            # Per-word detailed logging
            self.word_events = []  # list of dict per accepted word
            self._candidate_first_seen_time = None
            self._candidate_first_conf = None
            
            self._load_model()
            self._init_holistic()
    def set_min_stable_frames(self, is_video=False):
        """Set minimum stable frames based on source type"""
        if is_video:
            self.min_stable_frames = 15  # Video lebih stabil
        else:
            self.min_stable_frames = 8   # Kamera real-time lebih noise
    def _load_model(self):
        """Load the Keras model"""
        if not TF_AVAILABLE:
            return
        
        try:
            if Path(MODEL_PATH).exists():
                self.model = keras.models.load_model(MODEL_PATH)
                
                # Warm up
                dummy_input = np.zeros((1, 30, 1662), dtype=np.float32)
                for _ in range(3):
                    _ = self.model.predict(dummy_input, verbose=0)
                
                print(f"✅ Model loaded from {MODEL_PATH}")
                print(f"📊 Input shape: {self.model.input_shape}")
            else:
                print(f"⚠️ Model not found at {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def _init_holistic(self):
        """Initialize MediaPipe Holistic - OPTIMIZED"""
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0,  # 0 untuk speed, 1 untuk akurasi
            static_image_mode=False,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
    
    def extract_keypoints(self, results):
        """
        Extract keypoints - SAMA dengan Jupyter training
        Total: 33*4 + 468*3 + 21*3 + 21*3 = 132 + 1404 + 63 + 63 = 1662
        """
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33*4)
        
        face = np.array([[res.x, res.y, res.z] 
                         for res in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468*3)
        
        lh = np.array([[res.x, res.y, res.z] 
                       for res in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21*3)
        
        rh = np.array([[res.x, res.y, res.z] 
                       for res in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, face, lh, rh])
    
    def process_frame(self, frame):
            """Process a single frame with trigger gesture logic"""
            
            if self.use_trigger:  # use_trigger=True berarti mode kamera
                frame = cv2.flip(frame, 1)
            
            # MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # Draw landmarks
            annotated_frame = frame.copy()
            self._draw_landmarks(annotated_frame, results)
            
            # TAMBAHAN: Draw alignment guide if enabled
            if self.alignment_enabled:
                self.alignment_guide.draw_guide(annotated_frame)
            
            prediction = None
            confidence = 0.0
            probabilities = None
            hands_detected = (results.left_hand_landmarks is not None or 
                            results.right_hand_landmarks is not None)
            
            current_time = time.time()
            
            # TAMBAHAN: Check alignment if enabled
            is_aligned = True  # Default untuk mode tanpa alignment
            if self.alignment_enabled:
                is_aligned = self.alignment_guide.check_alignment(results, frame.shape[1], frame.shape[0])
                
                # Jika tidak aligned, blokir aktivasi
                if not is_aligned and (self.waiting_for_trigger or self.is_countdown):
                    # Tampilkan instruksi alignment
                    # cv2.putText( annotated_frame, "⚠️ ALIGN YOUR BODY FIRST", 
                    #         (50, frame.shape[0] - 50),
                    #         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Reset trigger jika countdown sudah mulai tapi tidak aligned
                    if self.is_countdown:
                        self.is_countdown = False
                        self.waiting_for_trigger = True
                        self.trigger_detector.reset()
                    
                    return annotated_frame, prediction, confidence, results, probabilities, hands_detected
            
            
            # ============================================================
            # LOGIKA UTAMA: Trigger Gesture & Countdown (HANYA UNTUK CAMERA)
            # ============================================================
            
            current_time = time.time()
            
            # ✅ SKIP TRIGGER LOGIC JIKA MODE VIDEO
            if self.use_trigger:
                # STATE 1: Menunggu Trigger Gesture
                if self.waiting_for_trigger:
                    trigger_detected = self.trigger_detector.detect_trigger(results)
                    
                    if trigger_detected:
                        print("\n🎯 Trigger gesture detected! Starting countdown...")
                        self.waiting_for_trigger = False
                        self.is_countdown = True
                        self.countdown_start_time = current_time
                        self.countdown_value = COUNTDOWN_DURATION
                        self.trigger_detector.reset()
                    
                    # Tampilkan status di frame
                    status_text = "Show PEACE SIGN (both hands) to start"
                    cv2.putText(annotated_frame, status_text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    return annotated_frame, prediction, confidence, results, probabilities, hands_detected
                
                # STATE 2: Countdown
                if self.is_countdown:
                    elapsed = current_time - self.countdown_start_time
                    self.countdown_value = max(0, COUNTDOWN_DURATION - int(elapsed))
                    
                    if self.countdown_value > 0:
                        # Tampilkan countdown besar di tengah
                        text = str(self.countdown_value)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 5
                        thickness = 10
                        
                        # Get text size
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Center position
                        x = (frame.shape[1] - text_width) // 2
                        y = (frame.shape[0] + text_height) // 2
                        
                        # Draw countdown dengan efek
                        cv2.putText(annotated_frame, text, (x, y),
                                font, font_scale, (0, 0, 0), thickness + 4)  # Shadow
                        cv2.putText(annotated_frame, text, (x, y),
                                font, font_scale, (0, 255, 0), thickness)  # Main text
                        
                        # Status text
                        status_text = "Get Ready!"
                        cv2.putText(annotated_frame, status_text, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                        
                        return annotated_frame, prediction, confidence, results, probabilities, hands_detected
                    else:
                        # Countdown selesai, aktifkan model
                        print("✅ Countdown complete! Model ACTIVATED\n")
                        self.is_countdown = False
                        self.is_active = True
                        self.sequence = []  # Reset sequence
                        self.predictions_buffer = []
                        self.sentence = []
            
            # STATE 3: Model Aktif - Proses Gesture
            if self.is_active:
                # Extract keypoints
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]
                
                # Debug first frame
                if self.total_predictions == 0 and len(self.sequence) > 0:
                    print(f"\n🔍 First Frame Debug:")
                    print(f"   Keypoints shape: {keypoints.shape}")
                    print(f"   Non-zero ratio: {(keypoints != 0).sum() / keypoints.size:.2%}")
                    print(f"   Hands detected: L={results.left_hand_landmarks is not None}, R={results.right_hand_landmarks is not None}")
                    print(f"   Face detected: {results.face_landmarks is not None}")
                    print()
                
                # Predict when we have 30 frames
                if len(self.sequence) == 30 and self.model is not None:
                    try:
                        start_time = time.time()
                        
                        # Prepare input
                        sequence_array = np.array(self.sequence, dtype=np.float32)
                        sequence_input = np.expand_dims(sequence_array, axis=0)
                        
                        # Debug shape pada prediksi pertama
                        if self.total_predictions == 0:
                            print(f"🔍 Prediction Debug:")
                            print(f"   Sequence shape: {sequence_array.shape}")
                            print(f"   Model input shape: {sequence_input.shape}")
                            print(f"   Expected: (1, 30, 1662)\n")
                        
                        # Predict
                        res = self.model.predict(sequence_input, verbose=0)[0]
                        probabilities = res
                        
                        pred_idx = np.argmax(res)
                        confidence = float(res[pred_idx])
                        prediction = CLASS_LABELS[pred_idx]
                        self.last_confidence = confidence
                        
                        # Track time
                        pred_time = time.time() - start_time
                        self.prediction_times.append(pred_time)
                        if len(self.prediction_times) > 50:
                            self.prediction_times = self.prediction_times[-50:]
                        
                        self.total_predictions += 1
                        
                        # Print probabilities setiap 30 prediksi
                        if self.total_predictions % 30 == 0:
                            print(f"\n📊 Prediction #{self.total_predictions}:")
                            for i, label in enumerate(CLASS_LABELS):
                                bar = "█" * int(res[i] * 30)
                                print(f"   {label:6s}: {res[i]:.4f} {bar}")
                            print()
                        
                        # Cek apakah prediksi valid (di atas threshold)
                        if confidence > self.threshold:
                            
                            # STABILITAS: Cek apakah prediksi sama dengan sebelumnya
                            if prediction == self.last_raw_prediction:
                                self.stable_count += 1
                            else:
                                self.stable_count = 1
                                self.last_raw_prediction = prediction
                                # Track when this candidate word was first seen (for per-word buffer latency)
                                self._candidate_first_seen_time = current_time
                                self._candidate_first_conf = confidence
                            
                            # HANYA PROSES jika sudah stabil minimal X frame
                            if self.stable_count >= self.min_stable_frames:
                                
                                # WORD GROUPING: Hanya tambahkan jika BERBEDA dengan kata terakhir di buffer
                                if len(self.predictions_buffer) == 0 or prediction != self.predictions_buffer[-1]:
                                    self.predictions_buffer.append(prediction)

                                    # Record per-word event (first seen -> accepted into buffer)
                                    first_seen = self._candidate_first_seen_time if self._candidate_first_seen_time else current_time
                                    event = {
                                        "word": prediction,
                                        "conf_accept": float(confidence),
                                        "conf_first": float(self._candidate_first_conf) if self._candidate_first_conf is not None else float(confidence),
                                        "first_seen_ts": float(first_seen),
                                        "accepted_ts": float(current_time),
                                        "delay_to_buffer_ms": round((current_time - first_seen) * 1000.0, 2),
                                        "stable_frames": int(self.stable_count)
                                    }
                                    self.word_events.append(event)

                                    self.last_prediction = prediction
                                    self.last_prediction_time = current_time
                                    
                                    print(f"✅ New word added: {prediction} (conf: {confidence:.3f}, stable: {self.stable_count} frames)")
                                    print(f"   Buffer: {self.predictions_buffer}")
                                
                                # Update sentence untuk tampilan real-time (tanpa duplikasi)
                                if len(self.sentence) == 0 or prediction != self.sentence[-1]:
                                    self.sentence.append(prediction)
                                    if len(self.sentence) > 10:
                                        self.sentence = self.sentence[-10:]
                                
                                # TRACKING "diam" untuk mendeteksi selesai
                                if prediction == "diam":
                                    if self.last_diam_time is None:
                                        self.last_diam_time = current_time
                                    else:
                                        self.diam_duration = current_time - self.last_diam_time
                                        
                                        # Jika "diam" sudah >= SILENT_DURATION, tandai selesai
                                        if self.diam_duration >= SILENT_DURATION and self.is_collecting:
                                            self.is_collecting = False
                                            self.buffer_finalized_time = time.time()
                                            print(f"\n⏸️ Gesture sequence completed! (silent {self.diam_duration:.1f}s)")
                                            print(f"   Final buffer: {self.predictions_buffer}")
                                            
                                            # NONAKTIFKAN MODEL
                                            print("🔴 Model DEACTIVATED during processing\n")
                                            self.is_active = False
                                else:
                                    # Reset diam tracking jika bukan "diam"
                                    self.last_diam_time = None
                                    self.diam_duration = 0
                                    if not self.is_collecting:
                                        self.is_collecting = True

                        else:
                            # RESET stabilitas jika confidence rendah
                            self.stable_count = 0
                            self.last_raw_prediction = None
                            self._candidate_first_seen_time = None
                            self._candidate_first_conf = None
                        
                        # Performance stats
                        if self.total_predictions % 50 == 0:
                            avg_time = np.mean(self.prediction_times) * 1000
                            print(f"⚡ Avg: {avg_time:.1f}ms ({1000/avg_time:.1f} pred/s)")
                        
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Tampilkan status ACTIVE
                status_text = "MODEL ACTIVE - Recording..." if self.use_trigger else "VIDEO MODE - Recording..."
                cv2.putText(annotated_frame, status_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # STATE 4: Model Nonaktif (Processing) - HANYA UNTUK CAMERA MODE
            elif self.use_trigger and not self.is_active and not self.waiting_for_trigger:
                # Tampilkan status PROCESSING
                cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.putText(annotated_frame, "PROCESSING...", 
                        (frame.shape[1]//2 - 200, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 4)
                cv2.putText(annotated_frame, "Show PEACE SIGN to continue", 
                        (frame.shape[1]//2 - 300, frame.shape[0]//2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            return annotated_frame, prediction, confidence, results, probabilities, hands_detected
    
    def _draw_landmarks(self, frame, results):
        """Draw landmarks - simplified"""
        # Draw hands
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2)
            )
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
            )
    
    def get_current_sentence(self):
        """Get current sentence"""
        return ' '.join(self.sentence)
    
    def reset(self):
            """Reset detector"""
            self.sequence = []
            self.predictions_history = []
            self.sentence = []
            self.total_predictions = 0
            self.prediction_times = []
            self.frame_counter = 0
            
            # Reset state
            self.predictions_buffer = []
            self.last_prediction = None
            self.last_prediction_time = 0
            self.is_collecting = True
            self.last_diam_time = None
            self.diam_duration = 0
            self.stable_count = 0
            self.last_raw_prediction = None
            self._candidate_first_seen_time = None
            self._candidate_first_conf = None
            self.word_events = []
            
            # Reset trigger state
            self.trigger_detector.reset()
            self.is_active = not self.use_trigger  
            self.waiting_for_trigger = self.use_trigger  # ✅ Camera mode tunggu trigger
            self.is_countdown = False
            self.countdown_start_time = None
            self.countdown_value = 0
    def _draw_landmarks(self, frame, results):
        """Draw MediaPipe landmarks on frame"""
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw left hand
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        # Draw right hand
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
    
    def get_buffer_status(self):
        """Get current buffer status"""
        return {
            'buffer': self.predictions_buffer.copy(),
            'is_collecting': self.is_collecting,
            'diam_duration': self.diam_duration,
            'buffer_count': len(self.predictions_buffer)
        }

    def get_buffer_and_reset(self):
        """Get prediction buffer + per-word events and reset for new collection"""
        buffer = self.predictions_buffer.copy()
        events = self.word_events.copy()

        # ✅ RESET SEMUA STATE
        self.predictions_buffer = []
        self.word_events = []
        self.sentence = []
        self.is_collecting = True
        self.last_diam_time = None
        self.diam_duration = 0
        self.last_prediction = None
        self.last_prediction_time = 0

        # ✅ TAMBAHAN: Reset stabilitas untuk menghindari carry-over
        self.stable_count = 0
        self.last_raw_prediction = None
        self._candidate_first_seen_time = None
        self._candidate_first_conf = None

        print("🔄 Buffer reset. Ready for new sequence.")

        return buffer, events
    def reactivate_after_processing(self):
            """Reactivate model after processing"""
            if self.use_trigger:
                # Camera mode: tunggu trigger gesture lagi
                print("🔄 Ready for new gesture sequence. Waiting for trigger...\n")
                self.waiting_for_trigger = True
                self.is_active = False
                self.trigger_detector.reset()
            else:
                # Video mode: langsung aktif lagi tanpa trigger
                print("🔄 Ready for new gesture sequence. Model reactivated.\n")
                self.is_active = True
                self.waiting_for_trigger = False
            
            self.is_collecting = True
            
    def should_process_ai(self):
        """Check if buffer should be processed by AI"""
        return not self.is_collecting and len(self.predictions_buffer) > 0

# ============================================================================
# VIDEO/CAMERA WORKER THREAD
# ============================================================================

class VideoWorker(QThread):
    """Worker thread for video/camera processing"""
    
    frame_ready = pyqtSignal(np.ndarray, str, float, bool, object)
    ai_result_ready = pyqtSignal(str, list)
    status_update = pyqtSignal(str)
    buffer_update = pyqtSignal(list, int, float)
    word_spoken = pyqtSignal(str)  # Signal: kata yang baru saja di-TTS
    final_sentence_spoken = pyqtSignal(str)  # Signal: kalimat final selesai di-TTS
    
    def __init__(self):
        super().__init__()
        # self.detector = SignLanguageDetector()
        self.detector = None
        self.ai_processor = AIProcessor()
        self.tts_manager = TTSManager()
        self.csv_logger = CsvLogger()
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.source_type = "camera"  # "camera" or "video"
        self.camera_index = 0
        self.video_path = None
        self.ai_processing = False
        self.trigger_detector = GestureTriggerDetector()
        self.is_active = False  # Model aktif/nonaktif
        self.waiting_for_trigger = True  # Menunggu gesture trigger
        self.is_countdown = False  # Sedang countdown
        self.countdown_start_time = None
        self.countdown_value = 0
        self.buffer_start_time = None
        self.ai_start_time = None
        self.tts_start_time = None
        self.tts_end_time = None
        self.tts_word_delay = TTS_WORD_DELAY  # Jeda antar kata TTS (detik)
        self._last_spoken_words = []  # Track kata yang sudah di-TTS dari buffer
        self.tts_final_enabled = TTS_FINAL_ENABLED  # TTS kalimat final hasil AI
    def set_camera(self, index):
        """Set camera source"""
        self.source_type = "camera"
        self.camera_index = index
        
    def set_video(self, path):
        """Set video source"""
        self.source_type = "video"
        self.video_path = path
    
    

    def run(self):
        """Main processing loop"""
        self.is_running = True
        
        # ✅ INISIALISASI DETECTOR BERDASARKAN MODE
        if self.source_type == "camera":
            self.detector = SignLanguageDetector(use_trigger=True)  # Camera pakai trigger
            self.detector.set_min_stable_frames(is_video=False)
            self.cap = cv2.VideoCapture(self.camera_index)
            self.status_update.emit(f"Camera {self.camera_index} opened")
        else:
            self.detector = SignLanguageDetector(use_trigger=False)  # Video tanpa trigger
            self.detector.set_min_stable_frames(is_video=True)
            self.cap = cv2.VideoCapture(self.video_path)
            self.status_update.emit(f"Video loaded: {Path(self.video_path).name}")
        
        if not self.cap.isOpened():
            self.status_update.emit("Error: Cannot open video source")
            return
        
        frame_delay = 1.0 / FPS_TARGET
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Process frame
            annotated_frame, prediction, confidence, results, probabilities, hands_detected = self.detector.process_frame(frame)
            
            # Emit frame with probabilities
            self.frame_ready.emit(annotated_frame, prediction or "", confidence, hands_detected, probabilities)
            
            # TAMBAHAN: Update buffer display
            buffer_status = self.detector.get_buffer_status()
            self.buffer_update.emit(
                buffer_status['buffer'], 
                buffer_status['buffer_count'],
                buffer_status['diam_duration']
            )
            
            # TAMBAHAN: TTS per-kata - ucapkan setiap kata baru yang masuk buffer
            if (self.tts_manager.enabled and 
                    buffer_status['is_collecting'] and 
                    not self.tts_manager.is_playing):
                current_buffer = buffer_status['buffer']
                new_words = [w for w in current_buffer if w not in self._last_spoken_words]
                if new_words:
                    word_to_speak = new_words[0]
                    self._last_spoken_words.append(word_to_speak)
                    word_delay = self.tts_word_delay
                    
                    def _speak_word(w, delay):
                        self.tts_manager.speak(w)
                        # Tunggu sampai selesai lalu jeda
                        t0 = time.time()
                        while self.tts_manager.is_playing and (time.time() - t0) < 10:
                            time.sleep(0.05)
                        time.sleep(delay)
                        self.word_spoken.emit(w)
                    
                    threading.Thread(target=_speak_word, args=(word_to_speak, word_delay), daemon=True).start()
            
            # TAMBAHAN: Check if buffer should be processed by AI
            if self.detector.should_process_ai() and not self.ai_processing:
                self.ai_processing = True
                self.status_update.emit("Processing with AI...")
                
                # Get buffer
                buffer_raw, word_events = self.detector.get_buffer_and_reset()
                self._last_spoken_words = []  # Reset daftar kata yang sudah di-TTS
                
                # Process with AI in background thread
                threading.Thread(
                    target=self._process_with_ai,
                    args=(buffer_raw, word_events,),
                    daemon=True
                ).start()
            
            # Frame rate control
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
        
        if self.cap:
            self.cap.release()

    def _process_with_ai(self, buffer_raw, word_events):
            """Process buffer with AI (runs in separate thread)"""
            try:
                # Record start time
                self.ai_start_time = time.time()
                
                # Validasi buffer tidak kosong
                if not buffer_raw or len(buffer_raw) == 0:
                    self.status_update.emit("Buffer kosong, tidak ada yang diproses")
                    self.detector.reactivate_after_processing()
                    return
                
                # Filter buffer dari kata yang terlalu pendek
                filtered_buffer = [word for word in buffer_raw if len(word) >= 2]
                # Keep per-word events aligned with filtered buffer
                filtered_events = [e for e in (word_events or [])
                                   if isinstance(e, dict) and len(str(e.get('word', ''))) >= 2]
                
                if not filtered_buffer:
                    self.status_update.emit("Buffer tidak valid setelah filtering")
                    self.detector.reactivate_after_processing()
                    return
                
                print(f"\n🤖 Processing with AI...")
                print(f"   Raw buffer: {buffer_raw}")
                print(f"   Filtered: {filtered_buffer}")
                
                result = self.ai_processor.process_predictions(filtered_buffer)
                
                # Calculate latency: buffer finalized → Ollama response received
                ai_end_time = time.time()
                latency_buffer_to_ollama = 0
                if self.detector.buffer_finalized_time:
                    latency_buffer_to_ollama = (ai_end_time - self.detector.buffer_finalized_time) * 1000
                
                print(f"   AI Result: {result}")
                print(f"   Latency (buffer→Ollama): {latency_buffer_to_ollama:.1f}ms\n")
                
                self.ai_result_ready.emit(result, filtered_buffer)
                self.status_update.emit("AI processing complete ✅")
                
                # TTS: Ucapkan kalimat final hasil AI (jika diaktifkan)
                self.tts_start_time = time.time()
                if self.tts_final_enabled and self.tts_manager.enabled and result:
                    print(f"🔊 TTS Final: '{result}'")
                    
                    def _speak_final():
                        # Tunggu jika masih ada TTS per-kata yang berjalan
                        wait_t = time.time()
                        while self.tts_manager.is_playing and (time.time() - wait_t) < 15:
                            time.sleep(0.1)
                        
                        self.tts_manager.speak(result)
                        # Tunggu selesai
                        wait_t2 = time.time()
                        while self.tts_manager.is_playing and (time.time() - wait_t2) < 15:
                            time.sleep(0.1)
                        
                        self.tts_end_time = time.time()
                        self.final_sentence_spoken.emit(result)
                        self._log_and_reactivate(buffer_raw, filtered_buffer, result, latency_buffer_to_ollama, filtered_events)
                    
                    threading.Thread(target=_speak_final, daemon=True).start()
                else:
                    self.tts_end_time = self.tts_start_time
                    self._log_and_reactivate(buffer_raw, filtered_buffer, result, latency_buffer_to_ollama, filtered_events)
                
            except Exception as e:
                print(f"AI Processing Error: {e}")
                import traceback
                traceback.print_exc()
                self.status_update.emit(f"AI Error: {e}")
                self.detector.reactivate_after_processing()
                self.ai_processing = False
        
    def _on_tts_complete(self):
        """Callback when TTS playback is complete"""
        self.tts_end_time = time.time()
        print(f"🔊 TTS playback finished\n")
        
        # Now we can calculate total latency and log
        # This will be called from _log_and_reactivate
    

    def _summarize_word_events(self, events):
        """Create a compact human-readable summary for CSV."""
        try:
            if not events:
                return ""
            parts = []
            for e in events:
                if not isinstance(e, dict):
                    continue
                w = str(e.get("word", ""))
                conf = e.get("conf_accept", None)
                delay = e.get("delay_to_buffer_ms", None)
                if conf is None and delay is None:
                    parts.append(w)
                else:
                    conf_s = f"{conf:.3f}" if isinstance(conf, (int, float)) else str(conf)
                    delay_s = f"{delay:.1f}ms" if isinstance(delay, (int, float)) else str(delay)
                    parts.append(f"{w}({conf_s},{delay_s})")
            return " | ".join(parts)
        except Exception:
            return ""
    def _log_and_reactivate(self, buffer_raw, buffer_filtered, ai_sentence, latency_buffer_to_ollama, word_events):
        """Log the inference and reactivate detector"""
        try:
            # Calculate total latency: buffer finalized → TTS complete
            latency_total = 0
            if self.detector.buffer_finalized_time and self.tts_end_time:
                latency_total = (self.tts_end_time - self.detector.buffer_finalized_time) * 1000
            
            # Calculate average model prediction time
            avg_predict_ms = 0
            if self.detector.prediction_times:
                avg_predict_ms = np.mean(self.detector.prediction_times) * 1000
            
            # Estimate FPS
            fps_estimate = 1000 / avg_predict_ms if avg_predict_ms > 0 else 0
            
            # Get alignment status
            aligned_status = 'N/A'
            if self.detector.alignment_enabled:
                aligned_status = 'ALIGNED' if self.detector.alignment_guide.is_aligned else 'NOT_ALIGNED'
            
            # Log to CSV
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'mode': self.source_type,
                'aligned_status': aligned_status,
                'trigger_used': self.detector.use_trigger,
                'buffer_raw': buffer_raw,
                'buffer_filtered': buffer_filtered,
                'ai_sentence': ai_sentence,
                'latency_buffer_to_ollama_ms': round(latency_buffer_to_ollama, 2),
                'latency_total_processing_ms': round(latency_total, 2),
                'avg_model_predict_ms': round(avg_predict_ms, 2),
                'confidence_last': round(self.detector.last_confidence, 3),
                'frame_fps_estimate': round(fps_estimate, 1),
                'notes': '',
                'word_events_json': json.dumps(word_events or [], ensure_ascii=False),
                'word_events_summary': self._summarize_word_events(word_events)
            }
            
            self.csv_logger.log_inference(log_data)
            print(f"📝 Logged to CSV: {self.csv_logger.log_file}")
            
        except Exception as e:
            print(f"❌ Logging error: {e}")
        
        finally:
            # Delay sebelum reactivate
            time.sleep(TRIGGER_COOLDOWN)
            
            # Reactivate untuk menunggu trigger baru
            self.detector.reactivate_after_processing()
            self.status_update.emit("Ready for new gesture - Show peace sign")
            self.ai_processing = False
    
    def stop(self):
        """Stop the worker"""
        self.is_running = False
        self.wait()
    
    def pause(self):
        """Pause processing"""
        self.is_paused = True
    
    def resume(self):
        """Resume processing"""
        self.is_paused = False
    
    def reset_detector(self):
            """Reset the detector"""
            if self.detector:
                self.detector.reset()

# ============================================================================
# MAIN GUI WINDOW
# ============================================================================

class SignLanguageGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_mode = None
        self.video_path = None
        
        self._setup_ui()
        self._setup_connections()
        self._check_ai_status()
    
    def _setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Sign Language Detection - LSTM Model")
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(self._get_stylesheet())
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Left sidebar
        left_sidebar = self._create_left_sidebar()
        main_layout.addWidget(left_sidebar, stretch=1)
        
        # Center video display
        video_container = self._create_video_display()
        main_layout.addWidget(video_container, stretch=3)
        
        # Right sidebar
        right_sidebar = self._create_right_sidebar()
        main_layout.addWidget(right_sidebar, stretch=1)
    
    def _create_left_sidebar(self):
        """Create left sidebar with controls"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(270)

        # Wrap semua konten dalam QScrollArea agar parameter bisa diakses
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame if PYQT_VERSION == 6 else QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff if PYQT_VERSION == 6 else Qt.ScrollBarAlwaysOff
        )
        # Transparan agar background QFrame#sidebar tetap terlihat
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }
        """)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("📹 Input Source")
        title.setObjectName("sidebarTitle")
        layout.addWidget(title)
        
        # Camera Section
        camera_group = QGroupBox("Camera Mode")
        camera_layout = QVBoxLayout(camera_group)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Camera {i}" for i in range(5)])
        camera_layout.addWidget(QLabel("Select Camera:"))
        camera_layout.addWidget(self.camera_combo)
        
        self.btn_start_camera = QPushButton("🎥 Start Camera")
        self.btn_start_camera.setObjectName("primaryButton")
        camera_layout.addWidget(self.btn_start_camera)
        
        layout.addWidget(camera_group)
        
        # Video Section
        video_group = QGroupBox("Video Mode")
        video_layout = QVBoxLayout(video_group)
        
        self.video_label = QLabel("No video selected")
        self.video_label.setWordWrap(True)
        self.video_label.setStyleSheet("color: #888; font-style: italic;")
        video_layout.addWidget(self.video_label)
        
        self.btn_select_video = QPushButton("📁 Select Video")
        video_layout.addWidget(self.btn_select_video)
        
        self.btn_start_video = QPushButton("▶️ Play Video")
        self.btn_start_video.setObjectName("primaryButton")
        self.btn_start_video.setEnabled(False)
        video_layout.addWidget(self.btn_start_video)
        
        layout.addWidget(video_group)
        
        # Control Buttons
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(6)
        
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet("font-size: 12px;")
        control_layout.addWidget(self.btn_pause)
        
        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("font-size: 12px;")
        control_layout.addWidget(self.btn_stop)
        
        self.btn_clear = QPushButton("🗑️ Clear All")
        self.btn_clear.setStyleSheet("font-size: 12px;")
        control_layout.addWidget(self.btn_clear)
        
        layout.addWidget(control_group)
        # Status
        self.status_label = QLabel("Status: Ready")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)
        
        # AI Status  (dipindah ke panel atas tengah, tapi tetap ada di sini sebagai referensi tersembunyi)
        self.ai_status_label = QLabel("AI: Checking...")
        self.ai_status_label.setObjectName("aiStatusLabel")
        self.ai_status_label.hide()  # disembunyikan — ditampilkan di atas panel tengah
        layout.addWidget(self.ai_status_label)

        # Tools — TTS ON/OFF dipindah ke sini dari panel kanan
        tools_group = QGroupBox("🛠️ Tools")
        tools_layout = QVBoxLayout(tools_group)
        tools_layout.setSpacing(6)
        
        # TTS Toggle (dipindah dari panel kanan)
        self.chk_tts = QPushButton("🔊 TTS: ON")
        self.chk_tts.setCheckable(True)
        self.chk_tts.setChecked(TTS_ENABLED)
        self.chk_tts.setStyleSheet("font-size: 12px;")
        self.chk_tts.clicked.connect(self._toggle_tts)
        tools_layout.addWidget(self.chk_tts)
        
        self.btn_tts_final = QPushButton(f"🔊 TTS Kalimat Final: {'ON' if TTS_FINAL_ENABLED else 'OFF'}")
        self.btn_tts_final.setCheckable(True)
        self.btn_tts_final.setChecked(TTS_FINAL_ENABLED)
        self.btn_tts_final.setStyleSheet("font-size: 12px;")
        self.btn_tts_final.clicked.connect(self._toggle_tts_final)
        tools_layout.addWidget(self.btn_tts_final)

        # Alignment Toggle (only for camera)
        self.chk_alignment = QPushButton("📐 Alignment: ON")
        self.chk_alignment.setCheckable(True)
        self.chk_alignment.setChecked(ALIGNMENT_ENABLED)
        self.chk_alignment.setStyleSheet("font-size: 12px;")
        self.chk_alignment.clicked.connect(self._toggle_alignment)
        tools_layout.addWidget(self.chk_alignment)
        
        # Open logs folder
        self.btn_open_logs = QPushButton("📂 Open Logs Folder")
        self.btn_open_logs.setStyleSheet("font-size: 12px;")
        self.btn_open_logs.clicked.connect(self._open_logs_folder)
        tools_layout.addWidget(self.btn_open_logs)
        
        layout.addWidget(tools_group)

        # Parameters — slider untuk setiap parameter yang bisa dikustom
        param_group = QGroupBox("⚙️ Parameters")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(6)

        # CONFIDENCE_THRESHOLD  0.50 – 1.00  (disimpan x100)
        self.lbl_conf_thresh = QLabel(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.2f}")
        self.lbl_conf_thresh.setStyleSheet("color: #ccc; font-size: 12px;")
        self.slider_conf_thresh = QSlider(
            Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        self.slider_conf_thresh.setMinimum(50)
        self.slider_conf_thresh.setMaximum(100)
        self.slider_conf_thresh.setValue(int(CONFIDENCE_THRESHOLD * 100))
        self.slider_conf_thresh.setTickInterval(5)
        self.slider_conf_thresh.valueChanged.connect(self._on_conf_thresh_changed)
        param_layout.addWidget(self.lbl_conf_thresh)
        param_layout.addWidget(self.slider_conf_thresh)

        # SILENT_DURATION  0.5 – 10.0 s  (disimpan x10)
        self.lbl_silent_dur = QLabel(f"Silent Duration: {SILENT_DURATION:.1f}s")
        self.lbl_silent_dur.setStyleSheet("color: #ccc; font-size: 12px;")
        self.slider_silent_dur = QSlider(
            Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        self.slider_silent_dur.setMinimum(5)
        self.slider_silent_dur.setMaximum(100)
        self.slider_silent_dur.setValue(int(SILENT_DURATION * 10))
        self.slider_silent_dur.setTickInterval(10)
        self.slider_silent_dur.valueChanged.connect(self._on_silent_dur_changed)
        param_layout.addWidget(self.lbl_silent_dur)
        param_layout.addWidget(self.slider_silent_dur)

        # PREDICTION_COOLDOWN  0.1 – 5.0 s  (disimpan x10)
        self.lbl_pred_cooldown = QLabel(f"Prediction Cooldown: {PREDICTION_COOLDOWN:.1f}s")
        self.lbl_pred_cooldown.setStyleSheet("color: #ccc; font-size: 12px;")
        self.slider_pred_cooldown = QSlider(
            Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        self.slider_pred_cooldown.setMinimum(1)
        self.slider_pred_cooldown.setMaximum(50)
        self.slider_pred_cooldown.setValue(int(PREDICTION_COOLDOWN * 10))
        self.slider_pred_cooldown.setTickInterval(5)
        self.slider_pred_cooldown.valueChanged.connect(self._on_pred_cooldown_changed)
        param_layout.addWidget(self.lbl_pred_cooldown)
        param_layout.addWidget(self.slider_pred_cooldown)

        # MIN_STABLE_FRAMES  5 – 100
        self.lbl_min_stable = QLabel(f"Min Stable Frames: {MIN_STABLE_FRAMES}")
        self.lbl_min_stable.setStyleSheet("color: #ccc; font-size: 12px;")
        self.slider_min_stable = QSlider(
            Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        self.slider_min_stable.setMinimum(5)
        self.slider_min_stable.setMaximum(100)
        self.slider_min_stable.setValue(MIN_STABLE_FRAMES)
        self.slider_min_stable.setTickInterval(10)
        self.slider_min_stable.valueChanged.connect(self._on_min_stable_changed)
        param_layout.addWidget(self.lbl_min_stable)
        param_layout.addWidget(self.slider_min_stable)

        # MIN_WORD_LENGTH  1 – 30
        self.lbl_min_word_len = QLabel(f"Min Word Length: {MIN_WORD_LENGTH}")
        self.lbl_min_word_len.setStyleSheet("color: #ccc; font-size: 12px;")
        self.slider_min_word_len = QSlider(
            Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        self.slider_min_word_len.setMinimum(1)
        self.slider_min_word_len.setMaximum(30)
        self.slider_min_word_len.setValue(MIN_WORD_LENGTH)
        self.slider_min_word_len.setTickInterval(5)
        self.slider_min_word_len.valueChanged.connect(self._on_min_word_len_changed)
        param_layout.addWidget(self.lbl_min_word_len)
        param_layout.addWidget(self.slider_min_word_len)

        # TTS Word Delay — slider dipindah ke sini (digabungkan dalam Parameters)
        self.lbl_tts_delay = QLabel(f"TTS Word Delay: {TTS_WORD_DELAY:.1f}s")
        self.lbl_tts_delay.setStyleSheet("color: #ccc; font-size: 12px;")
        self.slider_tts_delay = QSlider(
            Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        self.slider_tts_delay.setMinimum(0)
        self.slider_tts_delay.setMaximum(30)
        self.slider_tts_delay.setValue(int(TTS_WORD_DELAY * 10))
        self.slider_tts_delay.setTickInterval(5)
        self.slider_tts_delay.valueChanged.connect(self._on_tts_delay_changed)
        param_layout.addWidget(self.lbl_tts_delay)
        param_layout.addWidget(self.slider_tts_delay)

        layout.addWidget(param_group)
        
        layout.addStretch()

        scroll.setWidget(inner)

        # Pasang scroll ke dalam sidebar frame
        outer_layout = QVBoxLayout(sidebar)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(scroll)
        
        return sidebar
    
    def _create_video_display(self):
        """Create center video display area"""
        container = QFrame()
        container.setObjectName("videoContainer")
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── INFO BAR ATAS TENGAH (Ollama info + aktivitas) ────────────
        # Dipindah dari panel kiri ke sini sesuai perintah
        top_info_bar = QFrame()
        top_info_bar.setObjectName("infoBar")
        top_info_layout = QHBoxLayout(top_info_bar)
        top_info_layout.setContentsMargins(10, 6, 10, 6)

        self.top_ai_status_label = QLabel("AI: Checking...")
        self.top_ai_status_label.setObjectName("aiStatusLabel")
        self.top_ai_status_label.setStyleSheet(
            "color: #FF9800; font-size: 12px; font-weight: bold; padding: 3px 8px;")
        top_info_layout.addWidget(self.top_ai_status_label)

        top_info_layout.addStretch()

        self.top_activity_label = QLabel("⏳ Menunggu aktivitas...")
        self.top_activity_label.setStyleSheet(
            "color: #cccccc; font-size: 12px; padding: 3px 8px;")
        top_info_layout.addWidget(self.top_activity_label)

        layout.addWidget(top_info_bar)

        # Video display label
        self.video_display = QLabel()
        self.video_display.setObjectName("videoDisplay")
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setText("Select Camera or Video to start detection")
        self.video_display.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #4a4a6a;
                border-radius: 10px;
                color: #888;
                font-size: 18px;
            }
        """)
        layout.addWidget(self.video_display)
        
        # Info bar below video
        info_bar = QFrame()
        info_bar.setObjectName("infoBar")
        info_layout = QHBoxLayout(info_bar)
        
        self.lbl_prediction = QLabel("Prediction: -")
        self.lbl_prediction.setObjectName("predictionLabel")
        info_layout.addWidget(self.lbl_prediction)
        
        self.lbl_confidence = QLabel("Confidence: -")
        self.lbl_confidence.setObjectName("confidenceLabel")
        info_layout.addWidget(self.lbl_confidence)
        
        self.lbl_hands = QLabel("Hands: Not Detected")
        self.lbl_hands.setObjectName("handsLabel")
        info_layout.addWidget(self.lbl_hands)
        
        layout.addWidget(info_bar)
        
        # Confidence progress bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setObjectName("confidenceBar")
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("Confidence: %p%")
        layout.addWidget(self.confidence_bar)
        
        # System status label
        self.system_status_label = QLabel("System: Waiting for trigger gesture")
        self.system_status_label.setObjectName("systemStatusLabel")
        self.system_status_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a4e;
                color: #FFD700;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
                border: 2px solid #4a4a6a;
            }
        """)
        layout.addWidget(self.system_status_label)
        
        # Probability bars — hanya tampilkan 3 kata dengan confidence tertinggi
        prob_group = QFrame()
        prob_group.setObjectName("probGroup")
        prob_layout = QVBoxLayout(prob_group)
        prob_layout.setContentsMargins(5, 5, 5, 5)

        self.prob_bars = {}
        self.prob_bar_labels = {}
        for i in range(3):
            bar_container = QHBoxLayout()
            
            lbl = QLabel("---")
            lbl.setFixedWidth(80)
            lbl.setStyleSheet("color: white; font-weight: bold;")
            bar_container.addWidget(lbl)
            
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #4a4a6a;
                    border-radius: 4px;
                    text-align: center;
                    background-color: #0f0f1a;
                    height: 25px;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #4a90d9, stop:1 #5aa0e9);
                    border-radius: 3px;
                }
            """)
            bar_container.addWidget(bar)
            
            prob_layout.addLayout(bar_container)
            self.prob_bars[i] = bar
            self.prob_bar_labels[i] = lbl

        layout.addWidget(prob_group)
        
        
        return container
    
    def _create_right_sidebar(self):
        """Create right sidebar with results"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(300)
        
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📊 Detection Results")
        title.setObjectName("sidebarTitle")
        layout.addWidget(title)
        
        # Raw predictions
        raw_group = QGroupBox("Raw Predictions (Live)")
        raw_layout = QVBoxLayout(raw_group)
        
        self.raw_predictions_text = QTextEdit()
        self.raw_predictions_text.setReadOnly(True)
        self.raw_predictions_text.setMaximumHeight(150)
        self.raw_predictions_text.setPlaceholderText("Predictions will appear here...")
        raw_layout.addWidget(self.raw_predictions_text)
        
        layout.addWidget(raw_group)
        
        # Prediction buffer
        buffer_group = QGroupBox("Prediction Buffer")
        buffer_layout = QVBoxLayout(buffer_group)
        
        self.buffer_text = QTextEdit()
        self.buffer_text.setReadOnly(True)
        self.buffer_text.setMaximumHeight(100)
        self.buffer_text.setPlaceholderText("Buffer collecting...")
        buffer_layout.addWidget(self.buffer_text)
        
        self.buffer_count_label = QLabel("Words in buffer: 0")
        buffer_layout.addWidget(self.buffer_count_label)
        
        # Label kata yang sedang dibaca TTS
        self.lbl_spoken_word = QLabel("Kata Dibaca: -")
        self.lbl_spoken_word.setStyleSheet("color: #888; font-size: 13px; padding: 3px;")
        buffer_layout.addWidget(self.lbl_spoken_word)
        
        layout.addWidget(buffer_group)
        
        
        # AI processed result
        ai_group = QGroupBox("🤖 AI Processed Result")
        ai_layout = QVBoxLayout(ai_group)
        
        self.ai_result_text = QTextEdit()
        self.ai_result_text.setReadOnly(True)
        self.ai_result_text.setPlaceholderText("AI will process and display the final sentence here...")
        ai_layout.addWidget(self.ai_result_text)
        
        layout.addWidget(ai_group)
        
        # History
        history_group = QGroupBox("📜 History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setPlaceholderText("Processed sentences history...")
        history_layout.addWidget(self.history_text)
        
        self.btn_clear_history = QPushButton("Clear History")
        history_layout.addWidget(self.btn_clear_history)
        
        layout.addWidget(history_group)
        
        layout.addStretch()
        
        return sidebar
    
    def _setup_connections(self):
        """Setup signal connections"""
        self.btn_start_camera.clicked.connect(self._start_camera)
        self.btn_select_video.clicked.connect(self._select_video)
        self.btn_start_video.clicked.connect(self._start_video)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop.clicked.connect(self._stop_detection)
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_clear_history.clicked.connect(self._clear_history)
    
    def _check_ai_status(self):
        """Check AI availability"""
        ai = AIProcessor()
        if ai.is_available:
            self.ai_status_label.setText("AI: ✅ Connected (Ollama)")
            self.ai_status_label.setStyleSheet("color: #4CAF50;")
            self.top_ai_status_label.setText("🤖 AI: ✅ Ollama Connected")
            self.top_ai_status_label.setStyleSheet(
                "color: #4CAF50; font-size: 12px; font-weight: bold; padding: 3px 8px;")
        else:
            self.ai_status_label.setText("AI: ⚠️ Offline (Fallback mode)")
            self.ai_status_label.setStyleSheet("color: #FF9800;")
            self.top_ai_status_label.setText("🤖 AI: ⚠️ Offline (Fallback)")
            self.top_ai_status_label.setStyleSheet(
                "color: #FF9800; font-size: 12px; font-weight: bold; padding: 3px 8px;")
    def _update_buffer_display(self, buffer, count, diam_duration):
        """Update buffer display"""
        
        # ✅ Update RAW PREDICTIONS dengan sentence real-time
        if self.worker:
            sentence = self.worker.detector.get_current_sentence()
            print(f"   Sentence: '{sentence}'")
            
            if sentence:
                self.raw_predictions_text.setPlainText(sentence)  # ✅ Gunakan setPlainText
            else:
                self.raw_predictions_text.setPlainText("(waiting for predictions...)")
        
        # ✅ Update BUFFER TEXT (untuk Prediction Buffer section)
        if buffer:
            buffer_text = " → ".join(buffer)
            self.buffer_text.setPlainText(buffer_text)  
        else:
            self.buffer_text.setPlainText("(empty)")
        
        # ✅ Update count
        self.buffer_count_label.setText(f"Words in buffer: {count}")
        
        # ✅ Show diam duration
        if diam_duration > 0:
            status_text = f"⏳ Waiting for completion... (diam: {diam_duration:.1f}s / {SILENT_DURATION}s)"
            self.status_label.setText(status_text)
        
        # ✅ FORCE GUI UPDATE
        self.raw_predictions_text.repaint()
        self.buffer_text.repaint()
        # print(f"   ✅ GUI repaint forced!\n")
            
    def _select_video(self):
        """Select video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.video_label.setText(Path(file_path).name)
            self.video_label.setStyleSheet("color: #4CAF50;")
            self.btn_start_video.setEnabled(True)
            
    def _start_camera(self):
        """Start camera detection"""
        self._stop_detection()
        
        camera_idx = self.camera_combo.currentIndex()
        
        self.worker = VideoWorker()
        self.worker.set_camera(camera_idx)
        self.worker.frame_ready.connect(self._update_frame)
        self.worker.ai_result_ready.connect(self._update_ai_result)
        self.worker.status_update.connect(self._update_status)
        self.worker.buffer_update.connect(self._update_buffer_display)  # TAMBAHAN
        self.worker.word_spoken.connect(self._update_spoken_word)  # TTS per-kata
        self.worker.final_sentence_spoken.connect(self._on_final_sentence_spoken)  # TTS final
        self.worker.start()
        
        self.current_mode = "camera"
        self._update_controls(True)

    def _start_video(self):
        """Start video detection"""
        if not self.video_path:
            return
        
        self._stop_detection()
        
        self.worker = VideoWorker()
        self.worker.set_video(self.video_path)
        self.worker.frame_ready.connect(self._update_frame)
        self.worker.ai_result_ready.connect(self._update_ai_result)
        self.worker.status_update.connect(self._update_status)
        self.worker.buffer_update.connect(self._update_buffer_display)  # TAMBAHAN
        self.worker.word_spoken.connect(self._update_spoken_word)  # TTS per-kata
        self.worker.final_sentence_spoken.connect(self._on_final_sentence_spoken)  # TTS final
        self.worker.start()
        
        self.current_mode = "video"
        self._update_controls(True)
    
    def _start_video_DUPLICATE_REMOVED(self):
        """Start video detection"""
        pass
    
    def _toggle_pause(self):
        """Toggle pause/resume"""
        if self.worker:
            if self.worker.is_paused:
                self.worker.resume()
                self.btn_pause.setText("⏸️ Pause")
            else:
                self.worker.pause()
                self.btn_pause.setText("▶️ Resume")
    
    def _stop_detection(self):
        """Stop detection"""
        if self.worker:
            self.worker.stop()
            self.worker = None
        
        self._update_controls(False)
        self.video_display.setText("Detection stopped")
    
    def _clear_all(self):
        """Clear all data"""
        if self.worker:
            self.worker.reset_detector()
        
        # ✅ Clear semua text displays
        self.raw_predictions_text.clear()
        self.raw_predictions_text.setPlaceholderText("Predictions will appear here...")
        
        self.buffer_text.clear()
        self.buffer_text.setPlaceholderText("Buffer collecting...")
        
        self.ai_result_text.clear()
        self.ai_result_text.setPlaceholderText("AI will process and display the final sentence here...")
        
        self.buffer_count_label.setText("Words in buffer: 0")
        self.confidence_bar.setValue(0)
        self.lbl_prediction.setText("Prediction: -")
        self.lbl_confidence.setText("Confidence: -")
        self.lbl_hands.setText("Hands: Not Detected")
        
        # Reset video selection
        self.video_path = None
        self.video_label.setText("No video selected")
        self.video_label.setStyleSheet("color: #888; font-style: italic;")
        self.btn_start_video.setEnabled(False)
    
    def _clear_history(self):
        """Clear history only"""
        self.history_text.clear()
    
    def _update_controls(self, is_running):
        """Update control button states"""
        self.btn_start_camera.setEnabled(not is_running)
        self.btn_start_video.setEnabled(not is_running and self.video_path is not None)
        self.btn_select_video.setEnabled(not is_running)
        self.btn_pause.setEnabled(is_running)
        self.btn_stop.setEnabled(is_running)
        self.camera_combo.setEnabled(not is_running)
        
        if is_running:
            self.btn_pause.setText("⏸️ Pause")
    
    def _update_frame(self, frame, prediction, confidence, hands_detected, probabilities):
        """Update video display with new frame"""
        # Convert frame to QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, 
                        QImage.Format.Format_RGB888 if PYQT_VERSION == 6 else QImage.Format_RGB888)
        
        # Scale to fit display
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.video_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio if PYQT_VERSION == 6 else Qt.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation if PYQT_VERSION == 6 else Qt.SmoothTransformation
        )
        
        self.video_display.setPixmap(scaled_pixmap)
        
        # Update probability bars — tampilkan 3 tertinggi
        if probabilities is not None:
            indexed = [(CLASS_LABELS[i], probabilities[i]) for i in range(len(CLASS_LABELS))]
            indexed.sort(key=lambda x: x[1], reverse=True)
            top3 = indexed[:3]
            for rank, (action, prob) in enumerate(top3):
                self.prob_bar_labels[rank].setText(action.upper())
                self.prob_bars[rank].setValue(int(prob * 100))
        
        # Update prediction info
        if prediction:
            self.lbl_prediction.setText(f"Prediction: {prediction.upper()}")
            self.lbl_prediction.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_prediction.setText("Prediction: -")
            self.lbl_prediction.setStyleSheet("color: #888;")
        
        # Update confidence
        conf_percent = int(confidence * 100)
        self.lbl_confidence.setText(f"Confidence: {conf_percent}%")
        self.confidence_bar.setValue(conf_percent)
        
        # Update system status
        if self.worker and self.worker.detector:
            detector = self.worker.detector
            
            # ✅ TAMBAHAN: Cek mode
            if not detector.use_trigger:
                # Video mode - selalu tampilkan status ACTIVE atau PROCESSING
                if detector.is_active:
                    self.system_status_label.setText("🟢 VIDEO MODE: Recording gestures...")
                    self.system_status_label.setStyleSheet("""
                        QLabel {
                            background-color: #2a4e2a;
                            color: #00FF00;
                            font-size: 16px;
                            font-weight: bold;
                            padding: 10px;
                            border-radius: 6px;
                            border: 2px solid #00FF00;
                        }
                    """)
                else:
                    self.system_status_label.setText("🔴 VIDEO MODE: Processing...")
                    self.system_status_label.setStyleSheet("""
                        QLabel {
                            background-color: #4e2a2a;
                            color: #FF6B6B;
                            font-size: 16px;
                            font-weight: bold;
                            padding: 10px;
                            border-radius: 6px;
                            border: 2px solid #FF0000;
                        }
                    """)
            else:
                # Camera mode - tampilkan status dengan trigger
                if detector.waiting_for_trigger:
                    self.system_status_label.setText("🟡 WAITING: Show peace sign (both hands) to start")
                    self.system_status_label.setStyleSheet("""
                        QLabel {
                            background-color: #2a2a4e;
                            color: #FFD700;
                            font-size: 16px;
                            font-weight: bold;
                            padding: 10px;
                            border-radius: 6px;
                            border: 2px solid #FFD700;
                        }
                    """)
                elif detector.is_countdown:
                    self.system_status_label.setText(f"🟠 COUNTDOWN: {detector.countdown_value}")
                    self.system_status_label.setStyleSheet("""
                        QLabel {
                            background-color: #FF8C00;
                            color: white;
                            font-size: 20px;
                            font-weight: bold;
                            padding: 10px;
                            border-radius: 6px;
                            border: 2px solid #FFA500;
                        }
                    """)
                elif detector.is_active:
                    self.system_status_label.setText("🟢 ACTIVE: Recording gestures...")
                    self.system_status_label.setStyleSheet("""
                        QLabel {
                            background-color: #2a4e2a;
                            color: #00FF00;
                            font-size: 16px;
                            font-weight: bold;
                            padding: 10px;
                            border-radius: 6px;
                            border: 2px solid #00FF00;
                        }
                    """)
                else:
                    self.system_status_label.setText("🔴 PROCESSING: Please wait...")
                    self.system_status_label.setStyleSheet("""
                        QLabel {
                            background-color: #4e2a2a;
                            color: #FF6B6B;
                            font-size: 16px;
                            font-weight: bold;
                            padding: 10px;
                            border-radius: 6px;
                            border: 2px solid #FF0000;
                        }
                    """)
        
        # Update hands detection
        if hands_detected:
            self.lbl_hands.setText("Hands: ✅ Detected")
            self.lbl_hands.setStyleSheet("color: #4CAF50;")
        else:
            self.lbl_hands.setText("Hands: ❌ Not Detected")
            self.lbl_hands.setStyleSheet("color: #f44336;")
        
    def _update_ai_result(self, result, raw_buffer):
        """Update AI processed result"""
        if result:
            print(f"\n✅ AI RESULT RECEIVED:")
            print(f"   Input buffer: {raw_buffer}")
            print(f"   AI output: '{result}'")
            
            # ✅ PERBAIKAN: Update AI result text saja, jangan sentuh raw_predictions
            self.ai_result_text.setPlainText(result)
            
            # Add to history
            timestamp = time.strftime("%H:%M:%S")
            history_entry = f"[{timestamp}] {result}\n"
            current_history = self.history_text.toPlainText()
            self.history_text.setText(current_history + history_entry)
            
            # Clear buffer display setelah diproses AI
            self.buffer_text.setText("(processed - waiting for new sequence)")
            self.buffer_count_label.setText("Words in buffer: 0")
            
            self.raw_predictions_text.setPlainText("(waiting for predictions...)")
            
            # Reset spoken word display
            self.lbl_spoken_word.setText("Kata Dibaca: -")
    
    def _update_spoken_word(self, word):
        """Update label kata yang sedang dibaca TTS"""
        self.lbl_spoken_word.setText(f"🔊 Dibaca: {word.upper()}")
        self.lbl_spoken_word.setStyleSheet("""
            QLabel {
                color: #00BFFF;
                font-size: 14px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #1a2a3a;
                border-radius: 4px;
                border: 1px solid #00BFFF;
            }
        """)
    def _update_status(self, status):
        """Update status label"""
        self.status_label.setText(f"Status: {status}")
        self.top_activity_label.setText(f"🔄 {status}")
    
    def _on_tts_delay_changed(self, value):
        """Handle TTS delay slider change"""
        delay = value / 10.0
        self.lbl_tts_delay.setText(f"TTS Word Delay: {delay:.1f}s")
        if self.worker:
            self.worker.tts_word_delay = delay
    
    def _toggle_tts_final(self):
        """Toggle TTS kalimat final on/off"""
        enabled = self.btn_tts_final.isChecked()
        self.btn_tts_final.setText(f"🔊 TTS Kalimat Final: {'ON' if enabled else 'OFF'}")
        if self.worker:
            self.worker.tts_final_enabled = enabled
        self.status_label.setText(f"TTS Kalimat Final {'aktif' if enabled else 'nonaktif'}")
    
    def _on_final_sentence_spoken(self, sentence):
        """Callback saat TTS kalimat final selesai diucapkan"""
        self.lbl_spoken_word.setText(f"✅ Selesai: {sentence}")
        self.lbl_spoken_word.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 13px;
                font-weight: bold;
                padding: 4px 8px;
                background-color: #1a3a1a;
                border-radius: 4px;
                border: 1px solid #4CAF50;
            }
        """)
    
    def _get_stylesheet(self):
        """Return application stylesheet"""
        return """
            QMainWindow {
                background-color: #0f0f1a;
            }
            
            QFrame#sidebar {
                background-color: #1a1a2e;
                border: 1px solid #4a4a6a;
                border-radius: 10px;
                padding: 10px;
            }
            
            QLabel#sidebarTitle {
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
                padding: 10px 0;
            }
            
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #4a4a6a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            
            QPushButton {
                background-color: #2a2a4e;
                color: #ffffff;
                border: 1px solid #4a4a6a;
                border-radius: 6px;
                padding: 10px 15px;
                font-size: 13px;
            }
            
            QPushButton:hover {
                background-color: #3a3a5e;
                border-color: #6a6a8a;
            }
            
            QPushButton:pressed {
                background-color: #1a1a3e;
            }
            
            QPushButton:disabled {
                background-color: #1a1a2e;
                color: #666;
                border-color: #333;
            }
            
            QPushButton#primaryButton {
                background-color: #4a90d9;
                border-color: #5aa0e9;
            }
            
            QPushButton#primaryButton:hover {
                background-color: #5aa0e9;
            }
            
            QComboBox {
                background-color: #2a2a4e;
                color: #ffffff;
                border: 1px solid #4a4a6a;
                border-radius: 6px;
                padding: 8px;
            }
            
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            
            QComboBox QAbstractItemView {
                background-color: #2a2a4e;
                color: #ffffff;
                selection-background-color: #4a90d9;
            }
            
            QTextEdit {
                background-color: #0f0f1a;
                color: #ffffff;
                border: 1px solid #4a4a6a;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            
            QLabel#statusLabel, QLabel#aiStatusLabel {
                color: #888;
                font-size: 12px;
                padding: 5px;
            }
            
            QFrame#videoContainer {
                background-color: #1a1a2e;
                border-radius: 10px;
                padding: 10px;
            }
            
            QFrame#infoBar {
                background-color: #0f0f1a;
                border-radius: 6px;
                padding: 10px;
                margin-top: 10px;
            }
            
            QLabel#predictionLabel, QLabel#confidenceLabel, QLabel#handsLabel {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
                padding: 5px 15px;
            }
            
            QProgressBar {
                border: 1px solid #4a4a6a;
                border-radius: 6px;
                text-align: center;
                color: #ffffff;
                background-color: #0f0f1a;
            }
            
            QProgressBar::chunk {
                background-color: #4a90d9;
                border-radius: 5px;
            }
            
            QScrollBar:vertical {
                background-color: #1a1a2e;
                width: 10px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #4a4a6a;
                border-radius: 5px;
                min-height: 20px;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """
    # ── Handler untuk slider Parameters di panel kiri ─────────────────

    def _on_conf_thresh_changed(self, value):
        """Handle Confidence Threshold slider change"""
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = value / 100.0
        self.lbl_conf_thresh.setText(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.2f}")
        if self.worker and self.worker.detector:
            self.worker.detector.threshold = CONFIDENCE_THRESHOLD

    def _on_silent_dur_changed(self, value):
        """Handle Silent Duration slider change"""
        global SILENT_DURATION
        SILENT_DURATION = value / 10.0
        self.lbl_silent_dur.setText(f"Silent Duration: {SILENT_DURATION:.1f}s")

    def _on_pred_cooldown_changed(self, value):
        """Handle Prediction Cooldown slider change"""
        global PREDICTION_COOLDOWN
        PREDICTION_COOLDOWN = value / 10.0
        self.lbl_pred_cooldown.setText(f"Prediction Cooldown: {PREDICTION_COOLDOWN:.1f}s")

    def _on_min_stable_changed(self, value):
        """Handle Min Stable Frames slider change"""
        global MIN_STABLE_FRAMES
        MIN_STABLE_FRAMES = value
        self.lbl_min_stable.setText(f"Min Stable Frames: {MIN_STABLE_FRAMES}")
        if self.worker and self.worker.detector:
            self.worker.detector.min_stable_frames = MIN_STABLE_FRAMES

    def _on_min_word_len_changed(self, value):
        """Handle Min Word Length slider change"""
        global MIN_WORD_LENGTH
        MIN_WORD_LENGTH = value
        self.lbl_min_word_len.setText(f"Min Word Length: {MIN_WORD_LENGTH}")

    def _toggle_tts(self):
        """Toggle TTS on/off"""
        if self.worker and self.worker.tts_manager:
            enabled = self.chk_tts.isChecked()
            self.worker.tts_manager.set_enabled(enabled)
            self.chk_tts.setText(f"🔊 TTS: {'ON' if enabled else 'OFF'}")
            self.status_label.setText(f"TTS {'enabled' if enabled else 'disabled'}")
    
    def _toggle_alignment(self):
        """Toggle alignment guide on/off"""
        if self.worker and self.worker.detector:
            enabled = self.chk_alignment.isChecked()
            self.worker.detector.alignment_enabled = enabled
            self.chk_alignment.setText(f"📐 Alignment: {'ON' if enabled else 'OFF'}")
            self.status_label.setText(f"Alignment guide {'enabled' if enabled else 'disabled'}")
    
    def _open_logs_folder(self):
        """Open logs folder in file explorer"""
        if self.worker and self.worker.csv_logger:
            log_folder = self.worker.csv_logger.get_log_folder()
            
            # Cross-platform open folder
            try:
                if sys.platform == 'win32':
                    os.startfile(log_folder)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open "{log_folder}"')
                else:  # Linux
                    os.system(f'xdg-open "{log_folder}"')
                
                self.status_label.setText(f"Opened: {log_folder}")
            except Exception as e:
                QMessageBox.information(self, "Logs Folder", f"Log folder: {log_folder}")
                
    def closeEvent(self, event):
        """Handle window close"""
        self._stop_detection()
        event.accept()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("SIGN LANGUAGE DETECTION GUI")
    print("=" * 60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Classes: {CLASS_LABELS}")
    print(f"AI Model: {AI_MODEL_NAME}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Silent Duration: {SILENT_DURATION}s")
    print("=" * 60)
    
    # Check dependencies
    print("\nChecking dependencies...")
    print(f"✅ PyQt Version: {PYQT_VERSION}")
    print(f"✅ MediaPipe: Available")
    print(f"{'✅' if TF_AVAILABLE else '❌'} TensorFlow: {'Available' if TF_AVAILABLE else 'Not Available'}")
    
    # Check AI availability
    ai = AIProcessor()
    print(f"{'✅' if ai.is_available else '⚠️'} Ollama AI: {'Connected' if ai.is_available else 'Not Connected (using fallback)'}")
    
    if not ai.is_available:
        print("\n⚠️  To enable AI processing, install and run Ollama:")
        print("   1. Install Ollama: https://ollama.ai")
        print(f"   2. Pull model: ollama pull {AI_MODEL_NAME}")
        print("   3. Ollama will run automatically in background")
    
    print("\n" + "=" * 60)
    print("Starting application...")
    print("=" * 60 + "\n")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = SignLanguageGUI()
    window.show()
    
    sys.exit(app.exec() if PYQT_VERSION == 6 else app.exec_())


if __name__ == "__main__":
    main()