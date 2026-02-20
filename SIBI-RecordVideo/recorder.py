"""
Modul perekaman video.
"""
import cv2
from threading import Thread, Lock
import time

class VideoRecorder:
    """
    Thread-safe video recorder.
    """
    def __init__(self, filepath, fourcc, fps, frame_size):
        self.filepath = filepath
        self.fourcc = fourcc
        self.fps = fps
        self.frame_size = frame_size
        self.out = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
        self.recording = False
        self.lock = Lock()
        self.timestamps = []  # tambahkan
    def start(self):
        self.recording = True

    def write(self, frame):
        with self.lock:
            if self.recording:
                self.out.write(frame)
                self.timestamps.append(time.time())  # catat waktu

    def stop(self):
        with self.lock:
            self.recording = False
        self.out.release()

    def is_recording(self):
        with self.lock:
            return self.recording