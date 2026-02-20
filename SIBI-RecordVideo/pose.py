"""
Overlay skeleton pose menggunakan MediaPipe dengan deteksi tangan.
"""
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PoseDrawer:
    def __init__(self):
        # Pose detection
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Hand detection untuk deteksi jari detail
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Warna skeleton
        self.pose_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.hand_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)

    def draw(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process pose
        pose_results = self.pose.process(image_rgb)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                pose_results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_spec,
                connection_drawing_spec=self.pose_spec
            )
        
        # Process hands (detailed finger detection)
        hand_results = self.hands.process(image_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return image
    
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()