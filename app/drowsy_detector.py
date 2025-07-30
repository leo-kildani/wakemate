import numpy as np
import cv2
from config import *
import dlib
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from warning_manager import WarningManager
from camera import Camera


class DrowsyDetector:
    def __init__(self, camera: Camera, warning_manager: WarningManager):
        self.detector, self.predictor, self.eyenet, self.yawnnet, self.preprocess_transform = self._initialize_models()
        self.camera = camera
        self.warning_manager = warning_manager


    def start_detecting(self):
        frames_eyes_closed = 0
        frames_yawning = 0
        for frame in self.camera.get_frames():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # get grayscale frame
            faces = self.detector(gray)  # detect faces in the frame


            # loop through each face detected (ideally one camera in frame)
            for face in faces:  
                landmarks = self.predictor(gray, face)  # get facial landmarks
                left_eye_start, left_eye_end = self._extract_feature(frame, landmarks, LEFT_EYE_INDICES)
                right_eye_start, right_eye_end = self._extract_feature(frame, landmarks, RIGHT_EYE_INDICES)
                mouth_start, mouth_end = self._extract_feature(frame, landmarks, MOUTH_INDICES, padding_factor=1.2)


                # Convert feature regions to preprocessed tensors
                left_eye_tensor = self._preprocess_feature(frame, self.preprocess_transform, left_eye_start, left_eye_end)
                right_eye_tensor = self._preprocess_feature(frame, self.preprocess_transform, right_eye_start, right_eye_end)
                mouth_tensor = self._preprocess_feature(frame, self.preprocess_transform, mouth_start, mouth_end)

                # Predict awake/sleepy and yawn/no yawn states
                with torch.no_grad():
                    # Batched Eye Inference 
                    eye_batch = torch.cat([left_eye_tensor, right_eye_tensor], dim=0)

                    eye_outputs = self.eyenet(eye_batch)
                    
                    eye_probs = F.softmax(eye_outputs, dim=1)
                    eye_confs, eye_classes = torch.max(eye_probs, dim=1)

                    left_eye_conf, right_eye_conf = eye_confs[0].item(), eye_confs[1].item()
                    left_eye_class, right_eye_class = eye_classes[0].item(), eye_classes[1].item()

                    mouth_output = self.yawnnet(mouth_tensor)
                    mouth_prob = F.softmax(mouth_output, dim=1)
                    mouth_conf, mouth_class = torch.max(mouth_prob, dim=1)
                    mouth_conf, mouth_class = mouth_conf.item(), mouth_class.item()
                
                    # Drowsiness Logic
                    avg_eyes_conf = (left_eye_conf + right_eye_conf) / 2
                    is_eyes_sleepy = (left_eye_class == DROWSY and 
                                        right_eye_class == DROWSY and 
                                        avg_eyes_conf >= EYES_CLOSED_MODEL_THRESHOLD)
                    if is_eyes_sleepy:
                        frames_eyes_closed += 1
                    else:
                        frames_eyes_closed = 0
                    
                    is_yawning = (mouth_class == DROWSY and mouth_conf >= YAWN_MODEL_THRESHOLD)
                    if is_yawning:
                        frames_yawning += 1
                    else:
                        frames_yawning = 0
                    
                    if frames_eyes_closed == EYES_CLOSED_FRAMES_THRESHOLD:
                        self.warning_manager.record_eyes_closed()
                        frames_eyes_closed = 0

                    if frames_yawning == YAWN_FRAMES_THRESHOLD:
                        self.warning_manager.record_yawn()
                        frames_yawning = 0

                self._draw_detection_overlay(frame, 
                                             (left_eye_start, left_eye_end, right_eye_start, right_eye_end, mouth_start, mouth_end),
                                             (left_eye_class, right_eye_class, mouth_class),
                                             (left_eye_conf, right_eye_conf, mouth_conf))

            cv2.imshow("Camera Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def _initialize_models(self):
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(FACE_LANDMARKS_PATH)

        eyenet = models.resnet18(weights=None)
        yawnnet = models.resnet18(weights=None)

        num_ftrs = eyenet.fc.in_features  # both trained on the same architecture

        eyenet.fc = torch.nn.Linear(num_ftrs, 2)  # open or closed eyes
        yawnnet.fc = torch.nn.Linear(num_ftrs, 2)  # yawn or no yawn

        eyenet.load_state_dict(torch.load(EYENET_PATH, map_location=DEVICE))
        yawnnet.load_state_dict(torch.load(YAWNET_PATH, map_location=DEVICE))
        eyenet.to(DEVICE)
        yawnnet.to(DEVICE)

        eyenet.eval()
        yawnnet.eval()

        preprocess_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return detector, predictor, eyenet, yawnnet, preprocess_transform

    def _extract_feature(self, frame, landmarks, indices, padding_factor=1.5):
        '''
        Extracts a feature region from the frame based on facial landmarks.
        '''
        
        assert padding_factor > 0, "Padding factor must be greater than 0"

        # Extract the region of interest (ROI) for the given facial feature
        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in indices], dtype=np.int32)
        (x, y, w, h) = cv2.boundingRect(points)
        padding = int(padding_factor * w)  - w
        x_start = max(0, x - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_start = max(0, y - padding)
        y_end = min(frame.shape[0], y + h + padding)
        return (x_start, y_start), (x_end, y_end)

    def _preprocess_feature(self, frame, preprocess_transform, start, end):
        '''
        Convert feature region to tensor and apply preprocessing.
        '''
        feature = frame[start[1]:end[1], start[0]:end[0]]
        return preprocess_transform(feature).unsqueeze(0).to(DEVICE)

    def _draw_detection_overlay(self, frame, features_landmarks, features_class, features_conf):
        '''
        Draws rectangles and labels around detected features.
        feature_lanmarks: List of tuples containing start and end coordinates for each feature. (left_eye_start, left_eye_end, right_eye_start, right_eye_end, mouth_start, mouth_end)
        feature_class: List of classes for each feature. (left_eye_class, right_eye_class, mouth_class)
        feature_conf: List of confidence scores for each feature. (left_eye_conf, right_eye
        '''
        # Draw rectangles around detected features
        left_eye_color = (0, 255, 0)
        right_eye_color = (0, 255, 0)
        mouth_color = (0, 0, 255) if features_class[2] == DROWSY else (0, 255, 0)
        if features_class[0] == DROWSY and features_class[1] == DROWSY:
            left_eye_color = (0, 0, 255)
            right_eye_color = (0, 0, 255)
        elif features_class[0] == DROWSY:
            left_eye_color = (0, 165, 255)
        elif features_class[1] == DROWSY:
            right_eye_color = (0, 165, 255)


        cv2.rectangle(frame, features_landmarks[0], features_landmarks[1], left_eye_color, 2)
        cv2.rectangle(frame, features_landmarks[2], features_landmarks[3], right_eye_color, 2)
        cv2.rectangle(frame, features_landmarks[4], features_landmarks[5], mouth_color, 2)

        cv2.putText(frame, f"L Eye: {features_class[0]} ({features_conf[0]:.2f})", (features_landmarks[1][0], features_landmarks[0][1]), cv2.FONT_HERSHEY_PLAIN, 1, left_eye_color, 2)
        cv2.putText(frame, f"R Eye: {features_class[1]} ({features_conf[1]:.2f})", (features_landmarks[2][0] - 60, features_landmarks[2][1]), cv2.FONT_HERSHEY_PLAIN, 1, right_eye_color, 2)

        cv2.putText(frame, f"Mouth: {features_class[2]} ({features_conf[2]:.2f})", (features_landmarks[4][0], features_landmarks[4][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, mouth_color, 2)
        
        stats_x = 10
        stats_font_scale = 1.2
        stats_color = (255, 255, 255)
        stats_thickness = 2
        line_height = 30

        cv2.putText(frame, f"Warnings: {self.warning_manager.warning_count}",
                    (stats_x, line_height), 
                    cv2.FONT_HERSHEY_PLAIN, stats_font_scale, stats_color, stats_thickness)

        cv2.putText(frame, f"Yawns: {self.warning_manager.yawn_count}",
                    (stats_x, line_height * 2), 
                    cv2.FONT_HERSHEY_PLAIN, stats_font_scale, stats_color, stats_thickness)

        cv2.putText(frame, f"Eyes Closed: {self.warning_manager.eyes_closed_count}",
                    (stats_x, line_height * 3), 
                    cv2.FONT_HERSHEY_PLAIN, stats_font_scale, stats_color, stats_thickness)