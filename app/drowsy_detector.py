import numpy as np
import cv2
from config import *
import dlib
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from warning_manager import WarningManager
from camera import Camera
import threading


class DrowsyDetector:
    def __init__(self, camera: Camera, warning_manager: WarningManager):
        self.detector, self.predictor, self.eyenet, self.yawnnet, self.preprocess_transform = self._initialize_models()
        self.camera = camera
        self.warning_manager = warning_manager


    def start_detecting(self):
        for frame in self.camera.get_frames():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # get grayscale frame
            faces = self.detector(gray)  # detect faces in the frame

            # loop through each face detected
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
                                        avg_eyes_conf >= EYES_CLOSED_MODEL_THRESHOLD and 
                                        self.warning_manager.record_eyes_closed())
                    
                    is_yawning = (mouth_class == DROWSY and 
                                    mouth_conf >= YAWN_MODEL_THRESHOLD and 
                                    self.warning_manager.record_yawn())

                    if is_eyes_sleepy or is_yawning:
                        self.warning_manager.record_warning()
                        trigger_warning_thread = threading.Thread(target=self.warning_manager.trigger_warning)
                        trigger_warning_thread.start()
                    
                    # Draw info around detected features
                    cv2.rectangle(frame, left_eye_start, left_eye_end, (0, 255, 0), 2)
                    cv2.rectangle(frame, right_eye_start, right_eye_end, (0, 255, 0), 2)
                    cv2.rectangle(frame, mouth_start, mouth_end, (0, 255, 0), 2)

                    cv2.putText(frame, f"Left Eye: {EYENET_CLASSES[left_eye_class]} ({left_eye_conf:.2f})",
                                (left_eye_start[0] - 20, left_eye_start[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Right Eye: {EYENET_CLASSES[right_eye_class]} ({right_eye_conf:.2f})",
                                (right_eye_start[0] - 20, right_eye_start[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Mouth: {YAWNET_CLASSES[mouth_class]} ({mouth_conf:.2f})",
                                (mouth_start[0] - 20, mouth_start[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.putText(frame, f"Warnings: {self.warning_manager.warning_count}",
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Yawns: {self.warning_manager.yawn_count}",
                                (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Eyes Closed: {self.warning_manager.eyes_closed_count}",
                                (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


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
