import numpy as np
import cv2
from config import *
import dlib
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from camera import Camera

def _initialize_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACE_LANDMARKS_PATH)

    eyenet = models.resnet18(weights=None)
    yawnnet = models.resnet18(weights=None)

    num_ftrs = eyenet.fc.in_features  # both trained on the same architecture

    eyenet.fc = torch.nn.Linear(num_ftrs, 2)  # awake or sleepy
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

def _extract_feature(frame, landmarks, indices, padding_factor=1.5):
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

def _preprocess_feature(frame, preprocess_transform, start, end):
    '''
    Convert feature region to tensor and apply preprocessing.
    '''
    feature = frame[start[1]:end[1], start[0]:end[0]]
    return preprocess_transform(feature).unsqueeze(0).to(DEVICE)

def start_detecting(camera: Camera):
    detector, predictor, eyenet, yawnnet, preprocess_transform = _initialize_models()
    
    for frame in camera.get_frames():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # get grayscale frame
        faces = detector(gray)  # detect faces in the frame

        # loop through each face detected
        for face in faces:  
            landmarks = predictor(gray, face)  # get facial landmarks
            left_eye_start, left_eye_end = _extract_feature(frame, landmarks, LEFT_EYE_INDICES)
            right_eye_start, right_eye_end = _extract_feature(frame, landmarks, RIGHT_EYE_INDICES)
            mouth_start, mouth_end = _extract_feature(frame, landmarks, MOUTH_INDICES, padding_factor=1.2)


            # Convert feature regions to preprocessed tensors
            left_eye_tensor = _preprocess_feature(frame, preprocess_transform, left_eye_start, left_eye_end)
            right_eye_tensor = _preprocess_feature(frame, preprocess_transform, right_eye_start, right_eye_end)
            mouth_tensor = _preprocess_feature(frame, preprocess_transform, mouth_start, mouth_end)

            # Predict awake/sleepy and yawn/no yawn states
            with torch.no_grad():
                left_eye_output = eyenet(left_eye_tensor)
                right_eye_output = eyenet(right_eye_tensor)
                mouth_output = yawnnet(mouth_tensor)

                # Convert raw logits to probabilities
                left_eye_prob = F.softmax(left_eye_output, dim=1)
                right_eye_prob = F.softmax(right_eye_output, dim=1)
                mouth_prob = F.softmax(mouth_output, dim=1)

                # Take max probability and corresponding class
                left_eye_conf, left_eye_class = torch.max(left_eye_prob, dim=1)
                right_eye_conf, right_eye_class = torch.max(right_eye_prob, dim=1)
                mouth_conf, mouth_class = torch.max(mouth_prob, dim=1)

                left_eye_conf, left_eye_class = left_eye_conf.item(), left_eye_class.item()
                right_eye_conf, right_eye_class = right_eye_conf.item(), right_eye_class.item()
                mouth_conf, mouth_class = mouth_conf.item(), mouth_class.item()

                # Display results of frame
                cv2.putText(frame, f"Left Eye: {EYENET_CLASSES[left_eye_class]} ({left_eye_conf:.2f})", 
                            (left_eye_start[0], left_eye_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, left_eye_start, left_eye_end, (255, 0, 0), 1)

                cv2.putText(frame, f"Right Eye: {EYENET_CLASSES[right_eye_class]} ({right_eye_conf:.2f})",
                            (right_eye_start[0], right_eye_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, right_eye_start, right_eye_end, (255, 0, 0), 1)
                cv2.putText(frame, f"Mouth: {YAWNET_CLASSES[mouth_class]} ({mouth_conf:.2f})",
                            (mouth_start[0], mouth_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, mouth_start, mouth_end, (255, 0, 0), 1)

        cv2.imshow("Camera Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
