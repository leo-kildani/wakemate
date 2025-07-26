import torch

RIGHT_EYE_INDICES = list(range(36, 42))
LEFT_EYE_INDICES = list(range(42, 48))   
MOUTH_INDICES = list(range(48, 68))

FACE_LANDMARKS_PATH = "./models/shape_predictor_68_face_landmarks.dat"
EYENET_PATH = "./models/eyenet_model.pth"
YAWNET_PATH = "./models/yawnnet_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EYENET_CLASSES = ["Awake", "Sleepy"]

YAWNET_CLASSES = ["Not Yawning", "Yawning"]