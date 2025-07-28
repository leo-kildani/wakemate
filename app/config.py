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

DROWSY = 1
NOT_DROWSY = 0


FIRST_WARNING_PROMPT = "Generate a short, friendly, and engaging statement (1-2 sentences) to gently alert a driver how seems a bit drowsy. This is the driver's first warning."

SECOND_WARNING_PROMPT = "Generate an earnest satement (2-3 sentences) cautioning a drowsy driver about the dangers of drowsy driving. The driver has already received a verbal warning in a friendly tone, so this statement should be more serious and emphasize the importance of safety. Offer to play some upbeat music or suggest taking a break at the nearest rest area."

THIRD_WARNING_PROMPT = "Generate a strong warning (2-3 sentences) for a driver who is very drowsy after multiple alerts. This statement should urge the driver to take immediate action, such as pulling over as soon as it's safe to do so."

DEFAULT_WARNING = "Drowsiness detected. Please take a break."

EYES_CLOSED_MODEL_THRESHOLD = 0.8
YAWN_MODEL_THRESHOLD = 0.8

EYES_CLOSED_FRAMES_THRESHOLD = 30
YAWN_FRAMES_THRESHOLD = 30 

INIT_WARNING_COUNT = 4