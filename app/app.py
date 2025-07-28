from camera import Camera
from drowsy_detector import *
from warning_manager import WarningManager
from agent import WakeMateAgent
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    cam = Camera()
    cam.initialize()
    agent = WakeMateAgent(os.getenv("GEMINI_API_KEY"), os.getenv("ELEVENLABS_API_KEY"))
    warning_manager = WarningManager(agent)
    drowsy_detector = DrowsyDetector(cam, warning_manager)
    drowsy_detector.start_detecting()
