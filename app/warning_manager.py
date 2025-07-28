from config import *
from agent import WakeMateAgent

class WarningManager:
    def __init__(self, agent: WakeMateAgent):
        self.yawn_count = 0
        self.eyes_closed_count = 0
        self.warning_count = 0
        self.agent = agent
        self.warning_in_progress = False
    
    def record_yawn(self):
        if (self.yawn_count == YAWN_FRAMES_THRESHOLD):
            self.yawn_count = 1
            return True
        self.yawn_count += 1
        return False
    
    def record_eyes_closed(self):
        if (self.eyes_closed_count == EYES_CLOSED_FRAMES_THRESHOLD):
            self.eyes_closed_count = 1
            return True
        self.eyes_closed_count += 1
    
    def record_warning(self):
        self.warning_count += 1
    
    def reset_count(self):
        self.yawn_count = 0
        self.eyes_closed_count = 0
        self.warning_count = 0
    
    def _issue_warning(self, prompt):
        """Helper method to generate and speak a warning."""
        print("THREAD: Generating warning message...")
        warning = self.agent.generate_warning(prompt)

        print("THREAD: Playing warning message...")
        self.agent.text_to_speech(warning)
        print("THREAD: Warning message played.")

    def trigger_warning(self):
        """Checks if a warning should be triggered and handles the process."""
        # 1. Guard clause: Exit if a warning is already in progress.
        if self.warning_in_progress:
            return

        # 2. Map warning counts to their prompts for scalability.
        prompt_map = {
            INIT_WARNING_COUNT: FIRST_WARNING_PROMPT,
            INIT_WARNING_COUNT * 2: SECOND_WARNING_PROMPT,
            INIT_WARNING_COUNT * 3: THIRD_WARNING_PROMPT,
        }

        # 3. Get the prompt for the current count, if any.
        gemini_prompt = prompt_map.get(self.warning_count)

        # 4. If there's no prompt for this count, do nothing.
        if not gemini_prompt:
            return

        # 5. Issue the warning and manage state.
        try:
            self.warning_in_progress = True
            print(f"THREAD: Starting Warning for count {self.warning_count}")
            self._issue_warning(gemini_prompt)

            # Reset after the final warning
            if self.warning_count == INIT_WARNING_COUNT * 3:
                self.reset_count()
        finally:
            # Ensure the flag is always reset, even if an error occurs.
            self.warning_in_progress = False