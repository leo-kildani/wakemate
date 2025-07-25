import cv2


class Camera:
    """A camera class for video capture operations."""
    
    def __init__(self, camera_index=0):
        """
        Initialize the Camera object.
        
        Args:
            camera_index (int): Index of the camera device (default: 0)
        """
        self.camera_index = camera_index
        self.camera = None
        self.is_initialized = False
    
    def initialize(self):
        """
        Initialize the camera for video capture.
        
        Raises:
            IOError: If the camera could not be opened
        """
        self.camera = cv2.VideoCapture(self.camera_index)
        
        # Check camera opened successfully
        if not self.camera.isOpened():
            raise IOError("Could not open video device")
        
        self.is_initialized = True
    
    def release(self):
        """Release the camera and cleanup resources."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.is_initialized = False
    
    def get_frames(self):
        """
        Generator that yields frames from the camera.
        
        Yields:
            numpy.ndarray: Camera frame
            
        Raises:
            ValueError: If camera is not initialized
        """
        if not self.is_initialized or self.camera is None:
            raise ValueError("Camera not initialized. Call initialize() first.")
        
        try:
            while True:
                # Read a frame from the camera; ret is a boolean indicating if the frame was read successfully
                ret, frame = self.camera.read()
                # If frame is not read successfully, break the loop
                if not ret:
                    break
                yield frame
        # Ensure that the camera is released properly in any case
        finally:
            self.release()
    
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()