import mediapipe as mp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class FaceProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract_face_embedding(self, image):
        # Ensure the image is in the correct format (8-bit RGB)
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if image.ndim == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                if image.shape[2] > 3:  # More than 3 channels
                    image = image[:, :, :3]
                elif image.shape[2] == 1:  # Single channel
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Ensure the image is in HWC format
        if image.shape[0] == 3 and image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))

        # Process the image
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None

        # Extract landmarks
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Create a simple embedding using key facial landmarks
        key_points = [33, 133, 362, 263, 61, 291, 199]  # Example key points
        embedding = np.array([[pt.x, pt.y, pt.z] for pt in face_landmarks if pt.landmark_index in key_points]).flatten()

        return embedding

    def identity_consistency_loss(self, generated_image, ref_embedding):
        # Convert tensor to numpy array and ensure it's in the correct format
        if torch.is_tensor(generated_image):
            generated_image = generated_image.cpu().numpy()
        
        # Ensure the image is in the correct shape (H, W, C)
        if generated_image.shape[0] == 3:  # If in (C, H, W) format
            generated_image = np.transpose(generated_image, (1, 2, 0))
        
        gen_embedding = self.extract_face_embedding(generated_image)
        if gen_embedding is not None and ref_embedding is not None:
            return np.linalg.norm(gen_embedding - ref_embedding)
        return 0