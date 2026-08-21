import cv2
import numpy as np
from insightface.app import FaceAnalysis


# Minimum face size for employee registration
MIN_FACE_WIDTH = 100
MIN_FACE_HEIGHT = 100

# Minimum detection confidence
MIN_DETECTION_SCORE = 0.60


class FaceEmbeddingService:

    def __init__(self):
        print("Loading face recognition model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("Face recognition model loaded.")


    def get_embedding(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        # Detect all possible faces
        faces = self.app.get(image)

        print(f"DEBUG: Total detections = {len(faces)}")

        # Keep only valid faces
        valid_faces = []

        for index, face in enumerate(faces):

            x1, y1, x2, y2 = face.bbox

            width = x2 - x1
            height = y2 - y1

            score = float(face.det_score)

            print(
                f"Face {index + 1}: "
                f"Size={width:.1f}x{height:.1f}, "
                f"Score={score:.3f}"
            )

            # Ignore tiny / low-confidence false detections
            if (
                width >= MIN_FACE_WIDTH
                and height >= MIN_FACE_HEIGHT
                and score >= MIN_DETECTION_SCORE
            ):
                valid_faces.append(face)

        print(
            f"DEBUG: Valid faces = {len(valid_faces)}"
        )

        # No valid face
        if len(valid_faces) == 0:
            raise ValueError(
                f"No valid face detected in image: {image_path}"
            )

        # More than one actual valid face
        if len(valid_faces) > 1:
            raise ValueError(
                f"Multiple valid faces detected in image: {image_path}"
            )

        # Use the single valid face
        face = valid_faces[0]

        embedding = face.embedding

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding