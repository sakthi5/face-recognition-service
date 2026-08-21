import pickle
import numpy as np
from insightface.app import FaceAnalysis


SIMILARITY_THRESHOLD = 0.50



class FaceVerifier:

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

    def load_employee_embedding(self, employee_id):
        """
        Load the stored embedding for the employee.
        """

        path = f"data/embeddings/{employee_id}.pkl"

        with open(path, "rb") as file:
            embedding = pickle.load(file)

        return embedding

    def verify(self, frame, employee_id):
        """
        Compare the detected face in the frame with
        the logged-in employee's stored embedding.
        """

        try:

            # Get face from the captured frame
            faces = self.app.get(frame)

            if len(faces) == 0:
                return {
                    "success": False,
                    "status": "NO_FACE",
                    "message": "No face found for verification."
                }

            if len(faces) > 1:
                return {
                    "success": False,
                    "status": "MULTIPLE_FACES",
                    "message": "Only one person should be visible."
                }

            # Current face embedding
            current_embedding = faces[0].embedding

            # Stored employee embedding
            stored_embedding = self.load_employee_embedding(
                employee_id
            )

            # Normalize embeddings
            current_embedding = (
                current_embedding /
                np.linalg.norm(current_embedding)
            )

            stored_embedding = (
                stored_embedding /
                np.linalg.norm(stored_embedding)
            )

            # Cosine similarity
            similarity = float(
                np.dot(
                    current_embedding,
                    stored_embedding
                )
            )

            print(
                f"Similarity score: {similarity:.4f}"
            )

            # Face matched
            if similarity >= SIMILARITY_THRESHOLD:

                return {
                    "success": True,
                    "status": "MATCHED",
                    "message": "FACE VERIFIED",
                    "similarity": similarity
                }

            # Wrong person
            return {
                "success": False,
                "status": "WRONG_PERSON",
                "message": "WRONG PERSON",
                "similarity": similarity
            }

        except FileNotFoundError:

            return {
                "success": False,
                "status": "EMPLOYEE_NOT_FOUND",
                "message": (
                    f"Face data not found for {employee_id}."
                )
            }

        except Exception as error:

            print("Verification error:", error)

            return {
                "success": False,
                "status": "ERROR",
                "message": "Face verification failed."
            }