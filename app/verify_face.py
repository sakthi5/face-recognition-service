import numpy as np
from insightface.app import FaceAnalysis
from app.embedding_store import get_stored_embedding, SIMILARITY_THRESHOLD



class FaceVerifier:

    def __init__(self):
        print("Loading face recognition model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(320, 320)
        )

        print("Face recognition model loaded.")

    def load_employee_embedding(self, employee_id):
        """
        Load the stored embedding for the employee from MongoDB.
        """

        embedding = get_stored_embedding(employee_id)

        if embedding is None:
            raise FileNotFoundError(
                f"No embedding found for employee {employee_id}"
            )

        return np.array(embedding)
        

    def verify(self, current_embedding, employee_id):
        """
        Compare an already-extracted face embedding against the
        logged-in employee's stored embedding.

        Detection happens once, by the caller (the quality check
        already needs it) - this only does the comparison, so a
        /verify call doesn't pay for running InsightFace twice.
        """

        try:

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