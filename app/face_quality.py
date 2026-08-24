import cv2
import numpy as np


# Adjust these values later based on testing
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 220

MIN_FACE_WIDTH_RATIO = 0.15
MAX_FACE_WIDTH_RATIO = 0.75

MIN_DETECTION_SCORE = 0.60
MIN_SHARPNESS = 80.0


class FaceQualityChecker:

    def check(self, frame, faces):
        """
        Check whether the current frame is suitable
        for face verification or registration.
        """

        # -----------------------------
        # 1. NO FACE
        # -----------------------------
        if len(faces) == 0:
            return {
                "success": False,
                "status": "NO_FACE",
                "message": (
                    "No face detected. "
                    "Please position your face inside the camera."
                )
            }

        # -----------------------------
        # 2. MULTIPLE FACES
        # -----------------------------
        if len(faces) > 1:
            return {
                "success": False,
                "status": "MULTIPLE_FACES",
                "message": (
                    "Only one person should be visible."
                )
            }

        face = faces[0]

        # -----------------------------
        # 3. LOW DETECTION CONFIDENCE
        # -----------------------------
        if float(face.det_score) < MIN_DETECTION_SCORE:
            return {
                "success": False,
                "status": "LOW_CONFIDENCE",
                "message": (
                    "Could not clearly detect your face. "
                    "Please try again with better lighting."
                ),
                "det_score": float(face.det_score)
            }

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # -----------------------------
        # 4. TOO BLURRY
        # -----------------------------
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness < MIN_SHARPNESS:
            return {
                "success": False,
                "status": "TOO_BLURRY",
                "message": (
                    "Image is too blurry. Please hold the camera steady."
                ),
                "sharpness": sharpness
            }

        # -----------------------------
        # 5. TOO DARK / TOO BRIGHT
        # -----------------------------
        brightness = float(np.mean(gray))

        if brightness < MIN_BRIGHTNESS:
            return {
                "success": False,
                "status": "TOO_DARK",
                "message": (
                    "Your surroundings are too dark. "
                    "Please move to a brighter area."
                ),
                "brightness": brightness
            }

        if brightness > MAX_BRIGHTNESS:
            return {
                "success": False,
                "status": "TOO_BRIGHT",
                "message": (
                    "Too much light detected. "
                    "Please move away from direct light."
                ),
                "brightness": brightness
            }

        # -----------------------------
        # Face bounding box
        # -----------------------------
        x1, y1, x2, y2 = face.bbox.astype(int)

        frame_height, frame_width = frame.shape[:2]

        face_width = x2 - x1

        face_width_ratio = (
            face_width / frame_width
        )

        # -----------------------------
        # 6. FACE TOO FAR
        # -----------------------------
        if face_width_ratio < MIN_FACE_WIDTH_RATIO:
            return {
                "success": False,
                "status": "TOO_FAR",
                "message": (
                    "Move closer to the camera."
                ),
                "face_ratio": face_width_ratio
            }

        # -----------------------------
        # 7. FACE TOO CLOSE
        # -----------------------------
        if face_width_ratio > MAX_FACE_WIDTH_RATIO:
            return {
                "success": False,
                "status": "TOO_CLOSE",
                "message": (
                    "Move slightly away from the camera."
                ),
                "face_ratio": face_width_ratio
            }

        # -----------------------------
        # 8. FACE TURNED
        # -----------------------------
        landmarks = face.kps

        if landmarks is not None:

            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]

            eye_center_x = (
                left_eye[0] + right_eye[0]
            ) / 2

            eye_distance = abs(
                right_eye[0] - left_eye[0]
            )

            nose_offset = abs(
                nose[0] - eye_center_x
            )

            if eye_distance > 0:

                turn_ratio = (
                    nose_offset / eye_distance
                )

                if turn_ratio > 0.25:
                    return {
                        "success": False,
                        "status": "FACE_TURNED",
                        "message": (
                            "Please look directly at the camera."
                        ),
                        "turn_ratio": turn_ratio
                    }

        # -----------------------------
        # ALL CONDITIONS PASSED
        # -----------------------------
        return {
            "success": True,
            "status": "GOOD",
            "message": (
                "Face quality is good."
            ),
            "brightness": brightness,
            "face_ratio": face_width_ratio
        }