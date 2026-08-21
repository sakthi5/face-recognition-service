import sys
import os
import time
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from insightface.app import FaceAnalysis

from app.verify_face import FaceVerifier


MODEL_PATH = "models/blaze_face_short_range.tflite"

STABLE_FRAMES_REQUIRED = 8
ORIENTATION_CHECK_INTERVAL = 5

MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 210


def calculate_face_brightness(frame, x, y, width, height):

    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(frame_width, x + width)
    y2 = min(frame_height, y + height)

    if x1 >= x2 or y1 >= y2:
        return 0

    face_region = frame[y1:y2, x1:x2]

    if face_region.size == 0:
        return 0

    gray = cv2.cvtColor(
        face_region,
        cv2.COLOR_BGR2GRAY
    )

    return float(gray.mean())


def is_face_straight(frame, landmark_app):

    faces = landmark_app.get(frame)

    if len(faces) != 1:
        return False

    face = faces[0]

    landmarks = face.landmark_2d_106

    if landmarks is None:
        return False

    left_eye = landmarks[33:43]
    right_eye = landmarks[87:97]

    left_eye_center = left_eye.mean(axis=0)
    right_eye_center = right_eye.mean(axis=0)

    eye_height_difference = abs(
        left_eye_center[1] - right_eye_center[1]
    )

    face_width = face.bbox[2] - face.bbox[0]

    if face_width <= 0:
        return False

    tilt_ratio = eye_height_difference / face_width

    return tilt_ratio < 0.08

def save_verified_image(frame, employee_id):

    # Create attendance folder
    folder_path = f"data/attendance/{employee_id}"

    os.makedirs(folder_path, exist_ok=True)

    # Current date and time
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Image path
    image_path = (
        f"{folder_path}/"
        f"{employee_id}_{timestamp}.jpg"
    )

    # Save the verified frame
    cv2.imwrite(image_path, frame)

    print()
    print("Verified image saved:")
    print(image_path)

    return image_path

def start_face_verification(employee_id):

    employee_id = employee_id.upper()

    print(f"Starting verification for: {employee_id}")

    # ==========================================
    # MEDIAPIPE FACE DETECTOR
    # ==========================================

    print("Loading MediaPipe face detector...")

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )

    detector = vision.FaceDetector.create_from_options(
        options
    )

    print("MediaPipe face detector loaded.")

    # ==========================================
    # INSIGHTFACE LANDMARK MODEL
    # ==========================================

    print("Loading InsightFace landmark model...")

    landmark_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    landmark_app.prepare(
        ctx_id=0,
        det_size=(320, 320)
    )

    print("InsightFace landmark model loaded.")

    # ==========================================
    # FACE VERIFIER
    # ==========================================

    verifier = FaceVerifier()

    # ==========================================
    # CAMERA
    # ==========================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open camera.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera started.")
    print("Press Q to quit.")

    # ==========================================
    # VARIABLES
    # ==========================================

    frame_timestamp_ms = 0

    good_frame_count = 0
    orientation_frame_count = 0
    face_is_straight = False

    # Prevent verification from running repeatedly
    verification_done = False

    verification_status = ""
    verification_message = ""

    # Used to show failure message briefly
    retry_after_time = None

    # Store successful image path
    verified_image_path = None

    # ==========================================
    # CAMERA LOOP
    # ==========================================

    while True:

        # ======================================
        # RETRY DELAY AFTER FAILED VERIFICATION
        # ======================================

        if (
            retry_after_time is not None
            and not verification_done
        ):

            if time.time() < retry_after_time:

                remaining = (
                    retry_after_time - time.time()
                )

                status = verification_status

                message = (
                    f"{verification_message} "
                    f"Try again in {remaining:.1f}s"
                )

            else:

                # Retry period finished
                retry_after_time = None

                verification_status = ""
                verification_message = ""

                good_frame_count = 0

        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        frame_timestamp_ms += 33

        result = detector.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        face_count = len(result.detections)

        status = ""
        message = ""
        brightness = None

        # ======================================
        # IF VERIFICATION ALREADY COMPLETED
        # ======================================

        if verification_done:

            status = verification_status
            message = verification_message

        # ======================================
        # NO FACE
        # ======================================

        elif face_count == 0:

            good_frame_count = 0

            status = "NO_FACE"
            message = (
                "No face detected. "
                "Please position your face inside the camera."
            )

        # ======================================
        # MULTIPLE FACES
        # ======================================

        elif face_count > 1:

            good_frame_count = 0

            status = "MULTIPLE_FACES"
            message = "Only one person should be visible."

        # ======================================
        # ONE FACE
        # ======================================

        else:

            detection = result.detections[0]
            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height

            # ----------------------------------
            # TOO FAR
            # ----------------------------------

            if width < 120 or height < 120:

                good_frame_count = 0

                status = "TOO_FAR"
                message = "Move closer to the camera."

            # ----------------------------------
            # TOO CLOSE
            # ----------------------------------

            elif width > 450 or height > 450:

                good_frame_count = 0

                status = "TOO_CLOSE"
                message = "Move slightly away from the camera."

            else:

                # ----------------------------------
                # BRIGHTNESS CHECK
                # ----------------------------------

                brightness = calculate_face_brightness(
                    frame,
                    x,
                    y,
                    width,
                    height
                )

                if brightness < MIN_BRIGHTNESS:

                    good_frame_count = 0

                    status = "TOO_DARK"
                    message = (
                        "Your surroundings are too dark. "
                        "Please move to a brighter area."
                    )

                elif brightness > MAX_BRIGHTNESS:

                    good_frame_count = 0

                    status = "TOO_BRIGHT"
                    message = (
                        "Too much light detected. "
                        "Please move away from direct light."
                    )

                else:

                    # ----------------------------------
                    # ORIENTATION CHECK
                    # ----------------------------------

                    orientation_frame_count += 1

                    if (
                        orientation_frame_count
                        >= ORIENTATION_CHECK_INTERVAL
                    ):

                        orientation_frame_count = 0

                        face_is_straight = is_face_straight(
                            frame,
                            landmark_app
                        )

                    if not face_is_straight:

                        good_frame_count = 0

                        status = "FACE_TURNED"
                        message = (
                            "Please look directly at the camera."
                        )

                    else:

                        # ----------------------------------
                        # GOOD FRAME
                        # ----------------------------------

                        good_frame_count += 1

                        if (
                            good_frame_count
                            < STABLE_FRAMES_REQUIRED
                        ):

                            status = "GOOD"

                            message = (
                                f"Hold still... "
                                f"{good_frame_count}/"
                                f"{STABLE_FRAMES_REQUIRED}"
                            )

                        else:

                            # ==================================
                            # AUTOMATIC FACE VERIFICATION
                            # ==================================


                            status = "VERIFYING"
                            message = "Verifying face..."

                            # Capture ONE good frame
                            verification_frame = frame.copy()

                            # Verify against logged-in employee
                            verification_result = verifier.verify(
                                verification_frame,
                                employee_id
                            )

                            print()
                            print("Verification result:")
                            print(verification_result)


                            # ==================================
                            # FACE MATCHED
                            # ==================================

                            if verification_result["success"]:

                                verification_done = True

                                verification_status = "MATCHED"
                                verification_message = "FACE VERIFIED"

                                # Save only the successfully verified image
                                verified_image_path = save_verified_image(
                                    verification_frame,
                                    employee_id
                                )

                                print()
                                print("FACE VERIFIED")
                                print("Employee can continue.")

                                camera.release()
                                detector.close()
                                cv2.destroyAllWindows()

                                return {
                                    "success": True,
                                    "status": "MATCHED",
                                    "message": "FACE VERIFIED",
                                    "similarity": verification_result.get("similarity"),
                                    "image_path": verified_image_path
                                }


                            # ==================================
                            # FACE NOT MATCHED
                            # ==================================

                            else:

                                verification_status = (
                                    verification_result["status"]
                                )

                                verification_message = (
                                    verification_result["message"]
                                )

                                retry_after_time = time.time() + 2

                                # Reset stable frames for next attempt
                                good_frame_count = 0

                                print()
                                print("VERIFICATION FAILED")
                                print("Trying again in 2 seconds...")

        # ======================================
        # DRAW DETECTED FACES
        # ======================================

        for detection in result.detections:

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

        # ======================================
        # DISPLAY
        # ======================================

        cv2.putText(
            frame,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if brightness is not None:

            cv2.putText(
                frame,
                f"Brightness: {brightness:.1f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        if not verification_done:

            cv2.putText(
                frame,
                (
                    f"Good frames: {good_frame_count}/"
                    f"{STABLE_FRAMES_REQUIRED}"
                ),
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        # ======================================
        # SHOW CAMERA
        # ======================================

        cv2.imshow(
            "Live Face Verification",
            frame
        )

        # ======================================
        # QUIT
        # ======================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            camera.release()
            detector.close()
            cv2.destroyAllWindows()

            return {
                "success": False,
                "status": "CANCELLED",
                "message": "Face verification was cancelled."
            }
    # ==========================================
    # CLEANUP
    # ==========================================

    camera.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m app.live_camera EMP001")

    else:
        employee_id = sys.argv[1]
        result = start_face_verification(employee_id)
        print(result)
