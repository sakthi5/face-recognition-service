import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self):
        base_options = mp.tasks.BaseOptions(
            model_asset_path="models/blaze_face_short_range.tflite"
        )

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5
        )

        self.detector = mp.tasks.vision.FaceDetector.create_from_options(
            options
        )

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect(mp_image)

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

            score = detection.categories[0].score

            cv2.putText(
                frame,
                f"Face: {score:.2f}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return frame

    def close(self):
        self.detector.close()