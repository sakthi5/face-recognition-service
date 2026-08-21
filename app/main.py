import cv2
from app.face_detection import FaceDetector


def main():
    detector = FaceDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Could not open camera")
        return

    print("Camera started.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("❌ Could not read camera frame")
            break

        frame = detector.detect(frame)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()