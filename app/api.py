from fastapi import FastAPI, UploadFile, File, Form
from contextlib import asynccontextmanager
import cv2
import numpy as np

from app.verify_face import FaceVerifier
from app.face_quality import FaceQualityChecker
from fastapi.middleware.cors import CORSMiddleware
from app.embedding_store import save_or_update_embedding



face_verifier = None
quality_checker = FaceQualityChecker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_verifier

    print("Starting Face Recognition API...")
    print("Loading face recognition model once...")

    face_verifier = FaceVerifier()

    print("Face Recognition API is ready.")

    yield

    print("Shutting down Face Recognition API...")


app = FastAPI(
    title="Face Recognition API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Face Recognition API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "OK",
        "model_loaded": face_verifier is not None
    }


@app.post("/verify")
async def verify_face(
    employee_id: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        image_bytes = await image.read()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return {
                "success": False,
                "status": "INVALID_IMAGE",
                "message": "Could not read the uploaded image."
            }

        # Detect faces using the already loaded InsightFace model
        faces = face_verifier.app.get(frame)


        # Run face quality checks first
        quality_result = quality_checker.check(
            frame,
            faces
        )


        # Stop if face quality is not good
        if not quality_result.get("success"):

            return quality_result


        # Quality is good, now verify identity
        result = face_verifier.verify(
            frame,
            employee_id
        )

        return result

    except Exception as error:
        print("API verification error:", error)

        return {
            "success": False,
            "status": "ERROR",
            "message": "Face verification failed."
        }

@app.post("/embed")
async def embed_face(
    employee_id: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        image_bytes = await image.read()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return {
                "success": False,
                "status": "INVALID_IMAGE",
                "message": "Could not read the uploaded image."
            }

        faces = face_verifier.app.get(frame)

        quality_result = quality_checker.check(frame, faces)

        if not quality_result.get("success"):
            return quality_result

        embedding = faces[0].embedding
        embedding = embedding / np.linalg.norm(embedding)

        status = save_or_update_embedding(employee_id, embedding)

        return {
            "success": True,
            "status": status,
            "message": (
                "Employee face registered successfully."
                if status == "REGISTERED"
                else "Employee face updated successfully."
            )
        }

    except Exception as error:
        print("API embedding error:", error)

        return {
            "success": False,
            "status": "ERROR",
            "message": "Face registration failed."
        }