import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI environment variable is not set.")

client = MongoClient(MONGODB_URI)
db = client["face_recognition"]
embeddings_collection = db["employee_embeddings"]