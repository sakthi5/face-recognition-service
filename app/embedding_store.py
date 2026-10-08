from datetime import datetime, timezone
import numpy as np
from app.database import embeddings_collection


# How similar two faces need to be to be considered "the same person".
# Shared with FaceVerifier.verify() so enrollment and verification agree
# on what counts as a match.
SIMILARITY_THRESHOLD = 0.50


def get_stored_embedding(employee_id):
    """
    Fetch the stored embedding for an employee.
    Returns a list, or None if the employee has never registered.
    """
    document = embeddings_collection.find_one(
        {"employee_id": employee_id}
    )

    if document is None:
        return None

    return document["embedding"]


def save_or_update_embedding(employee_id, embedding):
    """
    Create a new embedding, or replace the existing one
    for this employee_id. Returns "REGISTERED" or "UPDATED".
    """
    existing = embeddings_collection.find_one(
        {"employee_id": employee_id}
    )

    now = datetime.now(timezone.utc)

    if existing is None:
        embeddings_collection.insert_one({
            "employee_id": employee_id,
            "embedding": embedding.tolist(),
            "created_at": now,
            "updated_at": now
        })
        return "REGISTERED"

    embeddings_collection.update_one(
        {"employee_id": employee_id},
        {"$set": {
            "embedding": embedding.tolist(),
            "updated_at": now
        }}
    )
    return "UPDATED"


def find_duplicate_face(embedding, exclude_employee_id=None):
    """
    Check whether this face already belongs to a DIFFERENT employee_id.

    Compares against every stored embedding except exclude_employee_id's
    own (so re-enrolling the same person under their own id is never
    flagged as a duplicate of themselves).

    Returns a (employee_id, similarity) tuple:
      - employee_id is the closest match's id if it's over
        SIMILARITY_THRESHOLD (a duplicate), otherwise None.
      - similarity is always the score of the single closest match
        found, even when it's None, so callers can log/inspect
        near-misses instead of only knowing "no duplicate found".
    """
    normalized_new = embedding / np.linalg.norm(embedding)

    query = {}
    if exclude_employee_id is not None:
        query["employee_id"] = {"$ne": exclude_employee_id}

    best_employee_id = None
    best_similarity = -1.0

    for document in embeddings_collection.find(query):
        stored = np.array(document["embedding"])
        stored = stored / np.linalg.norm(stored)

        similarity = float(np.dot(normalized_new, stored))

        if similarity > best_similarity:
            best_similarity = similarity
            best_employee_id = document["employee_id"]

    if best_similarity >= SIMILARITY_THRESHOLD:
        return best_employee_id, best_similarity

    return None, best_similarity


def delete_embedding(employee_id):
    """
    Delete the stored embedding for an employee.
    Returns "DELETED" if a document was removed,
    or "NOT_FOUND" if no document existed for this employee_id.
    """
    result = embeddings_collection.delete_one(
        {"employee_id": employee_id}
    )

    if result.deleted_count == 0:
        return "NOT_FOUND"

    return "DELETED"