from datetime import datetime, timezone
from app.database import embeddings_collection


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