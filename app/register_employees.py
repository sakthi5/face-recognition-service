import os
import pickle

from app.embeddings import FaceEmbeddingService
from app.employees import EMPLOYEES


def main():

    service = FaceEmbeddingService()

    os.makedirs("data/embeddings", exist_ok=True)

    for employee_id, employee in EMPLOYEES.items():

        print()
        print(f"Processing {employee_id} - {employee['name']}")

        try:
            embedding = service.get_embedding(
                employee["image"]
            )

            output_path = (
                f"data/embeddings/{employee_id}.pkl"
            )

            with open(output_path, "wb") as file:
                pickle.dump(embedding, file)

            print("✅ Face embedding created")
            print(f"Saved to: {output_path}")
            print(f"Embedding size: {len(embedding)}")

        except Exception as error:
            print(f"❌ Failed: {error}")


if __name__ == "__main__":
    main()