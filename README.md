# Face Recognition API

Python FastAPI backend for employee face verification, backed by MongoDB and InsightFace.

## Features

* Employee face enrollment (embedding generation)
* Employee face verification
* Employee deletion (admin-only)
* Duplicate-face detection (one face can only be enrolled under one employee ID)
* No face detection
* Multiple faces detection
* Low detection confidence check
* Too blurry detection
* Too dark detection
* Too bright detection
* Face too far detection
* Face too close detection
* Face turned detection
* Wrong person detection
* FastAPI REST API for Next.js integration

## Project Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in real values:

```powershell
copy .env.example .env
```

| Variable | Required | Purpose |
|---|---|---|
| `MONGODB_URI` | Yes | MongoDB connection string. Face embeddings are stored in the `face_recognition.employee_embeddings` collection. |
| `ADMIN_API_KEY` | Yes (for delete) | Shared-secret key required in the `X-API-Key` header to call `DELETE /employees/{employee_id}`. |

`.env` is gitignored — never commit real secrets.

## Run the API

From the project root:

```powershell
uvicorn app.api:app --reload
```

The API will run at:

`http://127.0.0.1:8000`

## API Documentation

Open:

`http://127.0.0.1:8000/docs`

## Endpoints

### GET /

Checks whether the API is running.

Response:

```json
{
  "message": "Face Recognition API is running"
}
```

### GET /health

Checks API and model status.

Response:

```json
{
  "status": "OK",
  "model_loaded": true
}
```

### POST /embed

Enrolls a new employee's face, or updates an existing one. Runs the same face quality checks as `/verify` before storing anything.

Request type:

`multipart/form-data`

Fields:

* `employee_id`: Employee ID, for example `EMP001`
* `image`: Face photo to enroll

Successful response:

```json
{
  "success": true,
  "status": "REGISTERED",
  "message": "Employee face registered successfully."
}
```

`status` is `"REGISTERED"` on first enrollment, or `"UPDATED"` if this `employee_id` already had a stored face (the old embedding is replaced).

Before saving, the face is also checked against every other employee's stored face. If it already matches a **different** `employee_id`, enrollment is rejected with `DUPLICATE_FACE` — one face can only be registered under one employee ID.

Possible responses: `REGISTERED`, `UPDATED`, `DUPLICATE_FACE`, `NO_FACE`, `MULTIPLE_FACES`, `LOW_CONFIDENCE`, `TOO_BLURRY`, `TOO_DARK`, `TOO_BRIGHT`, `TOO_FAR`, `TOO_CLOSE`, `FACE_TURNED`, `INVALID_IMAGE`, `ERROR`.

No authentication is currently required on this endpoint.

### POST /verify

Verifies an employee's live face against their stored embedding.

Request type:

`multipart/form-data`

Fields:

* `employee_id`: Employee ID, for example `EMP001`
* `image`: Captured face image

Successful response:

```json
{
  "success": true,
  "status": "MATCHED",
  "message": "FACE VERIFIED",
  "similarity": 0.7
}
```

Possible responses:

* `MATCHED`
* `WRONG_PERSON`
* `NO_FACE`
* `MULTIPLE_FACES`
* `LOW_CONFIDENCE`
* `TOO_BLURRY`
* `TOO_DARK`
* `TOO_BRIGHT`
* `TOO_FAR`
* `TOO_CLOSE`
* `FACE_TURNED`
* `INVALID_IMAGE`
* `EMPLOYEE_NOT_FOUND`
* `ERROR`

All responses (success or failure) come back as HTTP 200 — check the `status`/`success` fields in the body, not the HTTP status code.

No authentication is currently required on this endpoint.

### DELETE /employees/{employee_id}

Deletes an employee's stored face embedding. **Admin-only** — requires the `X-API-Key` header to match `ADMIN_API_KEY`.

Request:

```text
DELETE /employees/EMP001
X-API-Key: <ADMIN_API_KEY>
```

Successful response:

```json
{
  "success": true,
  "status": "DELETED",
  "message": "Employee face data deleted successfully."
}
```

If no face was on file for that `employee_id`:

```json
{
  "success": false,
  "status": "NOT_FOUND",
  "message": "No face data found for this employee."
}
```

Missing or incorrect `X-API-Key` returns a real HTTP 401 (this is the one endpoint that deviates from the "always 200" convention above, since it's an auth failure, not a domain result):

```json
{
  "detail": "Invalid or missing API key."
}
```

## Next.js Integration Flow

```text
Next.js Application
        ↓
Open browser camera
        ↓
Capture live face image
        ↓
Send employee_id + image
        ↓
POST /verify
        ↓
Face Recognition API
        ↓
Face Quality Check
        ↓
Face Verification
        ↓
Return verification result
```

## Example Requests

### Verify a face

```javascript
const formData = new FormData();

formData.append("employee_id", "EMP001");
formData.append("image", imageFile);

const response = await fetch(
  "http://127.0.0.1:8000/verify",
  {
    method: "POST",
    body: formData
  }
);

const result = await response.json();

console.log(result);
```

### Enroll/update a face

```javascript
const formData = new FormData();

formData.append("employee_id", "EMP001");
formData.append("image", imageFile);

const response = await fetch(
  "http://127.0.0.1:8000/embed",
  {
    method: "POST",
    body: formData
  }
);

const result = await response.json();

console.log(result);
```

### Delete an employee (admin-only)

```javascript
const response = await fetch(
  "http://127.0.0.1:8000/employees/EMP001",
  {
    method: "DELETE",
    headers: {
      "X-API-Key": "<ADMIN_API_KEY>"
    }
  }
);

const result = await response.json();

console.log(result);
```

Note: CORS currently only allows requests from `http://localhost:3000` and `http://127.0.0.1:3000` (see `app/api.py`). Add your deployed frontend's origin there before calling this API from a production web app.

## Employee Embeddings (storage)

Employee face embeddings are stored in **MongoDB**, not on disk:

* Database: `face_recognition`
* Collection: `employee_embeddings`
* Document shape:

```json
{
  "employee_id": "EMP001",
  "embedding": [0.0123, "... 512 floats total"],
  "created_at": "2026-01-01T00:00:00Z",
  "updated_at": "2026-01-01T00:00:00Z"
}
```

## Registering a New Employee

Enroll an employee by calling `POST /embed` with their `employee_id` and a face photo (see [Example Requests](#example-requests) above). This is the only enrollment path — the older `.pkl`-file/offline enrollment script has been removed since it predated the MongoDB integration and was never used by the live API.

## Removing an Employee

Call `DELETE /employees/{employee_id}` with a valid `X-API-Key` header (see [Endpoints](#delete-employeesemployee_id) above). This permanently removes their stored embedding from MongoDB — they will need to be re-enrolled via `/embed` before `/verify` will work for them again.
