# Face Recognition API

Python FastAPI backend for employee face verification.

## Features

* Employee face verification
* Face embedding generation
* No face detection
* Multiple faces detection
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

### POST /verify

Verifies an employee's face.

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
* `TOO_DARK`
* `TOO_BRIGHT`
* `TOO_FAR`
* `TOO_CLOSE`
* `FACE_TURNED`
* `INVALID_IMAGE`
* `EMPLOYEE_NOT_FOUND`
* `ERROR`

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

## Example Request

The Next.js frontend should send:

```text
POST http://127.0.0.1:8000/verify
```

Using `FormData`:

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

## Employee Embeddings

Employee face embeddings are stored in:

```text
data/embeddings/
```

Example:

```text
EMP001.pkl
EMP002.pkl
EMP003.pkl
EMP004.pkl
```

## Registering a New Employee

1. Add the employee image inside:

```text
data/employees/EMP005/
```

2. Add the employee details to `app/register_employees.py`.

3. Run:

```powershell
python -m app.register_employees
```

4. The embedding will be created:

```text
data/embeddings/EMP005.pkl
```

The new employee can then be verified using:

```text
employee_id = EMP005
```
