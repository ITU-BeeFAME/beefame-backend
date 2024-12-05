# BeeFair

## Requirements

- **Python**: Version 3.9 or higher
- **Dependencies**: Install using the provided `requirements.txt`

## Setup

### 1. Clone the repository

```
git clone https://github.com/ITU-BeeFair/beefair-backend.git
cd beefair-backend
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

## Running the application

```
uvicorn app.main:app --reload
```

The API documentation will be available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Redoc: `http://127.0.0.1:8000/redoc`
