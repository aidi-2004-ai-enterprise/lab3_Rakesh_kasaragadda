# lab3_Rakesh_kasaragadda
## Lab 3: Penguins Classification with XGBoost and FastAPI

### Demo
![Demo](rakesh_demo.gif)

Also available as a full [MP4 screen recording](./Rakesh_Lab3_Demo.mp4)

---

This project demonstrates how to build and deploy a machine learning pipeline using the **Seaborn Penguins** dataset. We train an **XGBoost classifier** and deploy it using **FastAPI**, ensuring proper input validation, logging, and error handling.

---

## Features

- Cleaned and preprocessed penguins dataset
- One-hot encoding for categorical features, label encoding for species
- XGBoost classifier model training
- FastAPI app with `/predict` and `/health` endpoints
- Input validation with Pydantic
- Logging and error handling (422, 400 status codes)
- Project dependency management using `uv` (an ultra-fast Python package manager)

---

## Getting Started

### Prerequisites

- Python 3.10+
- `uv` installed:  
  ```bash
  pip install uv
  ```

---

### 🔧 Installation

Clone the repository and navigate into the project:

```bash
git clone https://github.com/aidi-2004-ai-enterprise/lab3_Rakesh_kasaragadda.git
cd lab3_Rakesh_kasaragadda
```

Install all dependencies:

```bash
uv venv
uv pip install -r pyproject.toml
```

---

## 🏋️‍♂️ Train the Model

```bash
python train.py
```

This will generate:
- `model.json`
- `label_encoder.pkl`
- `columns.pkl`

under `app/data/`

---

## 🖥️ Run the FastAPI App

```bash
uvicorn app.main:app --reload
```

Then open your browser at [http://localhost:8000/docs](http://localhost:8000/docs) to access the interactive Swagger UI.

---

## 📁 Project Structure

```markdown
lab3_Rakesh_kasaragadda/
├── app/
│   ├── main.py                 # FastAPI app with prediction endpoint
│   ├── data/
│   │   ├── model.json          # Trained XGBoost model
│   │   ├── label_encoder.pkl   # Label encoder for species
│   │   └── columns.pkl         # Feature columns used in training
├── train.py                    # Script to train the XGBoost model
├── pyproject.toml              # Dependency configuration (managed by uv)
├── uv.lock                     # Lock file for reproducibility
├── .gitignore                  # Files ignored by Git
├── rakesh_demo.gif             # Demo GIF for README
├── Rakesh_Lab3_Demo.mp4        # Full screen recording
└── README.md                   # Project documentation
```

---

##  API Endpoints

### `GET /health`
Basic health check.
```json
{"status": "ok"}
```

### `POST /predict`
Takes a JSON payload of features and returns the predicted species.

Example:
```json
{
  "island": "Torgersen",
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0,
  "sex": "male"
}
```

---

## Author
Rakesh Kasaragadda
Duham college, Oshawa
