# lab3_Rakesh_kasaragadda
# 🐧 Lab 3: Penguins Classification with XGBoost and FastAPI

🎥 **[Click here to watch the demo](./Rakesh_Lab3_Demo.mp4)**

This project focuses on building a machine learning pipeline using the Seaborn Penguins dataset. 

- Cleaned the dataset by removing null values and dropping the unused `year` column.
- Trained an XGBoost classifier with one-hot encoding for categorical features and label encoding for target.
- Deployed the trained model using FastAPI with `/predict` and `/health` endpoints.
- Added Pydantic input validation, proper logging, and graceful error handling (422 for input validation errors and 400 for internal prediction errors).

## 📁 Project Structure

```markdown
├── train.py                  # Loads data, preprocesses, trains model, saves model files
├── app/
│   ├── main.py               # FastAPI app with prediction endpoints
│   └── data/
│       ├── model.json
│       ├── label_encoder.pkl
│       └── columns.pkl
├── pyproject.toml            # Dependency management with uv
├── Rakesh_Lab3_Demo.mp4      # Demo video
└── README.md                 # Project overview
