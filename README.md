# lab3_Rakesh_kasaragadda
# Lab 3: Penguins Classification with XGBoost and FastAPI

## 🎥 Demo

![Demo](rakesh_demo.gif)

This project focuses on building a machine learning pipeline using the Seaborn Penguins dataset. 

- Cleaned the dataset by removing null values and dropping the unused `year` column.
- Trained an XGBoost classifier with one-hot encoding for categorical features and label encoding for target.
- Deployed the trained model using FastAPI with `/predict` and `/health` endpoints.
- Added Pydantic input validation, proper logging, and graceful error handling (422 for input validation errors and 400 for internal prediction errors).

## 📁 Project Structure

```
lab3_Rakesh_kasaragadda/
├── app/
│   ├── main.py                 # FastAPI application with prediction endpoint
│   ├── data/
│   │   ├── model.json          # Trained XGBoost model
│   │   ├── label_encoder.pkl   # Label encoder for species
│   │   └── columns.pkl         # Feature columns used for prediction
├── train.py                    # Script to train and save model artifacts
├── pyproject.toml             # Project dependencies managed using uv
├── uv.lock                    # Lock file for reproducible builds
├── rakesh_demo.gif            # Git ignore file
├── .gitignore                 #gif
├── Rakesh_Lab3_Demo.mp4       # Demo screen recording of the application
└── README.md                  # Project documentation
```
