@echo off
REM ============================================================
REM  RailPredict AI - Windows helper script
REM  Run this from the project root folder (the one containing
REM  this file, backend\, frontend\, ml\, scripts\, etc.)
REM ============================================================

echo.
echo ===============================================
echo   RailPredict AI - Setup and Run
echo ===============================================
echo.

IF NOT EXIST venv (
    echo [1/6] Creating Python virtual environment...
    python -m venv venv
) ELSE (
    echo [1/6] Virtual environment already exists, skipping.
)

echo [2/6] Activating virtual environment...
call venv\Scripts\activate

echo [3/6] Installing backend dependencies...
pip install -r backend\requirements.txt

IF NOT EXIST data\train_data.csv (
    echo [4/6] Generating synthetic dataset...
    python scripts\generate_dataset.py
) ELSE (
    echo [4/6] Dataset already exists, skipping. Delete data\train_data.csv to regenerate.
)

IF NOT EXIST models\eta_model.joblib (
    echo [5/6] Training ML model...
    python ml\train_model.py
) ELSE (
    echo [5/6] Trained model already exists, skipping. Delete models\eta_model.joblib to retrain.
)

echo [6/6] Starting backend server on http://localhost:8000 ...
echo.
echo IMPORTANT: Keep this window open. Open a SECOND terminal and run:
echo.
echo     cd frontend
echo     npm install
echo     npm run dev
echo.
echo Then open the URL Vite prints (usually http://localhost:5173) in your browser.
echo.

uvicorn backend.main:app --reload

pause
