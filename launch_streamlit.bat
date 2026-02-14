@echo off
REM Launch script for Customer Support Chatbot with Streamlit UI
echo ============================================
echo   Customer Support Chatbot Launcher
echo ============================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call myenv\Scripts\activate.bat
    echo.
)

REM Start FastAPI backend in background
echo Starting FastAPI backend on port 8000...
start "FastAPI Backend" cmd /k "python api.py"
timeout /t 5 /nobreak > nul

REM Start Streamlit frontend
echo Starting Streamlit UI on port 8501...
streamlit run streamlit_app.py

pause
