@echo off
rem Double-click to launch the CompanionAI Streamlit interactive practice demo.
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
  if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run demo\streamlit\app.py %*
    pause
    exit /b 0
  )
  echo No Python virtual environment found at .venv.
  echo Please run the setup steps in README.md to create .venv and install dependencies:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\streamlit.exe" run demo\streamlit\app.py %*
pause
