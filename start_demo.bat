@echo off
rem Double-click to open the local safety test page. Needs the "Set up" steps in README.md once.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No Python environment found. Do the "Set up" steps in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" demo\web_demo.py %*
pause
