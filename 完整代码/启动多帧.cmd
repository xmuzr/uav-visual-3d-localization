@echo off
python -B "%~dp0run.py" --multi
if errorlevel 1 (
  echo Operation failed. See README.md for setup instructions.
  pause
  exit /b 1
)
