@echo off
if exist "%~dp0dependency_wheels\*.whl" (
  python -m pip install --no-index --find-links "%~dp0dependency_wheels" -r "%~dp0requirements.txt"
) else (
  python -m pip install -r "%~dp0requirements.txt"
)
if errorlevel 1 (
  echo Operation failed. See README.md for setup instructions.
  pause
  exit /b 1
)
