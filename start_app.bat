@echo off
cd /d "%~dp0"
".venv-train\Scripts\python.exe" run.py
if errorlevel 1 pause
