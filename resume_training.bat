@echo off
cd /d "%~dp0"
".venv-train\Scripts\python.exe" -u train.py --epochs 20 --batch-size 8 --resume
pause
