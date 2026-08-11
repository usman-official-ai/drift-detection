@echo off
cd /d "C:\Users\RAYAN COMPUTERs\drift-detection-system"
set PYTHONPATH=%CD%
python -m src.main --mode once --source file >> logs\scheduled_run.log 2>&1