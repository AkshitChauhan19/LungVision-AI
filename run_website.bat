@echo off
title PulmoScan AI - Clinical CAD & XAI Platform
cd /d "%~dp0"
echo ========================================================
echo        PULMOSCAN AI - CLINICAL CAD & XAI PLATFORM
echo ========================================================
echo Starting local inference server on http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000
pause
