@echo off
title Crypto Bot Dashboard
cd /d "%~dp0"
echo ================================================
echo   Crypto Bot Dashboard
echo ================================================
echo.
echo Oeffnet http://localhost:8501 im Browser
echo.
streamlit run dashboard.py --server.port 8501 --server.headless false
pause
