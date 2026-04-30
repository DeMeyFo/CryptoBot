@echo off
title Crypto Futures Bot
cd /d "%~dp0"
echo ================================================
echo   Crypto Futures Bot
echo ================================================
echo.
echo Installiere Abhaengigkeiten ...
pip install -r requirements.txt -q
echo.
python main.py
pause
