@echo off
title CDSS Application Launching...
call D:\anaconda\Scripts\activate.bat
call conda activate heart_failure
cd /d "%~dp0"
streamlit run app.py
pause