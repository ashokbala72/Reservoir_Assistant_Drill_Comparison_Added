@echo off
echo Starting HuggingFace Reservoir Assistant...

REM Set full path to your Python interpreter here:
set PYTHON_PATH="C:\Users\AshokBalasubramanian\AppData\Local\Programs\Python\Python311\python.exe"

%PYTHON_PATH% -m pip install --upgrade pip
%PYTHON_PATH% -m pip install streamlit transformers accelerate sentencepiece torch matplotlib fpdf pandas

echo Packages installed. Launching the app...
%PYTHON_PATH% -m streamlit run reservoir_assistant_HF.py

pause
