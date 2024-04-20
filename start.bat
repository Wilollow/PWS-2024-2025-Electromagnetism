@REM @echo off

if not defined CURRDIR (set CURRDIR=%~dp0%)

if not defined PYTHON (set PYTHON=python)
if defined GIT (set "GIT_PYTHON_GIT_EXECUTABLE=%GIT%")
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

set PYTHON="%VENV_DIR%\Scripts\Python.exe"
set VENVACTIVATE="%VENV_DIR%\Scripts\activate.bat"
set RUNSCRIPT="%CURRDIR%source\visualise\electricField\emfield.py"
echo venv %PYTHON%
echo script %RUNSCRIPT%

call %VENVACTIVATE%

@REM call python %RUNSCRIPT%

call manim -p -qh %RUNSCRIPT% SquareToCircle

pause
