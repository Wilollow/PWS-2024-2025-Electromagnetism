@REM @echo off

if not defined CURRDIR (set CURRDIR=%~dp0%)

if not defined PYTHON (set PYTHON=python)
if defined GIT (set "GIT_PYTHON_GIT_EXECUTABLE=%GIT%")
if not defined VENV_DIR (set "VENV_DIR=%~dp0%.venv")

mkdir tmp 2>NUL

goto :check_pip
pause

:check_pip
%PYTHON% -mpip --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :create-venv
if "%PIP_INSTALLER_LOCATION%" == "" goto :show_stdout_stderr
%PYTHON% "%PIP_INSTALLER_LOCATION%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :create-venv
echo Couldn't install pip
goto :show_stdout_stderr

:create-venv
dir "%VENV_DIR%\Scripts\Python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate-venv

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate-venv
echo Unable to create venv in directory "%VENV_DIR%"
goto :show_stdout_stderr

:activate-venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
set VENVACTIVATE="%VENV_DIR%\Scripts\activate.bat"
set RUNSCRIPT="%CURRDIR%source\visualise\electricField\emfield.py"
echo venv %PYTHON%
echo script %RUNSCRIPT%
call %VENVACTIVATE%
pip install -r %CURRDIR%requirements.txt
goto :run-program

:run-program
call cd source\visualise
call manimgl -w electricField/emfield.py InteractiveDevelopment
@REM call manimgl -wo render_scene.py RenderScene
if %ERRORLEVEL% == 0 goto :end_program_succesful
goto :show_stdout_stderr

:end_program_succesful
echo program exited succesfully with error code 0, exiting...
goto:end

:show_stdout_stderr
echo well, fuck... I give up, just mail me the problem or something
goto:end

:end
pause