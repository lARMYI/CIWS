@echo off
REM Start CIWS and open it in your browser.
setlocal
cd /d "%~dp0.."
if not exist .venv (echo Run scripts\setup.bat first. & exit /b 1)
if exist .env for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
  if not "%%a"=="" if not "%%a:~0,1%"=="#" set "%%a=%%b"
)
.venv\Scripts\python.exe -m ciws --open %*
endlocal
