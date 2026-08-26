@echo off
REM One-time setup for Windows.
setlocal
cd /d "%~dp0.."

echo ==^> CIWS setup

where python >/dev/null 2>/dev/null || (echo Python 3.10+ is required. Install it from python.org and tick "Add to PATH". & exit /b 1)
where node   >/dev/null 2>/dev/null || (echo Node 18+ is required for the interface. Install it from nodejs.org. & exit /b 1)

if not exist .venv (
  echo --^> creating the Python environment
  python -m venv .venv
)

echo --^> installing the server
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.venv\Scripts\python.exe -m pip install --quiet -e "apps/server[ingest]"
.venv\Scripts\python.exe -m pip install --quiet anthropic

echo --^> building the interface
pushd apps\web
call npm install --silent
call npm run build:fast
popd

if not exist .env (
  copy .env.example .env >nul
  echo --^> wrote .env ^(add your keys, or leave it empty^)
)

echo.
echo Done. Start it with:  scripts\start.bat
endlocal
