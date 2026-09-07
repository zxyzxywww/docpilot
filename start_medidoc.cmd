@echo off
title MediDoc - Start Services
cd /d "%~dp0"

set "PATH=C:\Users\29446\AppData\Local\Programs\DockerDesktop\resources\bin;%PATH%"

echo ==============================================
echo   MediDoc Knowledge Q&A - Service Launcher
echo   Qdrant(6333) + API(8000) + Web(3000)
echo ==============================================
echo.

echo [1/4] Checking Docker engine...
docker info >nul 2>&1
if not errorlevel 1 goto docker_ok

echo [!] Docker not running, starting Docker Desktop...
start "" "C:\Users\29446\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe"
echo     Waiting for Docker Desktop (up to 60s)...
set /a n=0

:waitdocker
timeout /t 5 /nobreak >nul
docker info >nul 2>&1
if not errorlevel 1 goto docker_ok
set /a n+=1
if %n% lss 12 goto waitdocker
echo [!] Docker failed to start. Open Docker Desktop manually, then rerun.
pause
exit /b 1

:docker_ok
echo [2/4] Starting containers (qdrant + api + web)...
docker compose up -d
if errorlevel 1 (
    echo [!] Container startup failed. See message above.
    pause
    exit /b 1
)

echo [3/4] Waiting for API to be ready...
set /a n=0

:waitapi
curl -s -o nul http://localhost:8000/api/health
if not errorlevel 1 goto api_ok
timeout /t 3 /nobreak >nul
set /a n+=1
if %n% lss 20 goto waitapi
echo [!] API not ready after ~60s. Check: docker compose ps
pause
exit /b 1

:api_ok
echo [4/4] Opening browser...
start "" http://localhost:3000
echo.
echo ==============================================
echo   [OK] Web  : http://localhost:3000
echo        API  : http://localhost:8000/docs
echo.
echo   Stop services : docker compose down
echo ==============================================
echo.
pause
