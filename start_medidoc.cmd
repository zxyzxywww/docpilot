@echo off
title MediDoc Medical QA Agent - Launcher
cd /d "%~dp0"

echo ==========================================
echo   MediDoc Medical Literature QA Agent
echo   (RAG + Agent, 35 top-journal papers)
echo ==========================================
echo.

REM Docker CLI path (adjust if Docker Desktop installed elsewhere)
set "PATH=C:\Users\29446\AppData\Local\Programs\DockerDesktop\resources\bin;%PATH%"

echo [1/3] Checking Docker engine...
docker info >nul 2>&1
if errorlevel 1 (
    echo [!] Docker is not running, starting Docker Desktop...
    start "" "C:\Users\29446\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe"
    echo     Waiting for Docker Desktop (about 30s)...
    timeout /t 30 /nobreak >nul
    docker info >nul 2>&1
    if errorlevel 1 (
        echo [!] Docker not ready. Open Docker Desktop manually and retry.
        pause
        exit /b 1
    )
)

echo [2/3] Starting containers (Qdrant + API + Web)...
docker compose up -d
if errorlevel 1 (
    echo [!] Container startup failed. Check docker compose config.
    pause
    exit /b 1
)

echo [3/3] Waiting for app, opening browser...
timeout /t 12 /nobreak >nul
start "" http://localhost:3000

echo.
echo ==========================================
echo   [OK] App started:  http://localhost:3000
echo.
echo   Stop service:  docker compose down
echo   Check status:  docker compose ps
echo   Local dev:     python -m streamlit run src/app/app.py
echo   (local dev needs qdrant.mode=local in config.yaml)
echo ==========================================
echo.
pause
