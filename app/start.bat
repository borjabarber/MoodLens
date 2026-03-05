@echo off
echo ============================================================
echo  MoodLens Web App - Iniciando servidores
echo ============================================================
echo.

echo Iniciando Backend (FastAPI) en puerto 8000...
start "MoodLens Backend" cmd /k "cd /d %~dp0backend && ..\..\.venv\Scripts\python main.py"

timeout /t 3 /nobreak > nul

echo Iniciando Frontend (React) en puerto 5173...
start "MoodLens Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ============================================================
echo  Servidores iniciados:
echo    Backend:  http://localhost:8000
echo    Frontend: http://localhost:5173
echo ============================================================
echo.
echo Abre http://localhost:5173 en tu navegador
pause
