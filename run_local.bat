@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_local.ps1" %*
set EXITCODE=%ERRORLEVEL%
exit /b %EXITCODE%
