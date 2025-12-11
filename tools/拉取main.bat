@echo off
echo ==========================================
echo   FORCE RESET main to origin/main
echo ==========================================
echo.

cd /d %~dp0

git checkout main
git fetch origin

echo.
echo HARD RESETTING main...
git reset --hard origin/main

echo.
echo Done! Local main = remote main.
pause
