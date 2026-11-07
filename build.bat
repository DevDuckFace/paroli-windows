@echo off
REM ============================================================
REM Compila o paroli para Windows (Release).
REM   build.bat          paroli-cli + paroli-stream (rapido)
REM   build.bat server   tambem o paroli-server (Drogon/Opus/soxr;
REM                      a primeira vez demora, o vcpkg compila tudo)
REM Pre-requisitos: Visual Studio 2022 (C++), CMake no PATH e o
REM vcpkg com a variavel VCPKG_ROOT definida.
REM ============================================================
setlocal
cd /d "%~dp0"

if "%VCPKG_ROOT%"=="" (
    echo [ERRO] Defina a variavel de ambiente VCPKG_ROOT apontando para a pasta do vcpkg.
    exit /b 1
)
where cmake >nul 2>nul
if errorlevel 1 (
    echo [ERRO] CMake nao encontrado no PATH.
    exit /b 1
)

REM O CMake baixa o piper-phonemize sozinho; o script so adianta o download.
echo == Dependencias (piper-phonemize: espeak-ng + onnxruntime) ==
powershell -NoProfile -ExecutionPolicy Bypass -File tools\fetch-deps.ps1
if errorlevel 1 (
    echo [ERRO] Falha ao baixar o piper-phonemize.
    exit /b 1
)

set BUILD_DIR=build
set EXTRA=-DPAROLI_BUILD_SERVER=OFF
if /i "%~1"=="server" (
    set BUILD_DIR=build-server
    set EXTRA=-DPAROLI_BUILD_SERVER=ON -DVCPKG_MANIFEST_FEATURES=server
)

echo == Configurando (%BUILD_DIR%) ==
cmake -B %BUILD_DIR% -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" %EXTRA%
if errorlevel 1 (
    echo [ERRO] Falha ao configurar o CMake.
    exit /b 1
)

echo == Compilando ==
cmake --build %BUILD_DIR% --config Release
if errorlevel 1 (
    echo [ERRO] Falha ao compilar.
    exit /b 1
)

echo.
echo Build concluido: %BUILD_DIR%\Release\
echo   paroli-cli.exe, paroli-stream.exe (e paroli-server.exe com "server"),
echo   DLLs e espeak-ng-data ao lado, prontos para copiar.
echo O DuckFaceLLM pega o runtime de ..\paroli-windows\build\Release.
endlocal
