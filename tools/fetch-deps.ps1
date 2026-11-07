# Downloads piper-phonemize (espeak-ng + onnxruntime 1.14.1) for Windows
# into deps\piper-phonemize. Run once before configuring CMake.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$deps = Join-Path $root 'deps'
New-Item -ItemType Directory -Force $deps | Out-Null
$zip = Join-Path $deps 'piper-phonemize_windows_amd64.zip'
if (-not (Test-Path $zip)) {
    $url = 'https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper-phonemize_windows_amd64.zip'
    Write-Host "Downloading $url"
    Invoke-WebRequest $url -OutFile $zip
}
if (-not (Test-Path (Join-Path $deps 'piper-phonemize\lib\piper_phonemize.lib'))) {
    tar.exe -xf $zip -C $deps
}
Write-Host "piper-phonemize ready in $deps\piper-phonemize"
