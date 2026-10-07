# cuda-env.ps1
# Prepends a managed CUDA / cuDNN runtime bin directory to PATH for the current
# shell so CTranslate2 can find the required DLLs on Windows.
#
# Safe to dot-source repeatedly: the entry is never duplicated.
#
# Usage:
#   . ".\cuda-env.ps1" -CudaBin "C:\Program Files\NVIDIA\CUDNN\bin"
#   # or set the CUDA_BIN environment variable first:
#   $env:CUDA_BIN = "C:\Program Files\NVIDIA\CUDNN\bin"
#   . ".\cuda-env.ps1"

param(
    [string]$CudaBin = $env:CUDA_BIN
)

if (-not $CudaBin) {
    Write-Warning "No CUDA bin directory provided. Pass -CudaBin or set CUDA_BIN."
    return
}

if (Test-Path -LiteralPath $CudaBin) {
    $target = $CudaBin.TrimEnd('\').ToLower()
    $present = @($env:PATH -split ';' |
        Where-Object { $_ -ne '' } |
        ForEach-Object { $_.TrimEnd('\').ToLower() }) -contains $target

    if (-not $present) {
        $env:PATH = "$CudaBin;$env:PATH"
    }
}
else {
    Write-Warning "CUDA bin directory not found: $CudaBin"
}
