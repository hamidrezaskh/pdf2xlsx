# Waits for the torch/transformers pip installs, then runs the TrOCR
# smoke test. Started detached; result lands in trocr_smoke.log.
$ErrorActionPreference = "SilentlyContinue"
$torchPid = Get-Content D:\pdf2xlsx\pip_torch2.pid
$tfPid = Get-Content D:\pdf2xlsx\pip_tf.pid
while ($true) {
    $t = Get-Process -Id $torchPid -ErrorAction SilentlyContinue
    $f = Get-Process -Id $tfPid -ErrorAction SilentlyContinue
    if (-not $t -and -not $f) { break }
    Start-Sleep -Seconds 10
}
Set-Location D:\pdf2xlsx
"=== deps finished, starting smoke $(Get-Date -Format o) ===" | Set-Content D:\pdf2xlsx\trocr_smoke.log
& D:\pdf2xlsx\.venv\Scripts\python.exe D:\pdf2xlsx\_smoke_trocr.py *>> D:\pdf2xlsx\trocr_smoke.log
"exit=$LASTEXITCODE" | Add-Content D:\pdf2xlsx\trocr_smoke.log
