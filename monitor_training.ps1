# Monitor training progress

$bg_medium = "C:\Users\J_lin\AppData\Local\Temp\claude\C--Users-J-lin-Desktop-franken-ao\36481111-0345-4854-92bd-441e7551fd10\tasks\brkfeotxg.output"
$bg_big = "C:\Users\J_lin\AppData\Local\Temp\claude\C--Users-J-lin-Desktop-franken-ao\36481111-0345-4854-92bd-441e7551fd10\tasks\blftbbz9d.output"

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host " TRAINING MONITOR" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan

# Check medium
Write-Host ""
Write-Host "MEDIUM (32/64/5, 8M chars, 2000 steps):" -ForegroundColor Yellow
if (Test-Path $bg_medium) {
    $size = (Get-Item $bg_medium).Length
    if ($size -eq 0) {
        Write-Host "  [RUNNING] No output yet (initializing...)" -ForegroundColor Gray
    } else {
        Write-Host "  [RUNNING] Output size: $size bytes" -ForegroundColor Green
        $lines = @(Get-Content $bg_medium -Tail 5)
        foreach ($line in $lines) {
            if ($line -match "step|acc") {
                Write-Host "    $line" -ForegroundColor Gray
            }
        }
    }
} else {
    Write-Host "  [NOT FOUND]" -ForegroundColor Red
}

# Check big
Write-Host ""
Write-Host "BIG (64/64/7, 17.9M chars, 5000 steps):" -ForegroundColor Yellow
if (Test-Path $bg_big) {
    $size = (Get-Item $bg_big).Length
    if ($size -eq 0) {
        Write-Host "  [RUNNING] No output yet (initializing...)" -ForegroundColor Gray
    } else {
        Write-Host "  [RUNNING] Output size: $size bytes" -ForegroundColor Green
        $lines = @(Get-Content $bg_big -Tail 5)
        foreach ($line in $lines) {
            if ($line -match "step|acc") {
                Write-Host "    $line" -ForegroundColor Gray
            }
        }
    }
} else {
    Write-Host "  [NOT FOUND]" -ForegroundColor Red
}

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host " Check again: .\monitor_training.ps1" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
