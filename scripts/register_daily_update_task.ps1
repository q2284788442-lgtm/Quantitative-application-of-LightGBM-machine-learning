param(
    [string]$TaskName = "LightGBM IM Daily Update Train",
    [string]$PythonExe = "D:\anaconda3\envs\py39\python.exe",
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$At = "18:30"
)

$DailyScript = Join-Path $ProjectRoot "scripts\run_daily_update_and_train.py"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path -LiteralPath $DailyScript)) {
    throw "Daily script not found: $DailyScript"
}

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$DailyScript`"" `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Update CSI 1000 daily K data with AkShare, retrain LightGBM, and write latest signal." `
    -Force

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Run time: daily at $At"
Write-Host "Project root: $ProjectRoot"
Write-Host "Python: $PythonExe"
Write-Host "Script: $DailyScript"
