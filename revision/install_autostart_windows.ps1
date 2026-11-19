# Registers a Windows scheduled task that re-starts run_all.bat automatically
# after a reboot / power cut (at user logon).  Run once in PowerShell:
#     powershell -ExecutionPolicy Bypass -File install_autostart_windows.ps1
# Remove with:
#     Unregister-ScheduledTask -TaskName "PCE-Revision" -Confirm:$false
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat = Join-Path $here "run_all.bat"
$log = Join-Path $here "run_all_autostart.log"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"`"$bat`" >> `"$log`" 2>&1`"" -WorkingDirectory $here
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Days 30) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "PCE-Revision" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Scheduled task 'PCE-Revision' registered. It runs run_all.bat at every logon of $env:USERNAME."
Write-Host "Tip: enable automatic logon and 'Restore on AC power loss = Power On' in the BIOS"
Write-Host "     so that the run continues by itself after a power cut."
