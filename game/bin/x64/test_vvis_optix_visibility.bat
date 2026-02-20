@echo off
pwsh.exe -ExecutionPolicy Bypass -File .\run_tests.ps1 -TestNames visibility %*
