@echo off
rem ---------------------------------------------------------------------------
rem  KEEP THIS FILE PURE ASCII.
rem
rem  With "chcp 65001" active, cmd.exe advances its position in a .bat file by
rem  BYTES while decoding the text as multi-byte UTF-8 characters. After any line
rem  holding non-ASCII text the file pointer lands in the middle of a line, and
rem  cmd runs the leftover fragment as a command:
rem
rem      '...' is not recognized as an internal or external command
rem
rem  Korean messages are printed by labeler.py instead, where it is safe.
rem ---------------------------------------------------------------------------
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" labeler.py review
pause
