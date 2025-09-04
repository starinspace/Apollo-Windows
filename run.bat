@echo off
REM Activate the Conda environment
call conda activate look2hear_win

REM Make sure output folder exists
if not exist output (
    mkdir output
)

REM List of supported extensions
set EXTENSIONS=mp3 m4a wav aiff aif flac

REM Loop over each extension
for %%e in (%EXTENSIONS%) do (
    for %%f in (input\*.%%e) do (
        REM Get the file name without extension
        set "filename=%%~nf"
        REM Enable delayed expansion for variable inside loop
        setlocal enabledelayedexpansion
        REM Run the Python inference script
        python inference2.py --in_wav="%%f" --out_wav="output\!filename!.wav"
        endlocal
    )
)

echo All files processed!
pause
