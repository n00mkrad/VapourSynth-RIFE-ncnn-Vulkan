@echo off

if not defined VSCMD_VER (
    echo Initializing MS Build Tools...
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    echo MS Build Tools environment already initialized.
)

echo Compiling: %~dp0build
ninja -C "%~dp0build"

if %errorlevel% neq 0 (
    echo Aborting after running ninja - ERRORLEVEL %errorlevel%
    goto end
)

echo Renaming to rifemv.dll...
move "%~dp0build\rife.dll" "%~dp0build\rifemv.dll"

:: If file exists at target (renamed) location, print its full path and size
if exist "%~dp0build\rifemv.dll" (
    for %%F in ("%~dp0build\rifemv.dll") do (
        echo File: %%~fF
        echo Size: %%~zF bytes
    )
) else (
    echo Error: rifemv.dll not found after attempting to rename.
)

if %errorlevel% neq 0 (
    echo Finished with ERRORLEVEL %errorlevel%
) else (
    echo Done.
)

:end