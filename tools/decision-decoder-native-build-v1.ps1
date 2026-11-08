param(
    [Parameter(Mandatory=$true)][string]$SourceRoot,
    [Parameter(Mandatory=$true)][string]$BuildRoot
)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$decoderWork='C:\Users\Suini\kanachan-data-work'
$decoderAllowed=[IO.Path]::GetFullPath("$decoderWork\diagnostics\decision-decoder-20261104-v1")
$decoderSource=[IO.Path]::GetFullPath($SourceRoot)
$decoderBuild=[IO.Path]::GetFullPath($BuildRoot)
if($decoderSource -ne "$decoderAllowed\source" -or $decoderBuild -ne "$decoderAllowed\native-build"){
    throw 'Native validation paths must remain inside the owned diagnostic directory.'
}
(Get-Process -Id $PID).PriorityClass='BelowNormal'
$decoderDependencies="$decoderWork\runtime\native-deps"
$decoderVs='C:\Program Files\Microsoft Visual Studio\18\Community'
$decoderDevCommand='"'+"$decoderVs\Common7\Tools\VsDevCmd.bat"+'" -no_logo -arch=x64 -host_arch=x64 >nul && set'
foreach($decoderLine in (& cmd.exe /d /s /c $decoderDevCommand)){
    $decoderPair=$decoderLine -split '=',2
    if($decoderPair.Length -eq 2 -and $decoderPair[0]){
        [Environment]::SetEnvironmentVariable($decoderPair[0],$decoderPair[1],'Process')
    }
}
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
$decoderCmake="$decoderVs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$decoderNinja="$decoderVs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$decoderPython="$decoderWork\runtime\python\cpython-3.12-windows-x86_64-none"
& $decoderCmake -S "$decoderSource\tools\runtime" -B $decoderBuild -G Ninja "-DCMAKE_MAKE_PROGRAM=$decoderNinja" -DCMAKE_BUILD_TYPE=Release "-DKANACHAN_BOOST_ROOT=$decoderDependencies\boost_1_85_0" "-DKANACHAN_NYANTEN_ROOT=$decoderDependencies\nyanten" "-DKANACHAN_PYTHON_INCLUDE_DIR=$decoderPython\include" "-DKANACHAN_PYTHON_LIBRARY=$decoderPython\libs\python312.lib"
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
& $decoderCmake --build $decoderBuild --parallel 2
exit $LASTEXITCODE
