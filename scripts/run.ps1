param(
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ModuleArgs
)
$ErrorActionPreference = 'Stop'
$opdRoot = Split-Path -Parent $PSScriptRoot
$opdPython = 'D:\.conda\envs\model\python.exe'
$opdPriorPythonPath = $env:PYTHONPATH
$opdPriorEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = Join-Path $opdRoot 'src'
    if ($opdPriorPythonPath) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $opdPriorPythonPath }
    $env:PYTHONIOENCODING = 'utf-8'
    & $opdPython -m $Module @ModuleArgs
    $opdExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $opdPriorPythonPath
    $env:PYTHONIOENCODING = $opdPriorEncoding
}
exit $opdExitCode
