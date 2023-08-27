$python_path = $null
try
{
	$pversions = py -0p
	$version = $pversions.Where( {$_.Contains(":3.9") }, 'First')
	if([bool] $version)
	{
		$path = $version.Split("  ")[-1].Trim()
		$python_path = $path
	}
}
catch
{
	"py launcher not found"
}

if ($null -eq $python_path)
{

	"Python 3.9 not found, make sure it is installed and added to PATH"
	"ERROR"
	exit
}
"Found Python 3.9 at ${python_path}"

& $python_path -m pip install -r requirements.txt --force-reinstall

if ($LASTEXITCODE -ne 0)
{
	"Could not install dependencies"
    "ERROR"
	exit
}

New-Item -ItemType Directory -Path .download -Force
try
{
    Invoke-WebRequest -Uri https://github.com/Vogelwarte/SnowfinchWire.Common/archive/refs/tags/v1.0-beggingcallsanalyzer.zip -OutFile .download/common.zip
    # This will only execute if the Invoke-WebRequest is successful.
} catch {
    $StatusCode = $_.Exception.Response.StatusCode.value__
	"Github API error: ${StatusCode}"
	"ERROR"
	exit
}

Expand-Archive .download/common.zip -DestinationPath .download/
New-Item -ItemType Directory -Path beggingcallsanalyzer/common -Force
Copy-Item -Recurse -Path .download/SnowfinchWire.Common-1.0-beggingcallsanalyzer/* -Destination beggingcallsanalyzer/common/
Remove-Item -Recurse .download
"SUCCESS"
