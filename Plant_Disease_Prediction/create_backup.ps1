# Create a timestamped backup ZIP of the local PlantVision project.
# Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File .\create_backup.ps1
#
# The archive is written beside the project folder, so it is not included in itself.
# Dataset and .keras model files are included. Virtual environments, Git internals,
# caches, and node_modules are excluded to keep the backup useful and portable.

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path $PSScriptRoot).Path
$ParentDirectory = Split-Path -Parent $ProjectRoot
$ProjectName = Split-Path -Leaf $ProjectRoot
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupPath = Join-Path $ParentDirectory "${ProjectName}_BACKUP_${Timestamp}.zip"

$ExcludedDirectoryNames = @(
    ".git",
    ".venv",
    ".venv-model",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules"
)

$Files = Get-ChildItem -LiteralPath $ProjectRoot -File -Recurse -Force |
    Where-Object {
        $relative = $_.FullName.Substring($ProjectRoot.Length).TrimStart([char]92, [char]47)
        $parts = $relative -split '[\\/]'
        -not ($parts | Where-Object { $ExcludedDirectoryNames -contains $_ })
    }

if (-not $Files -or $Files.Count -eq 0) {
    throw "No project files found under $ProjectRoot"
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$FileStream = [System.IO.File]::Open(
    $BackupPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None
)

try {
    $Archive = [System.IO.Compression.ZipArchive]::new(
        $FileStream,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $true
    )

    $ManifestLines = [System.Collections.Generic.List[string]]::new()
    $ManifestLines.Add("PlantVision AI local backup manifest")
    $ManifestLines.Add("Created: $(Get-Date -Format o)")
    $ManifestLines.Add("Project: $ProjectRoot")
    $ManifestLines.Add("Excluded directories: $($ExcludedDirectoryNames -join ', ')")
    $ManifestLines.Add("")
    $ManifestLines.Add("SHA256 | SizeBytes | RelativePath")

    foreach ($File in $Files) {
        $RelativePath = $File.FullName.Substring($ProjectRoot.Length).TrimStart([char]92, [char]47)
        $EntryName = $RelativePath.Replace('\', '/')
        $Entry = $Archive.CreateEntry(
            $EntryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        )

        $InputStream = [System.IO.File]::OpenRead($File.FullName)
        try {
            $OutputStream = $Entry.Open()
            try {
                $InputStream.CopyTo($OutputStream)
            }
            finally {
                $OutputStream.Dispose()
            }
        }
        finally {
            $InputStream.Dispose()
        }

        $Hash = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash
        $ManifestLines.Add("$Hash | $($File.Length) | $RelativePath")
    }

    $ManifestEntry = $Archive.CreateEntry(
        "BACKUP_MANIFEST.txt",
        [System.IO.Compression.CompressionLevel]::Optimal
    )
    $ManifestStream = $ManifestEntry.Open()
    try {
        $ManifestWriter = [System.IO.StreamWriter]::new(
            $ManifestStream,
            [System.Text.Encoding]::UTF8
        )
        try {
            foreach ($Line in $ManifestLines) {
                $ManifestWriter.WriteLine($Line)
            }
        }
        finally {
            $ManifestWriter.Dispose()
        }
    }
    finally {
        $ManifestStream.Dispose()
    }

    $Archive.Dispose()
}
catch {
    if (Test-Path -LiteralPath $BackupPath) {
        Remove-Item -LiteralPath $BackupPath -Force
    }
    throw
}
finally {
    $FileStream.Dispose()
}

# Re-open the archive and verify that its directory can be read.
$CheckArchive = [System.IO.Compression.ZipFile]::OpenRead($BackupPath)
try {
    if ($CheckArchive.Entries.Count -lt 2) {
        throw "Backup archive verification failed: too few entries."
    }
}
finally {
    $CheckArchive.Dispose()
}

$BackupInfo = Get-Item -LiteralPath $BackupPath
Write-Host ""
Write-Host "BACKUP CREATED AND ZIP DIRECTORY VERIFIED" -ForegroundColor Green
Write-Host "File: $($BackupInfo.FullName)"
Write-Host "Files included (excluding manifest): $($Files.Count)"
Write-Host "Archive size: $([math]::Round($BackupInfo.Length / 1MB, 2)) MB"
Write-Host "Dataset and .keras checkpoints are included if present in the project folder."
Write-Host "A SHA-256 manifest is included inside the archive."
