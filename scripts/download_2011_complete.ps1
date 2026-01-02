$baseDir = "data\raw\2011"
New-Item -ItemType Directory -Force -Path $baseDir | Out-Null

$baseUrl = "https://storage.googleapis.com/clusterdata-2011-2/job_events"

Write-Host "=========================================="
Write-Host "Downloading Google Cluster Data 2011"
Write-Host "=========================================="
Write-Host "Note: There are 500 files (parts 00000-00499)"
Write-Host "Downloading all 500 files..."
Write-Host ""

$downloaded = 0
$failed = 0

# Télécharger tous les fichiers (00000 à 00499)
0..499 | ForEach-Object {
    $shard = $_.ToString("D5")
    $filename = "part-$shard-of-00500.csv.gz"
    $url = "$baseUrl/$filename"
    $output = "$baseDir\$filename"
    
    if (Test-Path $output) {
        Write-Host "[OK] Already downloaded: $filename" -ForegroundColor Green
        $downloaded++
    } else {
        try {
            Write-Host "[DL] Downloading: $filename" -ForegroundColor Cyan
            Invoke-WebRequest -Uri $url -OutFile $output -TimeoutSec 300
            $size = (Get-Item $output).Length / 1MB
            Write-Host "  [OK] Success ($([Math]::Round($size, 2)) MB)" -ForegroundColor Green
            $downloaded++
        }
        catch {
            Write-Host "  [FAIL] Failed: $_" -ForegroundColor Red
            $failed++
        }
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Summary:"
Write-Host "  Downloaded: $downloaded"
Write-Host "  Failed: $failed"
Write-Host "=========================================="