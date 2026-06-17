# Build one TikZ figure to figures/<Out>.svg (production) and figures/<Out>.png (preview).
#
#   powershell -File build.ps1 -Name fig_2_2_block -Out fig-2-2-block
#
# Pipeline: pdflatex -> PDF (tight standalone crop); PDF -> SVG (text as paths,
# so it renders identically everywhere); MiKTeX ghostscript -> PNG.
#
# SVG step: dvisvgm --pdf for pure-vector figures. Figures that embed a raster
# via \includegraphics (e.g. 8.2's glyphs from fig 7.2) are routed through
# pdftocairo instead, because this dvisvgm build silently drops embedded images;
# pdftocairo carries them in as base64 and also renders text as paths.
param(
  [Parameter(Mandatory = $true)][string]$Name,
  [Parameter(Mandatory = $true)][string]$Out
)
$ErrorActionPreference = 'Stop'

$mik = "$env:USERPROFILE\scoop\apps\latex\current\texmfs\install\miktex\bin\x64"
if (Test-Path $mik) { $env:Path = "$mik;$env:Path" }

$src    = $PSScriptRoot
$build  = Join-Path $src 'build'
$figdir = (Resolve-Path (Join-Path $src '..')).Path
New-Item -ItemType Directory -Force -Path $build | Out-Null

Push-Location $src
try {
  # cwd = src so \input{ssmtikz.tex} resolves; PDF written into build/.
  # Capture pdflatex's verbose stdout; only surface it on failure.
  $texlog = & pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build" "$Name.tex"
  if ($LASTEXITCODE -ne 0) {
    $texlog | Select-Object -Last 30 | ForEach-Object { Write-Output $_ }
    throw "pdflatex failed for $Name (see $build\$Name.log)"
  }
}
finally { Pop-Location }

$pdf = Join-Path $build  "$Name.pdf"
$svg = Join-Path $figdir "$Out.svg"
$png = Join-Path $figdir "$Out.png"

# Route image-embedding figures through pdftocairo; dvisvgm drops embedded rasters.
$embedsRaster = Select-String -Path (Join-Path $src "$Name.tex") -Pattern '\\includegraphics' -Quiet
if ($embedsRaster) {
  $cairo = (Get-Command pdftocairo -ErrorAction SilentlyContinue)
  if (-not $cairo) {
    throw "$Name embeds a raster (\includegraphics) but pdftocairo was not found; its glyphs would be dropped by dvisvgm. Install poppler (pdftocairo)."
  }
  & pdftocairo -svg "$pdf" "$svg"
  if ($LASTEXITCODE -ne 0) { throw "pdftocairo failed for $Name" }
}
else {
  dvisvgm --pdf --no-fonts --exact-bbox --output="$svg" "$pdf"
  if ($LASTEXITCODE -ne 0) { throw "dvisvgm failed for $Name" }
}

mgs -dSAFER -dBATCH -dNOPAUSE -dQUIET -sDEVICE=pngalpha -r200 `
    -dGraphicsAlphaBits=4 -dTextAlphaBits=4 -o "$png" "$pdf"
if ($LASTEXITCODE -ne 0) { throw "ghostscript PNG failed for $Name" }

Write-Output "built $Out  ->  $svg  +  $png"
