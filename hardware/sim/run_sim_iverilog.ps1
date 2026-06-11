# Build golden memory files, copy weight ROMs in, compile and run with Icarus.
#
#   & .\run_sim_iverilog.ps1

$ErrorActionPreference = "Stop"

$here    = $PSScriptRoot
$rtl     = Join-Path $here "..\rtl"
$weights = Join-Path $here "..\..\models\weights_q4_12"
# Icarus bin dir. Override with $env:IVERILOG_BIN if installed elsewhere.
$bin = if ($env:IVERILOG_BIN) { $env:IVERILOG_BIN } else { "C:\iverilog\bin" }

Set-Location $here

# 1. regenerate golden memory files
python make_golden_mem.py

# 2. weight ROMs next to the sim so $readmemh finds them by name
Copy-Item (Join-Path $weights "layer1.hex") $here -Force
Copy-Item (Join-Path $weights "layer2.hex") $here -Force
Copy-Item (Join-Path $weights "layer3.hex") $here -Force

# 3. compile as SystemVerilog-2012 (needed for the unpacked array ports)
& "$bin\iverilog.exe" -g2012 -o mlp_sim.vvp `
    (Join-Path $rtl "mac_unit.v") `
    (Join-Path $rtl "dense_layer.v") `
    (Join-Path $rtl "mlp_top.v") `
    (Join-Path $here "tb_mlp.v")

# 4. run
& "$bin\vvp.exe" mlp_sim.vvp
