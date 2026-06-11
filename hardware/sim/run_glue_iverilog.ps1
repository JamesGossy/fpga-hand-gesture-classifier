# Compile and run the uart_mlp_glue testbench with Icarus.
#
#   & .\run_glue_iverilog.ps1

$ErrorActionPreference = "Stop"

$here    = $PSScriptRoot
$rtl     = Join-Path $here "..\rtl"
$weights = Join-Path $here "..\..\models\weights_q4_12"
# Icarus bin dir. Override with $env:IVERILOG_BIN if installed elsewhere.
$bin = if ($env:IVERILOG_BIN) { $env:IVERILOG_BIN } else { "C:\iverilog\bin" }

Set-Location $here

# regenerate glue_packet.hex (and the golden mem files) from the JSON
python make_golden_mem.py

# weight ROMs next to the sim so $readmemh finds them
Copy-Item (Join-Path $weights "layer1.hex") $here -Force
Copy-Item (Join-Path $weights "layer2.hex") $here -Force
Copy-Item (Join-Path $weights "layer3.hex") $here -Force

& "$bin\iverilog.exe" -g2012 -o glue_sim.vvp `
    (Join-Path $rtl "mac_unit.v") `
    (Join-Path $rtl "dense_layer.v") `
    (Join-Path $rtl "mlp_top.v") `
    (Join-Path $rtl "uart_mlp_glue.v") `
    (Join-Path $here "tb_glue.v")

& "$bin\vvp.exe" glue_sim.vvp
