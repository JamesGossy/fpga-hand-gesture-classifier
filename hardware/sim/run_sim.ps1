# Build the golden memory files, copy the weight ROMs in, compile the RTL and
# testbench with Questa, and run the sim. Run from anywhere.
#
#   powershell -ExecutionPolicy Bypass -File run_sim.ps1

$ErrorActionPreference = "Stop"

# Questa bin dir. Override with $env:QUESTA_BIN if installed elsewhere.
$questa = if ($env:QUESTA_BIN) { $env:QUESTA_BIN } else { "C:\altera_lite\25.1std\questa_fse\win64" }
$here   = $PSScriptRoot
$rtl    = Join-Path $here "..\rtl"
$weights = Join-Path $here "..\..\models\weights_q4_12"

Set-Location $here

# 1. regenerate golden memory files from the JSON
python make_golden_mem.py

# 2. bring the weight ROMs next to the sim so $readmemh finds them by name
Copy-Item (Join-Path $weights "layer1.hex") $here -Force
Copy-Item (Join-Path $weights "layer2.hex") $here -Force
Copy-Item (Join-Path $weights "layer3.hex") $here -Force

# 3. fresh work library
if (Test-Path work) { Remove-Item work -Recurse -Force }
& "$questa\vlib.exe" work

# 4. compile as SystemVerilog (the array ports need -sv)
& "$questa\vlog.exe" -sv `
    (Join-Path $rtl "mac_unit.v") `
    (Join-Path $rtl "dense_layer.v") `
    (Join-Path $rtl "mlp_top.v") `
    (Join-Path $here "tb_mlp.v")

# 5. run headless, print transcript
& "$questa\vsim.exe" -c -do "run -all; quit" work.tb_mlp
