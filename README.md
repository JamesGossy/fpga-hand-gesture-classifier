# FPGA Hand Gesture Classifier

A hand gesture classifier that runs its neural network on an FPGA. The laptop
reads the webcam, finds hand landmarks with MediaPipe, turns them into a small
list of numbers, and sends them to a DE10-Lite (Intel MAX 10) over the on-board
USB-Blaster. The FPGA runs a fixed-point neural net in hardware and sends back
the gesture. The same math is built three times, float in PyTorch, fixed-point
in Python, and Verilog, and the three are checked to agree.

![live demo](docs/demo.gif)

The overlay shows the laptop float prediction and the `fpga:` class returned by
the board over the JTAG link.

## What it does

```
Laptop (Python)                                         DE10-Lite (Verilog)
-----------------------------------------------------------------------
webcam -> MediaPipe hand landmarks (21 points)
       -> 42-float vector
       -> normalize
       -> quantize to Q4.12 int16
       -> send 85-byte packet  --------->  JTAG UART -> FSM -> fixed-point MLP
                                                              -> argmax
                               <---------  class byte <- 0x55 + class -> LEDs
```

Five gestures: fist, open palm, peace, thumbs up, point.

## Results

### Accuracy (2,348 captured samples)

| Model | Accuracy |
|-------|----------|
| Float (PyTorch) | 98.3% |
| Fixed-point Q4.12 (Python) | 97.87% |
| FPGA (Verilog) | 97.87% (bit-exact to fixed-point) |

Quantizing to 16-bit fixed point costs only 0.43% accuracy. The FPGA reproduces
the Python fixed-point result bit for bit, verified on 40 golden vectors in
simulation and confirmed on hardware, so its accuracy equals the fixed-point row
by construction.

### FPGA resource usage (MAX 10 10M50DA)

| Resource | Used | Available | Percent |
|----------|------|-----------|---------|
| Logic elements | 6,459 | 49,760 | 13% |
| Logic registers | 2,078 | 49,760 | 4% |
| Embedded 9-bit multipliers | 6 | 288 | 2% |
| Memory bits | 1,024 | 1,677,312 | < 1% |

The whole classifier fits in a small corner of the device. It uses one
multiply-accumulate unit reused across all neurons, which is why only 6
multipliers are needed.

### Latency

| Stage | Time |
|-------|------|
| MLP inference (Verilog, in simulation) | ~27 us |
| Full round trip over USB-Blaster JTAG UART | ~208 ms median |
| Throughput | ~4.8 classifications/sec |

The inference itself takes microseconds. The round trip is dominated by the
USB-Blaster JTAG UART link, which has a fixed polling latency around 200 ms. The
JTAG UART is a debug channel, not a fast data link, so it bounds the live demo
rate, not the hardware. A dedicated UART on the GPIO pins would remove this
bottleneck.

## Why run this on an FPGA, and what I found

The honest answer from this project is nuanced, so here is the real data.

### Inference time, the same network in three places

| Where | Time per classification |
|-------|-------------------------|
| FPGA MLP core (Verilog, simulated) | ~27 us |
| Laptop, PyTorch float | ~38 us |
| Laptop, pure-Python fixed-point | ~180 us |

The FPGA compute is genuinely the fastest of the three. It beats the Python
fixed-point loop by about 7x and edges out optimized PyTorch float, while using
6 multipliers and 13% of a small, cheap device.

### The catch: the link, not the compute, is the bottleneck

| Stage | Time |
|-------|------|
| FPGA inference | ~27 us |
| Full round trip over USB-Blaster JTAG UART | ~208 ms |

End to end, the laptop is far faster in wall-clock time, not because its math is
faster, but because the FPGA's math is hidden behind a slow debug cable. The
USB-Blaster JTAG UART has a fixed polling latency near 200 ms. That single link
dwarfs the 27 us of actual work by a factor of ~7,000.

### What this actually teaches

- **The compute case for an FPGA is real.** A few microseconds, deterministic,
  in a fraction of a tiny device. No OS, no scheduler jitter, no Python overhead.
  For a fixed network this is hard, predictable silicon.
- **The win only matters if the I/O can feed it.** Here the bottleneck is the
  data path, not the arithmetic. A real deployment would use a proper UART, SPI,
  or a sensor wired straight into the fabric, where data arrives in microseconds
  and the FPGA's speed and determinism pay off.
- **For a webcam on a laptop, the laptop is the right tool.** The data already
  lives on the laptop, MediaPipe runs there, and the network is tiny. Shipping
  landmarks to an FPGA and back over JTAG adds latency for no gain. The FPGA
  version is a hardware-design exercise and a building block for a standalone
  device, not a speed-up for this exact setup.

The point of the project was to build the network in real fixed-point hardware
and prove it matches the software bit for bit. It does. The latency finding is
part of the result, not a failure: it shows precisely where a hardware design
lives or dies, the interface, not the multiplier.

## The network

42 inputs -> 32 (ReLU) -> 32 (ReLU) -> 5 outputs, argmax over the 5 logits.

Fixed point is Q4.12, signed, 16 bits. Products are computed in 32 bits as
Q8.24, accumulated, then shifted back to Q4.12 and saturated. Each layer applies
an extra right shift (0, 1, 2) to keep activations in range. The Python and
Verilog do the exact same steps so the numbers match.

## How the pieces fit

### Software (`software/`)

| File | Role |
|------|------|
| `capture.py` | record landmark samples from the webcam |
| `preprocess.py` | normalize landmarks to the 42-number feature vector |
| `train.py` | train the float MLP, save `model_float.pt` |
| `fixed_point.py` | integer-only forward pass, the bit-exact reference |
| `export_weights.py` | quantize weights to Q4.12, write ROM files and golden vectors |
| `serial_link.py` | frame packets and talk to the board over the JTAG link |
| `demo.py` | live webcam demo with float and FPGA predictions side by side |

### Hardware (`hardware/rtl/`)

| File | Role |
|------|------|
| `mac_unit.v` | Q4.12 multiply-accumulate |
| `dense_layer.v` | one fully connected layer, one MAC, weights from a ROM |
| `mlp_top.v` | chains the three layers, argmax over the logits |
| `uart_mlp_glue.v` | reads the packet, runs the MLP, sends the class back |
| `DE10_LITE_Golden_Top.v` | board top level, wires the glue to the JTAG UART |

The JTAG UART core is built in Platform Designer (`hardware/quartusD/jtag_comms`).

## The PC-to-FPGA link

The DE10-Lite has no built-in USB-serial bridge, the USB-Blaster is JTAG only.
To send data without extra hardware, the design uses an on-chip JTAG UART. Two
ways to drive it from the PC did not work:

- `juart-terminal` mangles raw binary (it is a text terminal).
- Piping binary to System Console's stdin crashes its Tcl interpreter on Windows.

The working approach: a small Tcl bridge (`hardware/quartusD/jtag_bridge.tcl`)
runs inside System Console, opens the JTAG UART byte stream, and listens on a
local TCP socket. Python connects to that socket and streams packets. TCP sockets
in Tcl are binary clean and avoid both failure modes.

## Running it

### Software setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r software/requirements.txt
```

### Train and export (optional, a trained model is included)

```
python software/capture.py        # record gestures
python software/train.py          # train the float model
python software/export_weights.py # quantize, write ROMs and golden vectors
```

### Build and program the FPGA

1. Open `hardware/quartusD/gesture_classifier.qpf` in Quartus Prime Lite.
2. Compile (or headless: `quartus_sh --flow compile gesture_classifier`).
3. Program `output_files/gesture_classifier.sof` over the USB-Blaster.

### Run the live demo

Close any open `juart-terminal` first (only one tool can hold the JTAG UART).

```
python software/demo.py
```

The overlay shows the float prediction and `fpga:<gesture>` from the board. The
low LEDs (LEDR[2:0]) also show the class number.

## Verifying the hardware matches the model

The Verilog is checked against the Python fixed-point reference, not trusted.

```
cd hardware/sim
.\run_sim_iverilog.ps1   # MLP core vs 40 golden vectors, bit for bit
.\run_glue_iverilog.ps1  # full receive -> classify -> reply path
```

Both must report PASS. The golden vectors come from `export_weights.py`, so the
software and hardware can never silently drift apart.
