# Project Plan

A hand gesture classifier. The laptop reads the webcam, finds hand landmarks with
MediaPipe, turns them into a small list of numbers, and sends them over UART to a
DE10-Lite FPGA. The FPGA runs a small fixed-point neural net and sends back the
gesture class.

## Architecture

The DE10-Lite has no camera port, no ARM core, and no Linux. The laptop does all
the image work. The FPGA only does fixed-point inference on a small feature vector.

```
Laptop (Python)                                         DE10-Lite (Verilog)
-----------------------------------------------------------------------
webcam -> MediaPipe hand landmarks (21 pts)
       -> 42-float vector
       -> normalize
       -> quantize to int16
       -> send bytes over UART  ------>  UART RX -> fixed-point MLP -> argmax
                                                  -> predicted class
                                <------  class byte -> LEDs / 7-seg / UART TX
```

Why landmarks instead of raw pixels: MAX 10 has ~50K LEs, 144 multipliers, ~180KB
block RAM. A CNN on raw frames will not fit comfortably. MediaPipe gives 21 (x,y)
landmarks = 42 inputs. A 2-layer MLP (42 -> 32 -> N) classifies them with a few
thousand MACs, which is easy to quantize and implement.

## Gesture set

Five static gestures, in class order:

| Class | Name       |
|-------|------------|
| 0     | fist       |
| 1     | open_palm  |
| 2     | peace      |
| 3     | thumbs_up  |
| 4     | point      |

The order here is fixed. It matches the GESTURES list in software/preprocess.py
and will match the class number the FPGA sends back. Do not change the order
after training.

## Repo structure

```
fpga-hand-gesture-classifier/
├── PLAN.md                       this file
├── CLAUDE.md                     instructions for Claude Code
├── README.md
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml             CI: runs unit tests on push / PR
├── software/                     everything that runs on the laptop
│   ├── requirements.txt          opencv-python, mediapipe, numpy, torch, pyserial
│   ├── preprocess.py             normalize landmarks, shared GESTURES list
│   ├── capture.py                webcam + MediaPipe -> labeled CSVs in data/raw/
│   ├── train.py                  PyTorch MLP training, saves models/model_float.pt
│   ├── fixed_point.py            integer-only forward pass (FPGA reference)
│   ├── demo.py                   live float demo + UART sender (UART added Phase 3)
│   ├── serial_link.py            UART framing and comms with the board
│   └── export_weights.py         model_float.pt -> fixed-point weights in models/
├── data/
│   ├── raw/                      recorded landmark CSVs (one file per gesture)
│   └── golden_vectors.json       input ints -> expected class and logits
├── models/
│   ├── model_float.pt            trained float model
│   └── weights_q4_12/            exported .mif / .hex weight files
├── hardware/                     self-contained FPGA project
│   ├── rtl/                      Verilog source
│   │   ├── top.v                 board top-level (pins, LEDs, 7-seg, UART)
│   │   ├── uart_rx.v
│   │   ├── uart_tx.v
│   │   ├── mac_unit.v
│   │   ├── dense_layer.v
│   │   ├── mlp_top.v
│   │   └── de10_lite.qsf         Quartus pin assignments
│   ├── sim/                      testbenches
│   │   ├── tb_uart.v
│   │   └── tb_mlp.v              drives golden vectors, checks outputs
│   └── quartus/                  Quartus project files
└── tests/                        unit tests (pytest, run in CI)
    ├── test_preprocess.py
    └── test_model.py
```

## Phase plan

The rule throughout: the FPGA must reproduce the exact numbers the Python
fixed-point model produces. Get bit-exact agreement in Python first, then the
FPGA is a translation job.

---

### Phase 0 - Repo and tooling (done)

- [x] Directory structure
- [x] .gitignore
- [x] requirements.txt
- [x] GitHub Actions CI (runs unit tests on push and PR)
- [x] Install Quartus Prime Lite (manual, free from Intel)
- [ ] Install Python env: `python -m venv .venv` then `pip install -r software/requirements.txt`

---

### Phase 1 - Python data and model (done)

- [x] preprocess.py: normalize landmarks to be translation and scale invariant
- [x] capture.py: open webcam, run MediaPipe, save labeled CSVs to data/raw/
- [x] train.py: 42->32->32->5 MLP in PyTorch, saves model_float.pt
- [x] demo.py: live float demo, ground-truth reference, UART hook left for Phase 3
- [x] Unit tests for preprocess and model (10 tests, all passing)

To run Phase 1:

```
pip install -r software/requirements.txt
python software/capture.py    # record 300-500 samples per gesture
python software/train.py      # aim for >95% val accuracy
python software/demo.py       # watch it predict live
```

Capture tips:
- Press 1 to 5 to pick the gesture, space to save, q to quit.
- Vary hand position, distance, and lighting so the model sees variety.
- Aim for 300 to 500 samples per gesture.

---

### Phase 2 - Quantization and fixed-point reference (done)

- [x] software/fixed_point.py: integer-only forward pass, Q4.12 16-bit signed.
- [x] software/export_weights.py: model_float.pt -> .mif/.hex in models/weights_q4_12/.
- [x] data/golden_vectors.json: 40 real inputs with expected class and logits.
- [x] Fixed point accuracy matches float exactly (0.9945 on the full set).
- [x] Unit tests for the fixed point math and golden vector reproduction.

Note on dynamic range: plain Q4.12 alone dropped accuracy to 0.93 because the
hidden activations grow past the +-8 range (h2 reached ~14, logits ~49) and
saturated. The fix is a per-layer power-of-two right shift after each layer,
folded into the same accumulator shift. Shifts are [0, 1, 2] for the three
layers. This recovers full float accuracy and is a plain wire in Verilog. The
shifts are recorded in golden_vectors.json so the FPGA uses the same values.

1. Implement software/fixed_point.py: an integer-only forward pass that mirrors
   exactly what the FPGA will do. Format is Q4.12, 16-bit signed.
   - Multiply two Q4.12 values, accumulate in 32 bits, shift right by 12.
   - Saturate on overflow. ReLU is max(0, x).
2. Verify quantized accuracy stays close to float (a small drop is fine).
3. Implement tools/export_weights.py: model_float.pt -> fixed-point weights
   as .mif/.hex files in models/weights_q4_12/.
4. Save data/golden_vectors.json: many (input_ints -> expected_class, logits)
   pairs. The FPGA must match these exactly.

Fixed-point rules (Q4.12, 16-bit signed):
- Multiply two Q4.12 values -> Q8.24 (32 bits). Accumulate in 32 bits. Then
  shift right by 12 to return to Q4.12.
- Saturate on overflow: clamp to the 16-bit signed range.
- ReLU: max(0, x).
- Do the identical steps in fixed_point.py and in Verilog so results agree
  bit for bit.

---

### Phase 3 - UART protocol and loopback (todo)

1. Define the framed protocol (see below). Implement software/serial_link.py.
2. Build uart_rx.v and uart_tx.v. Test a byte echo loopback on real hardware
   before adding any math. This proves the wiring, baud rate, and framing.
3. Wire the UART sender into demo.py at the marked spot.

UART protocol:

```
Host -> FPGA:  0xAA  [42 x int16 little-endian = 84 bytes]  (85 bytes total)
FPGA -> Host:  0x55  [class: 1 byte]  [optional: N x int16 logits]
```

Start byte for resync, fixed-length payload so the FPGA state machine is trivial.
Baud rate: 115200. Use the GPIO header pins with a USB-serial adapter, not the
on-board USB (that is JTAG only).

---

### Phase 4 - FPGA inference core (todo)

1. hardware/rtl/mac_unit.v: fixed-point multiply-accumulate.
2. hardware/rtl/dense_layer.v: one fully-connected layer, weights from ROM
   (block RAM), ReLU.
3. hardware/rtl/mlp_top.v: chain the layers, argmax over output logits.
4. hardware/rtl/top.v: UART RX -> input register -> MLP -> class -> UART TX
   and LEDs/7-seg.
5. Verify against golden vectors in simulation (ModelSim or Verilator/cocotb),
   then on hardware.

---

### Phase 5 - Integration and polish (todo)

- Full loop: webcam -> laptop -> UART -> FPGA -> class -> back to laptop overlay.
- Measure latency and accuracy on hardware vs the Python float model.
- README with demo GIF, results table, and the block diagram.

---

## Tooling

- Python: opencv-python, mediapipe, numpy, torch, pyserial
- FPGA: Intel Quartus Prime Lite (free) for MAX 10
- Simulation: ModelSim-Intel or Verilator + cocotb
- DE10-Lite device part: 10M50DAF484C7G
- Tests: pytest (run with `pytest tests/ -v`)

## Stretch goals

- Dynamic gestures (swipe/wave) via a small sequence model.
- Pipeline the dense layers for higher throughput.
- Show confidence on the 7-seg display.
