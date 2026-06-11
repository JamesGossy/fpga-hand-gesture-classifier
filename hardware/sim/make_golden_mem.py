"""
Turn data/golden_vectors.json into flat hex memory files the Verilog testbench
loads with $readmemh. One word per line, 16-bit two's complement, lower case.

  golden_inputs.hex   40 vectors x 42 inputs, in order
  golden_class.hex    40 expected class numbers
  golden_logits.hex   40 vectors x 5 logits, in order

Run from hardware/sim: python make_golden_mem.py
"""

import json
import os

HERE = os.path.dirname(__file__)
GOLDEN = os.path.join(HERE, "..", "..", "data", "golden_vectors.json")


def hex_word(value):
    return format(value & 0xFFFF, "04x")


def main():
    data = json.load(open(GOLDEN))
    vectors = data["vectors"]

    with open(os.path.join(HERE, "golden_inputs.hex"), "w") as f:
        for v in vectors:
            for word in v["input_q4_12"]:
                f.write(hex_word(word) + "\n")

    with open(os.path.join(HERE, "golden_class.hex"), "w") as f:
        for v in vectors:
            f.write(hex_word(v["expected_class"]) + "\n")

    with open(os.path.join(HERE, "golden_logits.hex"), "w") as f:
        for v in vectors:
            for word in v["expected_logits_q4_12"]:
                f.write(hex_word(word) + "\n")

    # glue_packet.hex: vector 0 framed as the 85-byte wire packet (0xAA start
    # then 42 little-endian int16), one byte per line. tb_glue feeds this to the
    # mock JTAG UART.
    v0 = vectors[0]
    with open(os.path.join(HERE, "glue_packet.hex"), "w") as f:
        f.write("aa\n")  # start byte
        for word in v0["input_q4_12"]:
            low = word & 0xFF
            high = (word >> 8) & 0xFF
            f.write(format(low, "02x") + "\n")   # little-endian: low byte first
            f.write(format(high, "02x") + "\n")

    print("wrote", len(vectors), "vectors to golden_inputs/class/logits.hex")
    print("wrote glue_packet.hex (vector 0, 85 bytes)")


if __name__ == "__main__":
    main()
