# CLAUDE.md

This file tells Claude Code how to work in this repo. Read PLAN.md for the full project plan.

## What this project is

A hand gesture classifier. The laptop reads the webcam, finds hand landmarks with
MediaPipe, turns them into a small list of numbers, and sends them over UART to a
DE10-Lite FPGA. The FPGA runs a small fixed point neural net and sends back the
gesture. Build it all in Python first, then move the math to Verilog.

## Writing style for all code

Keep everything simple. Write code the way you would explain it to a person out loud.

- Use plain words for names. Say `hand_box_size`, not `hbSz`. Say `read_one_byte`, not `rdB`.
- Keep functions short and doing one thing.
- No clever one liners. If a line is hard to read, split it up.
- Do not use em-dashes anywhere, in code or comments or docs. Use a comma or a full stop.
- Avoid hard grammar. Short sentences. One idea per sentence.

## Comments

- Only comment when the reason for the code is not obvious.
- All functions should have a short comment at the start explaining what they do.
- 1 line comments that break-up long stretches of code are encouraged.
- A comment should say why, not repeat what the line already shows.
- Keep comments short and plain. No long blocks. No fancy words.
- Do not comment every line. Most lines need no comment.

Good comment:
  # wrist point is the origin so the gesture works anywhere on screen
Bad comment:
  # subtract wrist x from landmark x and wrist y from landmark y for each point

## How to work

- Match the file layout in PLAN.md. Do not add files that are not in the plan unless asked.
- Python first. Get the float model working, then the fixed point version, then the Verilog.
- The fixed point Python code and the Verilog must give the same numbers. Use the golden
  test vectors to check this. Do not change one side without checking the other.
- Test the UART as a plain byte echo on hardware before adding any math.
- Ask before adding a new library. Keep the dependency list small.

## Fixed point rules

- Format is Q4.12, signed, 16 bits.
- Multiply two numbers, add them up in 32 bits, then shift right by 12 to scale back.
- Stop values from going past the limit. Use ReLU as max of 0 and the value.
- Do the exact same steps in Python and in Verilog so the results match.

## Do not

- Do not use em-dashes.
- Do not write long or fancy comments.
- Do not add tools or libraries without asking.
- Do not push raw camera images to the FPGA. Send landmarks only.