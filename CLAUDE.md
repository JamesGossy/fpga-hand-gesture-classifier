# CLAUDE.md

This file tells Claude Code how to work in this repo. Read PLAN.md for the full project plan.

## Comments
- No AI-sounding language. Plain, direct English only.
- No em dashes anywhere (not in comments, not in strings, not in docs).
- Default: no comments. Only add one when the WHY is non-obvious — a hidden constraint, a subtle invariant, a workaround for a known bug.
- Strongly prefer inline comments (`// ...` at end of line) over standalone lines.
- File top: a short block comment is allowed (2-5 lines max). Describe what the file does and any key design constraint. Not a list of functions.
- Function top: one short line is allowed if the function is non-obvious. No multi-line docstrings.
- Logic blocks: one short standalone comment above a new logical stage is allowed (e.g. `// feedforward from curvature`). Keep these to one line.
- Never describe WHAT the code does. Only WHY it does something non-obvious.

## Section dividers
- Split long files into named sections with: `/* ---- section name ---- */`
- Use these in headers too (e.g. `/* ---- steering ---- */`, `/* ---- torque vectoring ---- */`).
- In long functions, number the main steps: `// 1. project onto line`, `// 2. Stanley feedback`, etc.

## C formatting
- clang-format, WebKit-based. Run `make format` before committing.
- Allman braces on functions, attached on control flow.
- 4-space indent, 100-col line limit.
- Do not hand-align operands inside expressions — clang-format will normalise it.

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