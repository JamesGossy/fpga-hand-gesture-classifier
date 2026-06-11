"""
Integer-only forward pass. This is the reference the FPGA must match bit for bit.

Format is Q4.12, signed, 16 bits. One sign bit, three integer bits, twelve
fraction bits. The value range is about -8.0 to just under +8.0. The smallest
step is 1/4096.

The math is plain integers on purpose. Multiply two Q4.12 numbers to get a
Q8.24 result in 32 bits, accumulate in 32 bits, then shift right by 12 to get
back to Q4.12. Saturate to the 16-bit signed range. ReLU is max(0, x). Do the
same steps here and in Verilog so the numbers agree.
"""

# Q4.12 constants.
FRACTION_BITS = 12
SCALE = 1 << FRACTION_BITS  # 4096

# 16-bit signed range.
INT16_MIN = -32768
INT16_MAX = 32767


def to_fixed(value):
    """Turn a float into a Q4.12 int. Rounds to nearest, then saturates."""
    raw = int(round(value * SCALE))
    return saturate(raw)


def to_float(fixed):
    """Turn a Q4.12 int back into a float. For checking only."""
    return fixed / SCALE


def saturate(value):
    """Clamp an int to the 16-bit signed range."""
    if value > INT16_MAX:
        return INT16_MAX
    if value < INT16_MIN:
        return INT16_MIN
    return value


def relu(value):
    """Max of 0 and the value."""
    if value < 0:
        return 0
    return value


def dense_layer(inputs, weights, biases, use_relu, output_shift):
    """
    One fully connected layer in fixed point.

    inputs is a list of Q4.12 ints. weights is a list of rows, one row per
    output neuron, each row holding one Q4.12 weight per input. biases is one
    Q4.12 int per output neuron. Returns a list of Q4.12 ints.

    The accumulator stays in 32 bits the whole time. We add the bias shifted up
    by 12 first so the bias lines up with the Q8.24 products, then shift the
    whole sum down to get back to Q4.12.

    output_shift is an extra right shift past the normal 12. The activations in
    this net grow past the Q4.12 range of +-8 between layers, so each layer
    divides its result by a power of two to keep it in range. The shift is a
    plain wire in Verilog, so this stays cheap and bit exact on both sides.
    """
    outputs = []
    for neuron_index, weight_row in enumerate(weights):
        # bias starts the sum, shifted to Q8.24 to match the products
        accumulator = biases[neuron_index] << FRACTION_BITS

        for input_value, weight in zip(inputs, weight_row):
            accumulator += input_value * weight  # Q4.12 * Q4.12 -> Q8.24

        # back to Q4.12 with the extra rescale folded into the same shift,
        # then clamp to 16 bits like the hardware register. >> floors toward
        # negative infinity, which matches an arithmetic shift in Verilog.
        scaled = accumulator >> (FRACTION_BITS + output_shift)
        scaled = saturate(scaled)

        if use_relu:
            scaled = relu(scaled)

        outputs.append(scaled)

    return outputs


def forward(inputs, layers):
    """
    Run the whole network in fixed point.

    inputs is a list of Q4.12 ints. layers is a list of
    (weights, biases, use_relu, output_shift) tuples in order. Returns the final
    logits as Q4.12 ints.
    """
    values = inputs
    for weights, biases, use_relu, output_shift in layers:
        values = dense_layer(values, weights, biases, use_relu, output_shift)
    return values


def argmax(values):
    """Index of the largest value. Ties go to the lowest index."""
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values):
        if value > best_value:
            best_value = value
            best_index = index
    return best_index


def classify(inputs, layers):
    """Run the network and return (class_number, logits)."""
    logits = forward(inputs, layers)
    return argmax(logits), logits
