/* dense_layer.v
   One fully connected layer in Q4.12, computed with a single MAC over many
   cycles. Weights and biases live in a ROM loaded from a .hex file by
   $readmemh. The ROM layout matches export_weights.flatten_layer: all weights
   first, neuron by neuron and input by input, then all biases in neuron order.

   For each output neuron the MAC starts from the bias (shifted up to Q8.24),
   adds one input*weight product per cycle, then the result is shifted right by
   (12 + OUTPUT_SHIFT), saturated to 16-bit signed, and optionally passed through
   ReLU. These steps mirror fixed_point.dense_layer exactly so the numbers match. */

module dense_layer #(
    parameter NUM_INPUTS   = 42,
    parameter NUM_NEURONS  = 32,
    parameter OUTPUT_SHIFT = 0,
    parameter USE_RELU     = 1,
    parameter MEM_FILE     = "layer1.hex"
) (
    input  wire        clk,
    input  wire        start,        // begin computing the whole layer
    input  wire signed [15:0] in_flat [0:NUM_INPUTS-1], // input vector, Q4.12
    output reg  signed [15:0] out_flat [0:NUM_NEURONS-1], // result, Q4.12
    output reg         done
);

localparam FRACTION_BITS = 12;
localparam TOTAL_SHIFT = FRACTION_BITS + OUTPUT_SHIFT;
localparam ROM_DEPTH = NUM_INPUTS * NUM_NEURONS + NUM_NEURONS;
localparam BIAS_BASE = NUM_INPUTS * NUM_NEURONS; // biases start here in the ROM

// weight/bias ROM, 16-bit Q4.12 words
reg signed [15:0] rom [0:ROM_DEPTH-1];
initial $readmemh(MEM_FILE, rom);

/* ---- MAC ---- */
// All MAC control is combinational off the FSM state so the operand, the clear,
// and the enable all line up in the same cycle. The acc register updates on the
// edge that ends that cycle.
reg               mac_clear;
reg               mac_enable;
reg signed [31:0] mac_init;
reg signed [15:0] mac_a;
reg signed [15:0] mac_b;
wire signed [31:0] mac_acc;

mac_unit mac (
    .clk        (clk),
    .clear      (mac_clear),
    .enable     (mac_enable),
    .init_value (mac_init),
    .a          (mac_a),
    .b          (mac_b),
    .acc        (mac_acc)
);

/* ---- control ---- */
// ACCUM streams one product per cycle, with the bias loaded on the first cycle.
// DRAIN gives the last product one edge to settle into acc before WRITE reads it.
localparam IDLE    = 3'd0;
localparam ACCUM   = 3'd1; // stream inputs through the MAC
localparam DRAIN   = 3'd2; // let the final product settle into acc
localparam WRITE   = 3'd3; // shift, saturate, relu, store

reg [2:0]  state = IDLE;
reg [$clog2(NUM_NEURONS):0] neuron;
reg [$clog2(NUM_INPUTS):0]  input_index;

// arithmetic shift back to Q4.12 with the per-layer rescale folded in
wire signed [31:0] shifted = mac_acc >>> TOTAL_SHIFT;

// saturate to 16-bit signed, then optional relu
function signed [15:0] finish_value(input signed [31:0] value);
    reg signed [15:0] clamped;
    begin
        if (value > 32767)
            clamped = 16'sd32767;
        else if (value < -32768)
            clamped = -16'sd32768;
        else
            clamped = value[15:0];

        if (USE_RELU && clamped < 0)
            finish_value = 16'sd0;
        else
            finish_value = clamped;
    end
endfunction

// All MAC control is combinational off state, so operand, clear, and enable
// share the same cycle. On the first ACCUM cycle (input_index==0) clear and
// enable are both high, so the edge loads bias + product[0]. After that enable
// alone accumulates. The DRAIN state is not needed because the acc holding the
// last product is the same edge that takes us out of ACCUM.
always @(*) begin
    mac_init   = $signed(rom[BIAS_BASE + neuron]) <<< FRACTION_BITS;
    mac_a      = in_flat[input_index];
    mac_b      = rom[neuron * NUM_INPUTS + input_index];
    mac_clear  = (state == ACCUM) && (input_index == 0);
    mac_enable = (state == ACCUM);
end

always @(posedge clk) begin
    done <= 0;

    case (state)

        IDLE: begin
            if (start) begin
                neuron      <= 0;
                input_index <= 0;
                state       <= ACCUM;
            end
        end

        ACCUM: begin
            if (input_index == NUM_INPUTS - 1)
                state <= DRAIN; // last product lands on this same edge
            else
                input_index <= input_index + 1;
        end

        // the edge leaving ACCUM committed the last product into acc, so by the
        // time we are in DRAIN the acc is complete and ready to read in WRITE
        DRAIN: begin
            state <= WRITE;
        end

        WRITE: begin
            out_flat[neuron] <= finish_value(shifted);
            if (neuron == NUM_NEURONS - 1) begin
                done  <= 1;
                state <= IDLE;
            end else begin
                neuron      <= neuron + 1;
                input_index <= 0;
                state       <= ACCUM;
            end
        end

    endcase
end

endmodule
