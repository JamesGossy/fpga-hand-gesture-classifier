/* mlp_top.v
   The whole network: 42 -> 32 -> 32 -> 5, Q4.12. Three dense_layer instances run
   one after another, each started when the previous one finishes. The output
   shifts [0, 1, 2] and the ReLU flags match export_weights.py. When the last
   layer is done, argmax picks the class. Ties go to the lowest index, same as
   fixed_point.argmax. */

module mlp_top #(
    parameter NUM_INPUTS  = 42,
    parameter HIDDEN      = 32,
    parameter NUM_CLASSES = 5
) (
    input  wire        clk,
    input  wire        start,
    input  wire signed [15:0] in_flat [0:NUM_INPUTS-1],
    output reg  [$clog2(NUM_CLASSES)-1:0] class_out,
    output wire signed [15:0] logits [0:NUM_CLASSES-1],
    output reg         done
);

/* ---- layer wiring ---- */
wire signed [15:0] h1 [0:HIDDEN-1];
wire signed [15:0] h2 [0:HIDDEN-1];
wire done1, done2, done3;

reg start1, start2, start3;

dense_layer #(
    .NUM_INPUTS(NUM_INPUTS), .NUM_NEURONS(HIDDEN),
    .OUTPUT_SHIFT(0), .USE_RELU(1), .MEM_FILE("layer1.hex")
) layer1 (
    .clk(clk), .start(start1), .in_flat(in_flat), .out_flat(h1), .done(done1)
);

dense_layer #(
    .NUM_INPUTS(HIDDEN), .NUM_NEURONS(HIDDEN),
    .OUTPUT_SHIFT(1), .USE_RELU(1), .MEM_FILE("layer2.hex")
) layer2 (
    .clk(clk), .start(start2), .in_flat(h1), .out_flat(h2), .done(done2)
);

dense_layer #(
    .NUM_INPUTS(HIDDEN), .NUM_NEURONS(NUM_CLASSES),
    .OUTPUT_SHIFT(2), .USE_RELU(0), .MEM_FILE("layer3.hex")
) layer3 (
    .clk(clk), .start(start3), .in_flat(h2), .out_flat(logits), .done(done3)
);

/* ---- sequencing ---- */
localparam IDLE = 3'd0;
localparam RUN1 = 3'd1;
localparam RUN2 = 3'd2;
localparam RUN3 = 3'd3;
localparam PICK = 3'd4;

reg [2:0] state = IDLE;

always @(posedge clk) begin
    start1 <= 0;
    start2 <= 0;
    start3 <= 0;
    done   <= 0;

    case (state)
        IDLE: begin
            if (start) begin
                start1 <= 1;
                state  <= RUN1;
            end
        end
        RUN1: if (done1) begin start2 <= 1; state <= RUN2; end
        RUN2: if (done2) begin start3 <= 1; state <= RUN3; end
        RUN3: if (done3) state <= PICK;
        PICK: begin
            done  <= 1;
            state <= IDLE;
        end
    endcase
end

/* ---- argmax over the logits ---- */
integer i;
reg signed [15:0] best_value;
reg [$clog2(NUM_CLASSES)-1:0] best_index;

always @(*) begin
    best_value = logits[0];
    best_index = 0;
    for (i = 1; i < NUM_CLASSES; i = i + 1) begin
        if (logits[i] > best_value) begin
            best_value = logits[i];
            best_index = i[$clog2(NUM_CLASSES)-1:0];
        end
    end
end

always @(posedge clk) begin
    if (state == RUN3 && done3)
        class_out <= best_index;
end

endmodule
