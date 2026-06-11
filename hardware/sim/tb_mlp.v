/* tb_mlp.v
   Drive every golden vector through mlp_top and check the class and all logits
   match the fixed point reference bit for bit. The golden data is loaded from
   the .hex files made by make_golden_mem.py. Any mismatch prints and fails. */

`timescale 1ns/1ps

module tb_mlp;

localparam NUM_INPUTS  = 42;
localparam NUM_CLASSES = 5;
localparam NUM_VECTORS = 40;

reg clk = 0;
always #5 clk = ~clk; // 100 MHz sim clock

reg start = 0;
reg signed [15:0] in_flat [0:NUM_INPUTS-1];
wire [$clog2(NUM_CLASSES)-1:0] class_out;
wire signed [15:0] logits [0:NUM_CLASSES-1];
wire done;

mlp_top #(
    .NUM_INPUTS(NUM_INPUTS), .HIDDEN(32), .NUM_CLASSES(NUM_CLASSES)
) dut (
    .clk(clk), .start(start), .in_flat(in_flat),
    .class_out(class_out), .logits(logits), .done(done)
);

// golden data
reg signed [15:0] g_inputs [0:NUM_VECTORS*NUM_INPUTS-1];
reg signed [15:0] g_class  [0:NUM_VECTORS-1];
reg signed [15:0] g_logits [0:NUM_VECTORS*NUM_CLASSES-1];

integer v, k;
integer errors = 0;

initial begin
    $readmemh("golden_inputs.hex", g_inputs);
    $readmemh("golden_class.hex",  g_class);
    $readmemh("golden_logits.hex", g_logits);

    for (v = 0; v < NUM_VECTORS; v = v + 1) begin
        // load this vector's inputs
        for (k = 0; k < NUM_INPUTS; k = k + 1)
            in_flat[k] = g_inputs[v*NUM_INPUTS + k];

        // pulse start for one cycle
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        // wait for the done pulse on a clock edge so a one-cycle pulse is not missed
        do @(posedge clk); while (done !== 1);
        @(posedge clk); // let class_out and logits settle

        // check class
        if (class_out !== g_class[v][$clog2(NUM_CLASSES)-1:0]) begin
            errors = errors + 1;
            $display("vector %0d: class mismatch, got %0d expected %0d",
                     v, class_out, g_class[v]);
        end

        // check every logit
        for (k = 0; k < NUM_CLASSES; k = k + 1) begin
            if (logits[k] !== g_logits[v*NUM_CLASSES + k]) begin
                errors = errors + 1;
                $display("vector %0d: logit %0d mismatch, got %0d expected %0d",
                         v, k, logits[k], g_logits[v*NUM_CLASSES + k]);
            end
        end
    end

    if (errors == 0)
        $display("PASS: all %0d golden vectors match bit for bit", NUM_VECTORS);
    else
        $display("FAIL: %0d mismatches", errors);

    $finish;
end

// safety timeout so a hang does not run forever. Each vector takes ~27us, 40
// vectors is ~1.1ms, so 3ms is comfortable headroom.
initial begin
    #3000000;
    $display("TIMEOUT");
    $finish;
end

endmodule
