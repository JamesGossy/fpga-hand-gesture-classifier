/* mac_unit.v
   Q4.12 multiply-accumulate. One product per clock into a 32-bit accumulator.
   Product of two Q4.12 numbers is Q8.24, which fits in 32 bits. The accumulator
   holds the running Q8.24 sum. clear loads the accumulator with a start value
   (the bias, already shifted to Q8.24); enable adds one more product. */

module mac_unit (
    input  wire               clk,
    input  wire               clear,      // load acc with init_value this cycle
    input  wire               enable,     // add a*b this cycle
    input  wire signed [31:0] init_value, // bias in Q8.24, used when clear is high
    input  wire signed [15:0] a,          // Q4.12
    input  wire signed [15:0] b,          // Q4.12
    output reg  signed [31:0] acc          // Q8.24 running sum
);

wire signed [31:0] product = a * b; // Q4.12 * Q4.12 = Q8.24

always @(posedge clk) begin
    if (clear)
        acc <= enable ? init_value + product : init_value;
    else if (enable)
        acc <= acc + product;
end

endmodule
