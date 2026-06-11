/* tb_glue.v
   Test uart_mlp_glue with a mock JTAG UART slave. The mock feeds the 85-byte
   golden packet one byte per data-register read (rvalid set), and captures the
   bytes the FSM writes back. We check the reply is 0x55 then the expected class.

   This tests the FSM logic: start-byte hunt, little-endian byte assembly, MLP
   handoff, and reply framing. The real JTAG UART timing may differ, but the
   read/write handshake (waitrequest, rvalid) is modelled here. */

`timescale 1ns/1ps

module tb_glue;

localparam NUM_INPUTS  = 42;
localparam NUM_CLASSES = 5;
localparam PACKET_BYTES = 85;
localparam EXPECTED_CLASS = 0;

reg clk = 0;
always #5 clk = ~clk;

reg reset_n = 0;

// Avalon master <-> mock slave
wire [0:0]  av_address;
wire        av_read;
wire        av_write;
wire [31:0] av_writedata;
reg  [31:0] av_readdata;
reg         av_waitrequest;

uart_mlp_glue #(.NUM_INPUTS(NUM_INPUTS), .NUM_CLASSES(NUM_CLASSES)) dut (
    .clk(clk), .reset_n(reset_n),
    .av_address(av_address), .av_read(av_read), .av_write(av_write),
    .av_writedata(av_writedata), .av_readdata(av_readdata),
    .av_waitrequest(av_waitrequest),
    .last_class()
);

/* ---- mock JTAG UART slave ---- */
reg [7:0] packet [0:PACKET_BYTES-1];
integer   rx_index = 0;       // next packet byte to hand to the FSM
reg [7:0] sent [0:7];         // bytes the FSM wrote back
integer   tx_count = 0;

initial $readmemh("glue_packet.hex", packet);

// the mock never stalls: waitrequest low, data ready immediately
always @(*) begin
    av_waitrequest = 1'b0;
    if (rx_index < PACKET_BYTES)
        av_readdata = {16'b0, 1'b1, 7'b0, packet[rx_index]}; // rvalid=1, byte
    else
        av_readdata = 32'b0; // rvalid=0, nothing left
end

// advance the rx queue on an accepted read, capture writes
always @(posedge clk) begin
    if (av_read && !av_waitrequest && rx_index < PACKET_BYTES)
        rx_index <= rx_index + 1;
    if (av_write && !av_waitrequest) begin
        sent[tx_count] <= av_writedata[7:0];
        tx_count       <= tx_count + 1;
    end
end

/* ---- check ---- */
integer errors = 0;

initial begin
    reset_n = 0;
    repeat (4) @(posedge clk);
    reset_n = 1;

    // wait for the FSM to send both reply bytes
    wait (tx_count == 2);
    @(posedge clk);

    if (sent[0] !== 8'h55) begin
        errors = errors + 1;
        $display("reply header wrong: got %02x expected 55", sent[0]);
    end
    if (sent[1] !== EXPECTED_CLASS) begin
        errors = errors + 1;
        $display("class wrong: got %0d expected %0d", sent[1], EXPECTED_CLASS);
    end

    if (errors == 0)
        $display("PASS: glue returned 0x55 and class %0d", EXPECTED_CLASS);
    else
        $display("FAIL: %0d errors", errors);

    $finish;
end

initial begin
    #3000000;
    $display("TIMEOUT (tx_count=%0d)", tx_count);
    $finish;
end

endmodule
