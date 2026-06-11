/* uart_mlp_glue.v
   Avalon-MM master that drives the JTAG UART and runs the MLP on its own.

   Protocol (matches PLAN.md and serial_link.py):
     Host -> FPGA:  0xAA  [42 x int16 little-endian = 84 bytes]
     FPGA -> Host:  0x55  [class byte]

   The JTAG UART has two 32-bit Avalon registers:
     offset 0 (data):    write low 8 bits to send. read: [7:0]=rx byte,
                         [15]=rvalid, [31:16]=rx bytes available.
     offset 1 (control): [31:16]=tx free space (wspace).

   The FSM hunts for the start byte, gathers the payload into 42 Q4.12 words,
   runs mlp_top, then sends the start byte and class back. Bytes pair up
   little-endian: first byte is the low half of the word, second is the high. */

module uart_mlp_glue #(
    parameter NUM_INPUTS  = 42,
    parameter NUM_CLASSES = 5
) (
    input  wire        clk,
    input  wire        reset_n,

    // Avalon-MM master to the JTAG UART slave
    output reg  [0:0]  av_address,
    output reg         av_read,
    output reg         av_write,
    output reg  [31:0] av_writedata,
    input  wire [31:0] av_readdata,
    input  wire        av_waitrequest,

    // last class, held after each inference for a visible check
    output reg  [$clog2(NUM_CLASSES)-1:0] last_class
);

localparam START_TO_FPGA   = 8'hAA;
localparam START_FROM_FPGA = 8'h55;
localparam PAYLOAD_BYTES    = NUM_INPUTS * 2; // two bytes per int16

/* ---- MLP ---- */
reg                mlp_start;
reg  signed [15:0] in_flat [0:NUM_INPUTS-1];
wire [$clog2(NUM_CLASSES)-1:0] class_out;
wire signed [15:0] logits [0:NUM_CLASSES-1];
wire               mlp_done;

mlp_top #(
    .NUM_INPUTS(NUM_INPUTS), .HIDDEN(32), .NUM_CLASSES(NUM_CLASSES)
) mlp (
    .clk(clk), .start(mlp_start), .in_flat(in_flat),
    .class_out(class_out), .logits(logits), .done(mlp_done)
);

/* ---- state ---- */
localparam HUNT       = 4'd0;  // read data reg, wait for the start byte
localparam HUNT_WAIT  = 4'd1;
localparam RECV       = 4'd2;  // read one payload byte
localparam RECV_WAIT  = 4'd3;
localparam RUN        = 4'd4;  // pulse the MLP
localparam RUN_WAIT   = 4'd5;  // wait for done
localparam SEND_HDR   = 4'd6;  // write 0x55
localparam SEND_HDR_W = 4'd7;
localparam SEND_CLS   = 4'd8;  // write the class byte
localparam SEND_CLS_W = 4'd9;

reg [3:0] state = HUNT;
reg [$clog2(PAYLOAD_BYTES):0] byte_count;
reg [7:0] low_byte; // first byte of an int16 held until the high byte arrives

// the data register read fields
wire [7:0]  rx_byte  = av_readdata[7:0];
wire        rx_valid = av_readdata[15];

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        state      <= HUNT;
        av_address <= 0;
        av_read    <= 0;
        av_write   <= 0;
        mlp_start  <= 0;
        byte_count <= 0;
        last_class <= 0;
    end else begin
        av_read   <= 0;
        av_write  <= 0;
        mlp_start <= 0;

        case (state)

            // start a read of the data register
            HUNT: begin
                av_address <= 1'b0;
                av_read    <= 1'b1;
                state      <= HUNT_WAIT;
            end

            // hold read until the slave accepts it, then check for start byte
            HUNT_WAIT: begin
                if (av_waitrequest) begin
                    av_read <= 1'b1; // keep asserting until accepted
                end else if (rx_valid && rx_byte == START_TO_FPGA) begin
                    byte_count <= 0;
                    state      <= RECV;
                end else begin
                    state <= HUNT; // no start byte yet, poll again
                end
            end

            RECV: begin
                av_address <= 1'b0;
                av_read    <= 1'b1;
                state      <= RECV_WAIT;
            end

            RECV_WAIT: begin
                if (av_waitrequest) begin
                    av_read <= 1'b1;
                end else if (rx_valid) begin
                    // even byte is the low half, odd byte completes the word
                    if (byte_count[0] == 1'b0) begin
                        low_byte <= rx_byte;
                    end else begin
                        in_flat[byte_count >> 1] <= $signed({rx_byte, low_byte});
                    end

                    if (byte_count == PAYLOAD_BYTES - 1)
                        state <= RUN;
                    else begin
                        byte_count <= byte_count + 1;
                        state      <= RECV;
                    end
                end else begin
                    state <= RECV; // byte not ready yet, read again
                end
            end

            RUN: begin
                mlp_start <= 1'b1;
                state     <= RUN_WAIT;
            end

            RUN_WAIT: begin
                if (mlp_done) begin
                    last_class <= class_out; // hold the result for the LEDs
                    state      <= SEND_HDR;
                end
            end

            // write the reply start byte
            SEND_HDR: begin
                av_address   <= 1'b0;
                av_writedata <= {24'b0, START_FROM_FPGA};
                av_write     <= 1'b1;
                state        <= SEND_HDR_W;
            end

            SEND_HDR_W: begin
                if (av_waitrequest) begin
                    av_write <= 1'b1;
                end else begin
                    state <= SEND_CLS;
                end
            end

            // write the class byte
            SEND_CLS: begin
                av_address   <= 1'b0;
                av_writedata <= {29'b0, class_out};
                av_write     <= 1'b1;
                state        <= SEND_CLS_W;
            end

            SEND_CLS_W: begin
                if (av_waitrequest) begin
                    av_write <= 1'b1;
                end else begin
                    state <= HUNT; // ready for the next packet
                end
            end

        endcase
    end
end

endmodule
