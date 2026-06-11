# Binary bridge between Python and the JTAG UART, over a localhost TCP socket.
#
# System Console runs this. It opens the bytestream service to the JTAG UART
# (the JTAG-Atlantic channel, binary clean) and listens on a TCP port. Python
# connects and streams 85-byte packets; for each one the bridge sends it to the
# board, waits for 0x55 + class, and writes the single class byte back over the
# socket. A TCP socket is used because System Console's Tcl crashes when reading
# raw binary from stdin, sockets are a separate, stable, binary-clean path.
#
# Run: system-console --script=jtag_bridge.tcl

set PORT 51000
set PACKET_BYTES 85
set START_FROM_FPGA 0x55

# open the byte stream to the JTAG UART
set path [lindex [get_service_paths bytestream] 0]
set stream [claim_service bytestream $path ""]
puts "BRIDGE: bytestream open on $path"
flush stdout

# read exactly n bytes from the board, polling until they arrive
proc recv_board {stream n} {
    set got {}
    while {[llength $got] < $n} {
        set chunk [bytestream_receive $stream [expr {$n - [llength $got]}]]
        if {[llength $chunk] > 0} {
            foreach b $chunk { lappend got $b }
        }
    }
    return $got
}

# handle one connected client: loop reading packets and returning the class
proc serve {sock} {
    global PACKET_BYTES START_FROM_FPGA stream
    fconfigure $sock -translation binary -blocking 1

    while {1} {
        set packet [read $sock $PACKET_BYTES]
        if {[eof $sock] || [string length $packet] < $PACKET_BYTES} break

        # send the packet to the board as a byte (int) list
        set out {}
        for {set i 0} {$i < $PACKET_BYTES} {incr i} {
            lappend out [scan [string index $packet $i] %c]
        }
        bytestream_send $stream $out

        # wait for the start byte, then read the class byte
        while {1} {
            set b [lindex [recv_board $stream 1] 0]
            if {$b == $START_FROM_FPGA} break
        }
        set class [lindex [recv_board $stream 1] 0]

        # return the raw class byte to Python
        puts -nonewline $sock [binary format c $class]
        flush $sock
    }
    close $sock
}

proc on_connect {sock addr port} {
    puts "BRIDGE: client connected"
    flush stdout
    serve $sock
}

socket -server on_connect $PORT
puts "BRIDGE: listening on port $PORT"
flush stdout

vwait forever
