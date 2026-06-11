"""
UART framing and comms with the board.

The link runs over the on-board USB-Blaster using a JTAG UART core, not a plain
COM port. Two earlier transports failed: juart-terminal mangles raw binary, and
piping binary to System Console's stdin crashes its native Tcl. So the link now
uses a TCP socket. A Tcl bridge (hardware/quartusD/jtag_bridge.tcl) runs inside
System Console, opens the JTAG UART bytestream, and listens on localhost. We
launch System Console once, connect as a client, stream 85-byte packets, and
read one class byte back per packet.

Protocol (same as PLAN.md):
  Host -> FPGA:  0xAA  [42 x int16 little-endian = 84 bytes]  (85 bytes total)
  FPGA -> Host:  0x55  [class: 1 byte]
The 0x55 is stripped by the bridge, so over the socket we just get the class byte.
"""

import os
import socket
import struct
import subprocess
import time

# path to System Console. Override with the SYSTEM_CONSOLE env var if your
# Quartus install lives somewhere else.
SYSTEM_CONSOLE = os.environ.get(
    "SYSTEM_CONSOLE",
    r"C:\altera_lite\25.1std\quartus\sopc_builder\bin\system-console.exe",
)
BRIDGE_TCL = os.path.join(
    os.path.dirname(__file__), "..", "hardware", "quartusD", "jtag_bridge.tcl"
)

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 51000

START_TO_FPGA = 0xAA
NUM_FEATURES = 42


class SerialLink:
    """Launches the System Console bridge and talks to it over a TCP socket."""

    def __init__(self, console_path=SYSTEM_CONSOLE, bridge_tcl=BRIDGE_TCL):
        self.console_path = console_path
        self.bridge_tcl = bridge_tcl
        self.process = None
        self.sock = None

    def open(self, connect_timeout=40):
        # launch System Console running the bridge. It needs time to boot the
        # JVM and open the bytestream before the socket is up.
        self.process = subprocess.Popen(
            [self.console_path, "--script=" + os.path.abspath(self.bridge_tcl)],
            stdin=subprocess.DEVNULL,  # no stdin pipe; that path crashes its Tcl
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # retry connecting until the Tcl socket server is listening
        deadline = time.time() + connect_timeout
        last_error = None
        while time.time() < deadline:
            try:
                self.sock = socket.create_connection((BRIDGE_HOST, BRIDGE_PORT), 2)
                self.sock.settimeout(2)
                return
            except OSError as problem:
                last_error = problem
                time.sleep(0.5)
        raise RuntimeError("bridge did not start listening: " + str(last_error))

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self.process is not None:
            self.process.terminate()
            self.process = None

    def frame_features(self, features):
        """Turn 42 float features into the 85-byte packet the FPGA expects."""
        if len(features) != NUM_FEATURES:
            raise ValueError("expected " + str(NUM_FEATURES) + " features")

        # import here so frame logic can be unit-tested without a live link
        from fixed_point import to_fixed

        payload = bytearray()
        payload.append(START_TO_FPGA)
        for value in features:
            fixed = to_fixed(value)  # float -> Q4.12 int16
            payload += struct.pack("<h", fixed)  # little-endian signed 16-bit
        return bytes(payload)

    def send_features(self, features):
        """Send one packet to the board through the bridge socket."""
        if self.sock is None:
            raise RuntimeError("link is not open")
        self.sock.sendall(self.frame_features(features))

    def read_reply(self):
        """
        Return the class byte the board sent, or None if none is waiting.

        Non-blocking so it never stalls the webcam loop. The reply for a packet
        sent this frame usually arrives within a frame or two, which is fine for
        a live overlay.
        """
        if self.sock is None:
            raise RuntimeError("link is not open")
        self.sock.setblocking(False)
        try:
            data = self.sock.recv(1)
        except (BlockingIOError, socket.timeout):
            return None
        finally:
            self.sock.setblocking(True)
            self.sock.settimeout(2)
        if not data:
            return None
        return data[0]
