
import serial
import time
import uuid
import re

PORT = "COM12"
BAUD = 115200

def run_board_command(command, timeout=180):
    marker = "HACKEDGE_DONE_" + uuid.uuid4().hex

    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        time.sleep(0.5)
        ser.reset_input_buffer()

        wrapped = (
            "cd /root/hackedge && "
            f"({command}); rc=$?; "
            f"printf '\\n{marker}:%s\\n' \"$rc\""
        )

        ser.write((wrapped + "\n").encode()
                  )
        ser.flush()

        response = ""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            chunk = ser.read(4096)
            if chunk:
                response += chunk.decode(errors="replace")

                match = re.search(
                    r"(?m)^" + re.escape(marker) + r":(\d+)\r?$",
                    response
                )

                if match:
                    status = int(match.group(1))
                    if status != 0:
                        raise RuntimeError(
                            f"Board command failed ({status}):\n{response}"
                        )
                    return response

        raise TimeoutError(
            f"Board did not finish within {timeout}s:\n{response}"
        )

def list_people():
    return run_board_command(
        "python3 board_shift.py people"
    )

def train_model():
    return run_board_command(
        "python3 board_shift.py train",
        timeout=900
    )
