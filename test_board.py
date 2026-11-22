import serial
import time

with serial.Serial("COM12", 115200, timeout=1) as ser:
    time.sleep(1)
    ser.reset_input_buffer()

    command = (
        "cd /root/hackedge && "
        "python3 board_shift.py people && "
        "ls -lh models/shift_model.json"
    )

    ser.write((command + "\n").encode())
    ser.flush()

    deadline = time.monotonic() + 15
    response = ""

    while time.monotonic() < deadline:
        chunk = ser.read(4096)
        if chunk:
            response += chunk.decode(errors="replace")
            if "root@am62xx-evm:" in response:
                break

    print(response)
