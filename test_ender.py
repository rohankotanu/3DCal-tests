import py3DCal as p3d

num_tests = 0
num_passed = 0

def test_ender():
    global num_tests
    global num_passed

    # ender3 = p3d.Ender3("/dev/ttyUSB0")
    ender3 = p3d.Ender3("/dev/tty.usbserial-110")

    num_tests += 1
    # Try connecting to the Ender 3
    try:
        ender3.connect()
        print(f"\033[92m[Test {num_tests}]: Connected to Ender 3 successfully.\033[0m\n")

        num_passed += 1
    
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Ender 3 connection failed: {e}\033[0m\n")

    num_tests += 1
    # Try initializing printer
    try:
        ender3.initialize()

        print(f"\033[92m[Test {num_tests}]: Ender 3 initialized successfully.\033[0m\n")

        num_passed += 1

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Error during printer initialization: {e}\033[0m\n")

    num_tests += 1
    # Try moving printhead
    try:
        ender3.go_to(x=5, y=5, z=5)

        print(f"\033[92m[Test {num_tests}]: Ender 3 printhead moved successfully.\033[0m\n")

        num_passed += 1

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Error during printhead movement: {e}\033[0m\n")

    num_tests += 1
    # Try moving printhead
    try:
        ender3.send_gcode("M117 Send G-Code Test Passed")

        resp = input("Does the Ender 3 display show the message 'Send G-Code Test Passed'? (Y/N): ")

        if resp.strip().upper() == "Y":
            print(f"\033[92m[Test {num_tests}]: Sent G-Code to Ender 3 successfully.\033[0m\n")

            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: Error while sending G-Code to Ender 3: {e}\033[0m\n")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Error while sending G-Code to Ender 3: {e}\033[0m\n")

    num_tests += 1
    # Try getting response from Ender 3
    try:
        response = ender3.get_response().strip()

        if response == "ok":
            print(f"\033[92m[Test {num_tests}]: Received response from Ender 3: {response}\033[0m\n")

            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: Error while getting response from Ender 3.\033[0m\n")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Error while getting response from Ender 3: {e}\033[0m\n")

    num_tests += 1
    # Try disconnecting from the Ender 3
    try:
        ender3.disconnect()
        print(f"\033[92m[Test {num_tests}]: Disconnected from Ender 3 successfully.\033[0m\n")

        num_passed += 1
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Ender 3 disconnection failed: {e}\033[0m\n")

if __name__ == "__main__":
    test_ender()

    print(f"{num_passed}/{num_tests} TEST PASSED.")