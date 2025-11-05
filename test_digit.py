import py3DCal as p3d

num_tests = 0
num_passed = 0

def test_digit():
    global num_tests

    digit = p3d.DIGIT("D20966")

    num_tests += 1
    # Try connecting to the DIGIT
    try:
        digit.connect()
        print(f"\033[92m[Test {num_tests}]: Connected to DIGIT successfully.\033[0m\n")
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: DIGIT connection failed: {e}\033[0m\n")

    num_tests += 1
    # Try capturing an image
    try:
        image = digit.capture_image()

        if image is not None:
            print(f"\033[92m[Test {num_tests}]: Image captured successfully.\033[0m\n")
        else:
            print(f"\033[91m[Test {num_tests}]: Failed to capture image.\033[0m\n")
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Error during image capture: {e}\033[0m\n")

    num_tests += 1
    # Try disconnecting from the DIGIT
    try:
        digit.disconnect()
        print(f"\033[92m[Test {num_tests}]: Disconnected from DIGIT successfully.\033[0m\n")
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: DIGIT disconnection failed: {e}\033[0m\n")

if __name__ == "__main__":
    test_digit()

    print(f"{num_passed}/{num_tests} TEST PASSED.")