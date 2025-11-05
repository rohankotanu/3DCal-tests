import py3DCal as p3d
from PIL import Image
import os

num_tests = 0
num_passed = 0

def test_digit():
    global num_tests
    global num_passed

    digit = p3d.DIGIT("D20966")

    num_tests += 1
    # Try connecting to the DIGIT
    try:
        digit.connect()
        print(f"\033[92m[Test {num_tests}]: Connected to DIGIT successfully.\033[0m\n")

        num_passed += 1

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: DIGIT connection failed: {e}\033[0m\n")

    num_tests += 1
    # Try capturing an image
    try:

        for i in range(30):
            image = digit.capture_image()

        Image.fromarray(image).save("misc/digit_test_capture.png")

        if image is not None and os.path.exists('misc/digit_test_capture.png'):
            print(f"\033[92m[Test {num_tests}]: Image captured successfully.\033[0m\n")

            num_passed += 1

        else:
            print(f"\033[91m[Test {num_tests}]: Failed to capture image.\033[0m\n")
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Error during image capture: {e}\033[0m\n")

    num_tests += 1
    # Try disconnecting from the DIGIT
    try:
        digit.disconnect()
        print(f"\033[92m[Test {num_tests}]: Disconnected from DIGIT successfully.\033[0m\n")

        num_passed += 1

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: DIGIT disconnection failed: {e}\033[0m\n")

if __name__ == "__main__":
    test_digit()

    print(f"{num_passed}/{num_tests} TEST PASSED.")