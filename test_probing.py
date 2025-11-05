import py3DCal as p3d

num_tests = 0
num_passed = 0

def test_probing():
    global num_tests
    global num_passed

    num_tests += 1
    # Try probing sensor
    try:
        digit = p3d.DIGIT("D20966")
        ender3 = p3d.Ender3("/dev/ttyUSB0")
        # ender3 = p3d.Ender3("/dev/tty.usbserial-110")

        calibrator = p3d.Calibrator(printer=ender3, sensor=digit)

        calibrator.probe(calibration_file_path="misc/probe_points.csv", save_images=True)
        print(f"\033[92m[Test {num_tests}]: Probing completed successfully.\033[0m")

        num_passed += 1
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Error during probing: {e}\033[0m")

if __name__ == "__main__":
    test_probing()

    print(f"{num_passed}/{num_tests} TEST PASSED.")