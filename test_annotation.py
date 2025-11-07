import py3DCal as p3d

num_tests = 0
num_passed = 0

def test_annotate_no_root():
    global num_tests
    global num_passed

    num_tests += 1

    try:
        p3d.annotate(dataset_path=None, probe_radius_mm=2)

        print(f"\033[91m[Test {num_tests}]: Annotation without root should have failed but didn't.\033[0m\n")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Annotation without root raised ValueError as expected: {e}\033[0m\n")

        num_passed += 1

def test_annotate_nonexistent_root():
    global num_tests
    global num_passed

    num_tests += 1

    try:
        p3d.annotate(dataset_path="digit_calibration_datas/", probe_radius_mm=2)

        print(f"\033[91m[Test {num_tests}]: Annotation with nonexistent root should have failed but didn't.\033[0m\n")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Annotation with nonexistent root raised ValueError as expected: {e}\033[0m\n")

        num_passed += 1

def test_annotate_no_radius():
    global num_tests
    global num_passed

    num_tests += 1

    try:
        p3d.annotate(dataset_path="digit_calibration_data/", probe_radius_mm=None)

        print(f"\033[91m[Test {num_tests}]: Annotation without radius should have failed but didn't.\033[0m\n")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Annotation without radius raised ValueError as expected: {e}\033[0m\n")

        num_passed += 1

def test_annotate_invalid_radius():
    global num_tests
    global num_passed

    num_tests += 1

    try:
        p3d.annotate(dataset_path="digit_calibration_data/", probe_radius_mm="invalid")

        print(f"\033[91m[Test {num_tests}]: Annotation with invalid radius should have failed but didn't.\033[0m\n")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Annotation with invalid radius raised ValueError as expected: {e}\033[0m\n")

        num_passed += 1

def test_annotate_negative_radius():
    global num_tests
    global num_passed

    num_tests += 1

    try:
        p3d.annotate(dataset_path="digit_calibration_data/", probe_radius_mm=-2)

        print(f"\033[91m[Test {num_tests}]: Annotation with negative radius should have failed but didn't.\033[0m\n")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Annotation with negative radius raised ValueError as expected: {e}\033[0m\n")

        num_passed += 1

def test_annotate_bool_indices():
    global num_tests
    global num_passed

    num_tests += 1

    try:
        p3d.annotate(dataset_path="digit_calibration_data/", probe_radius_mm=2, img_idxs=True)

        print(f"\033[91m[Test {num_tests}]: Annotation with boolean indices should have failed but didn't.\033[0m\n")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Annotation with boolean indices raised ValueError as expected: {e}\033[0m\n")

        num_passed += 1

def test_annotate_float_indices():
    global num_tests
    global num_passed

    num_tests += 1

    try:
        p3d.annotate(dataset_path="digit_calibration_data/", probe_radius_mm=2, img_idxs=[4.0, 1.0])

        print(f"\033[91m[Test {num_tests}]: Annotation with float indices should have failed but didn't.\033[0m\n")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Annotation with float indices raised ValueError as expected: {e}\033[0m\n")

        num_passed += 1

def test_annotate_one_index():
    global num_tests
    global num_passed

    num_tests += 1

    try:
        p3d.annotate(dataset_path="digit_calibration_data/", probe_radius_mm=2, img_idxs=[2])

        print(f"\033[91m[Test {num_tests}]: Annotation with one index should have failed but didn't.\033[0m\n")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Annotation with one index raised ValueError as expected: {e}\033[0m\n")

        num_passed += 1

def test_annotate_three_indices():
    global num_tests
    global num_passed

    num_tests += 1

    try:
        p3d.annotate(dataset_path="digit_calibration_data/", probe_radius_mm=2, img_idxs=[2, 3, 4])

        print(f"\033[91m[Test {num_tests}]: Annotation with three indices should have failed but didn't.\033[0m\n")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Annotation with three indices raised ValueError as expected: {e}\033[0m\n")

        num_passed += 1

if __name__ == "__main__":
    test_annotate_no_root()
    test_annotate_nonexistent_root()
    test_annotate_no_radius()
    test_annotate_invalid_radius()
    test_annotate_negative_radius()
    test_annotate_bool_indices()
    test_annotate_float_indices()
    test_annotate_one_index()
    test_annotate_three_indices()

    print(f"{num_passed}/{num_tests} TESTS PASSED.")