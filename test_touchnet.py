import py3DCal as p3d
import os
import numpy as np

num_tests = 0
num_passed = 0

def test_incorrect_sensor_type():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type="INVALID_SENSOR", load_pretrained_model=False, download_dataset=False, root="data/", device="mps")
    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for incorrect sensor type: {e}\033[0m\n")

        global num_passed
        num_passed += 1

def test_incorrect_device():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.DIGIT, load_pretrained_model=False, download_dataset=False, root="data/", device=None)
    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for incorrect device: {e}\033[0m\n")

        global num_passed
        num_passed += 1

def test_download_dataset_with_custom():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.CUSTOM, load_pretrained_model=False, download_dataset=True, root="data/", device="cpu")
    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for dataset download with custom sensor: {e}\033[0m\n")

        global num_passed
        num_passed += 1

def test_load_model_with_custom():
    global num_tests
    num_tests += 1
    
    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.CUSTOM, load_pretrained_model=True, download_dataset=False, root="data/", device="cpu")
    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for loading model with custom sensor: {e}\033[0m\n")

        global num_passed
        num_passed += 1

def test_digit_dataset_download():
    global num_tests
    num_tests += 1

    touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.DIGIT, load_pretrained_model=False, download_dataset=True, device="mps")

    # Check if ./digit_calibration_data exists
    dataset_dir = os.path.join(".", "digit_calibration_data")

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        print(f"\033[92m[Test {num_tests}]: DIGIT dataset downloaded successfully.\n\033[0m")

        global num_passed
        num_passed += 1
    else:
        print(f"\033[91m[Test {num_tests}]: DIGIT dataset download failed.\n\033[0m")

def test_digit_download_model():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.DIGIT, load_pretrained_model=True, download_dataset=False, device="mps")

        if os.path.exists("digit_pretrained_weights.pth"):
            print(f"\033[92m[Test {num_tests}]: DIGIT model downloaded successfully.\n\033[0m")

            global num_passed
            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: DIGIT model download failed.\n\033[0m")
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: DIGIT model download failed: {e}\n\033[0m")

def test_digit_load_model():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.DIGIT, load_pretrained_model=False, download_dataset=False, device="mps")

        touchnet.load_model_weights(model_path="digit_pretrained_weights.pth")

        touchnet.save_depthmap_image(image_path="misc/digit_pill.png", save_path="misc/digit_depthmap.png")

        if os.path.exists("misc/digit_depthmap.png"):
            print(f"\033[92m[Test {num_tests}]: DIGIT pretrained model downloaded from .pth file successfully.\n\033[0m")

            global num_passed
            num_passed += 1
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Load DIGIT model from .pth file failed: {e}\n\033[0m")

def test_digit_depthmap():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.DIGIT, load_pretrained_model=True, download_dataset=False, device="mps")

        depthmap = touchnet.get_depthmap(image_path="misc/digit_pill.png")

        if depthmap is not None and depthmap.shape == (320, 240):
            print(f"\033[92m[Test {num_tests}]: DIGIT depthmap generated successfully.\n\033[0m")

            global num_passed
            num_passed += 1

        else:
            print(f"\033[91m[Test {num_tests}]: DIGIT depthmap generation failed.\n\033[0m")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: DIGIT depthmap generation failed: {e}\n\033[0m")

def test_gsmini_dataset_download():
    global num_tests
    num_tests += 1

    touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.GELSIGHTMINI, load_pretrained_model=False, download_dataset=True, device="mps")

    # Check if ./gsmini_calibration_data exists
    dataset_dir = os.path.join(".", "gsmini_calibration_data")

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        print(f"\033[92m[Test {num_tests}]: GelSight Mini dataset downloaded successfully.\n\033[0m")

        global num_passed
        num_passed += 1
    else:
        print(f"\033[91m[Test {num_tests}]: GelSight Mini dataset download failed.\n\033[0m")

def test_gsmini_download_model():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.GELSIGHTMINI, load_pretrained_model=True, download_dataset=False, device="mps")

        if os.path.exists("gsmini_pretrained_weights.pth"):
            print(f"\033[92m[Test {num_tests}]: GelSight Mini model downloaded successfully.\n\033[0m")

            global num_passed
            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: GelSight Mini model download failed.\n\033[0m")
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: GelSight Mini model download failed: {e}\n\033[0m")

def test_gsmini_load_model():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.GELSIGHTMINI, load_pretrained_model=False, download_dataset=False, device="mps")

        touchnet.load_model_weights(model_path="gsmini_pretrained_weights.pth")

        touchnet.save_depthmap_image(image_path="misc/gsmini_pill.png", save_path="misc/gsmini_depthmap.png")

        if os.path.exists("misc/gsmini_depthmap.png"):
            print(f"\033[92m[Test {num_tests}]: GelSight Mini pretrained model downloaded from .pth file successfully.\n\033[0m")

            global num_passed
            num_passed += 1
    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Load GelSight Mini model from .pth file failed: {e}\n\033[0m")

def test_gsmini_depthmap():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.GELSIGHTMINI, load_pretrained_model=True, download_dataset=False, device="mps")

        depthmap = touchnet.get_depthmap(image_path="misc/gsmini_pill.png")

        if depthmap is not None and depthmap.shape == (240, 320):
            print(f"\033[92m[Test {num_tests}]: GelSight Mini depthmap generated successfully.\n\033[0m")

            global num_passed
            num_passed += 1

        else:
            print(f"\033[91m[Test {num_tests}]: GelSight Mini depthmap generation failed.\n\033[0m")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: GelSight Mini depthmap generation failed: {e}\n\033[0m")

def test_training():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.TouchNet(sensor_type=p3d.SensorType.DIGIT, load_pretrained_model=False, download_dataset=True, device="mps")

        touchnet.train(num_epochs=1, batch_size=32)

        print(f"\033[92m[Test {num_tests}]: TouchNet training completed successfully.\n\033[0m")

        global num_passed
        num_passed += 1

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: TouchNet training failed: {e}\n\033[0m")

if __name__ == "__main__":
    # TouchNet Initialization Tests
    test_incorrect_sensor_type()
    test_incorrect_device()
    test_download_dataset_with_custom()
    test_load_model_with_custom()

    # DIGIT Testing
    test_digit_dataset_download()
    test_digit_download_model()
    test_digit_load_model()
    test_digit_depthmap()

    # Mini Testing
    test_gsmini_dataset_download()
    test_gsmini_download_model()
    test_gsmini_load_model()
    test_gsmini_depthmap()

    # Test TouchNet Training
    # test_training()

    print(f"{num_passed}/{num_tests} TESTS PASSED.")

