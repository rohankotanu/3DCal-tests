import os
import math
import torch.nn as nn
import py3DCal as p3d
from py3DCal import datasets, models

num_tests = 0
num_passed = 0

def test_digit_dataset_download():
    global num_tests
    num_tests += 1

    dataset = datasets.DIGIT(root='.', download=True, add_coordinate_embeddings=False)

    # Check if ./digit_calibration_data exists
    dataset_dir = os.path.join(".", "digit_calibration_data")

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir) and dataset[0][0].size() == (3, 320, 240):
        print(f"\033[92m[Test {num_tests}]: DIGIT dataset downloaded successfully.\n\033[0m")

        global num_passed
        num_passed += 1
    else:
        print(f"\033[91m[Test {num_tests}]: DIGIT dataset download failed.\n\033[0m")

def test_gsmini_dataset_download():
    global num_tests
    num_tests += 1

    dataset = datasets.GelSightMini(root='.', download=True, add_coordinate_embeddings=False)

    # Check if ./gsmini_calibration_data exists
    dataset_dir = os.path.join(".", "gsmini_calibration_data")

    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir) and dataset[0][0].size() == (3, 240, 320):
        print(f"\033[92m[Test {num_tests}]: GelSight Mini dataset downloaded successfully.\n\033[0m")

        global num_passed
        num_passed += 1
    else:
        print(f"\033[91m[Test {num_tests}]: GelSight Mini dataset download failed.\n\033[0m")

def test_digit_dataset_no_root():
    global num_tests
    num_tests += 1

    try:
        dataset = datasets.DIGIT(root=None, download=True, add_coordinate_embeddings=False)

        print(f"\033[91m[Test {num_tests}]: DIGIT dataset download without root should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for missing DIGIT dataset root: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_gsmini_dataset_no_root():
    global num_tests
    num_tests += 1

    try:
        dataset = datasets.GelSightMini(root=None, download=True, add_coordinate_embeddings=False)

        print(f"\033[91m[Test {num_tests}]: GelSight Mini dataset download without root should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for missing GelSight Mini dataset root: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_digit_dataset_invalid_root():
    global num_tests
    num_tests += 1

    try:
        dataset = datasets.DIGIT(root=True, download=False, add_coordinate_embeddings=False)

        print(f"\033[91m[Test {num_tests}]: DIGIT dataset loading from invalid root should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for invalid DIGIT dataset root: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_gsmini_dataset_invalid_root():
    global num_tests
    num_tests += 1

    try:
        dataset = datasets.GelSightMini(root=True, download=False, add_coordinate_embeddings=False)

        print(f"\033[91m[Test {num_tests}]: GelSight Mini dataset loading from invalid root should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for invalid GelSight Mini dataset root: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_load_custom_mini_dataset():
    global num_tests
    num_tests += 1

    try:
        dataset = datasets.GelSightMini(root='./', add_coordinate_embeddings=False, subtract_blank=True, transform=None)

        if len(dataset) == 36270 and dataset[0][0].size() == (3, 240, 320) and dataset[0][1].size() == (2, 240, 320):
            print(f"\033[92m[Test {num_tests}]: Custom GelSight Mini dataset loaded successfully with {len(dataset)} samples.\n\033[0m")

            global num_passed
            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: Custom GelSight Mini dataset loading failed. Expected 36270 samples, got {len(dataset)}.\n\033[0m")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Custom GelSight Mini dataset loading failed: {e}\n\033[0m")

def test_load_custom_tactile_sensor_dataset():
    global num_tests
    num_tests += 1

    try:
        dataset = datasets.TactileSensorDataset(root='./digit_calibration_data', add_coordinate_embeddings=False, subtract_blank=True, transform=None)

        if len(dataset) == 36630 and dataset[0][0].size() == (3, 320, 240) and dataset[0][1].size() == (2, 320, 240):
            print(f"\033[92m[Test {num_tests}]: Custom TactileSensorDataset loaded successfully with {len(dataset)} samples.\n\033[0m")

            global num_passed
            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: Custom TactileSensorDataset loading failed. Expected 36630 samples, got {len(dataset)}.\n\033[0m")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Custom TactileSensorDataset loading failed: {e}\n\033[0m")

def test_load_custom_tactile_sensor_dataset_no_root():
    global num_tests
    num_tests += 1

    try:
        dataset = datasets.TactileSensorDataset(root=None, add_coordinate_embeddings=False, subtract_blank=True, transform=None)
        print(f"\033[91m[Test {num_tests}]: TactileSensorDataset loading without root should have failed but didn't.\n\033[0m")

    except Exception as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for missing TactileSensorDataset root: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_load_custom_tactile_sensor_dataset_invalid_root():
    global num_tests
    num_tests += 1

    try:
        dataset = datasets.TactileSensorDataset(root=True, add_coordinate_embeddings=False, subtract_blank=True, transform=None)
        print(f"\033[91m[Test {num_tests}]: TactileSensorDataset loading with invalid root should have failed but didn't.\n\033[0m")

    except Exception as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for invalid TactileSensorDataset root: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_add_coordinate_embeddings():
    global num_tests
    num_tests += 1

    dataset = datasets.DIGIT(root='.', download=False, add_coordinate_embeddings=True)

    sample_image, _ = dataset[0]

    if sample_image.shape[0] == 5:
        print(f"\033[92m[Test {num_tests}]: Coordinate embeddings added successfully to DIGIT dataset (Shape: {list(sample_image.size())}).\n\033[0m")

        global num_passed
        num_passed += 1
    else:
        print(f"\033[91m[Test {num_tests}]: Failed to add coordinate embeddings to DIGIT dataset (Shape: {list(sample_image.size())}).\n\033[0m")

def test_split_dataset():
    global num_tests
    num_tests += 1

    dataset = datasets.GelSightMini(root='.', download=False)

    train_dataset, val_dataset = p3d.split_dataset(dataset, train_ratio=0.8)

    expected_train_size = math.floor(0.8 * len(dataset) / 30) * 30
    expected_val_size = math.ceil(0.2 * len(dataset) / 30) * 30

    if len(train_dataset) == expected_train_size and len(val_dataset) == expected_val_size:
        print(f"\033[92m[Test {num_tests}]: Dataset ({len(dataset)}) split successfully into train ({len(train_dataset)}) and validation ({len(val_dataset)}) sets.\n\033[0m")

        global num_passed
        num_passed += 1
    else:
        print(f"\033[91m[Test {num_tests}]: Dataset split failed. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}.\n\033[0m")

def test_digit_load_model():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.models.TouchNet(sensor_type=p3d.SensorType.DIGIT, load_pretrained=True, root=".")

        if os.path.exists("digit_pretrained_weights.pth"):
            print(f"\033[92m[Test {num_tests}]: Pretrained DIGIT model loaded successfully.\n\033[0m")

            global num_passed
            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: Pretrained DIGIT model loading failed.\n\033[0m")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Pretrained DIGIT model loading failed: {e}\n\033[0m")

def test_gsmini_load_model():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.models.TouchNet(sensor_type=p3d.SensorType.GELSIGHTMINI, load_pretrained=True, root=".")

        if os.path.exists("gsmini_pretrained_weights.pth"):
            print(f"\033[92m[Test {num_tests}]: Pretrained GelSightMini model loaded successfully.\n\033[0m")

            global num_passed
            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: Pretrained GelSightMini model loading failed.\n\033[0m")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Pretrained GelSightMini model loading failed: {e}\n\033[0m")

def test_load_model_no_sensor_type():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.models.TouchNet(load_pretrained=True, root=".")

        print(f"\033[91m[Test {num_tests}]: Model loading without sensor type should have failed but didn't.\n\033[0m")
    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for missing sensor type: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_load_model_wrong_sensor_type():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.models.TouchNet(load_pretrained=True, sensor_type="hello",root=".")

        print(f"\033[91m[Test {num_tests}]: Model loading with incorrect sensor type should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for incorrect sensor type: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_load_model_no_root():
    global num_tests
    num_tests += 1

    try:
        touchnet = p3d.models.TouchNet(load_pretrained=True, sensor_type=p3d.SensorType.DIGIT, root=None)

        print(f"\033[91m[Test {num_tests}]: Model loading without root should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for missing root: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_digit_depthmap():
    global num_tests
    num_tests += 1

    model = p3d.models.TouchNet(load_pretrained=True, sensor_type=p3d.SensorType.DIGIT, root=".")
    try:
        p3d.save_2d_depthmap(model, image_path="misc/digit_pill.png", blank_image_path="./digit_calibration_data/blank_images/blank.png", save_path="misc/digit_depthmap.png")

        if os.path.exists("misc/digit_depthmap.png"):
            print(f"\033[92m[Test {num_tests}]: Digit depthmap generated successfully.\n\033[0m")

            global num_passed
            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: Digit depthmap generation failed.\n\033[0m")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Digit depthmap generation failed: {e}\n\033[0m")

def test_gsmini_depthmap():
    global num_tests
    num_tests += 1

    model = p3d.models.TouchNet(load_pretrained=True, sensor_type=p3d.SensorType.GELSIGHTMINI, root=".")
    try:
        p3d.save_2d_depthmap(model, image_path="misc/gsmini_pill.png", blank_image_path="./gsmini_calibration_data/blank_images/blank.png", save_path="misc/gsmini_depthmap.png")

        if os.path.exists("misc/gsmini_depthmap.png"):
            print(f"\033[92m[Test {num_tests}]: GelSight Mini depthmap generated successfully.\n\033[0m")

            global num_passed
            num_passed += 1
        else:
            print(f"\033[91m[Test {num_tests}]: GelSight Mini depthmap generation failed.\n\033[0m")

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: GelSight Mini depthmap generation failed: {e}\n\033[0m")

def test_get_depthmap_wrong_device():
    global num_tests
    num_tests += 1

    try:
        model = p3d.models.TouchNet(load_pretrained=False)

        depthmap = p3d.get_depthmap(model, image_path="misc/digit_pill.png", blank_image_path="./digit_calibration_data/blank_images/blank.png", device="hello")

        print(f"\033[91m[Test {num_tests}]: Depthmap generation on unsupported device should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for unsupported device: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_train_model():
    global num_tests
    num_tests += 1

    try:
        model = p3d.models.TouchNet(load_pretrained=False)
        dataset = datasets.DIGIT(root='.', download=False)

        p3d.train_model(model, dataset, num_epochs=1, batch_size=32, learning_rate=1e-4, train_ratio=0.1,loss_fn=nn.MSELoss(), device="mps")

        print(f"\033[92m[Test {num_tests}]: Model training completed successfully.\n\033[0m")

        global num_passed
        num_passed += 1

    except Exception as e:
        print(f"\033[91m[Test {num_tests}]: Model training failed: {e}\n\033[0m")

def test_train_model_wrong_device():
    global num_tests
    num_tests += 1

    try:
        model = p3d.models.TouchNet(load_pretrained=False)
        dataset = datasets.DIGIT(root='.', download=False)

        p3d.train_model(model, dataset, num_epochs=1, batch_size=32, learning_rate=1e-4, loss_fn=nn.MSELoss(), device="hello")

        print(f"\033[91m[Test {num_tests}]: Model training on unsupported device should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for unsupported device during training: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_train_model_bad_model():
    global num_tests
    num_tests += 1

    try:
        dataset = datasets.DIGIT(root='.', download=False)

        p3d.train_model(dataset, dataset, num_epochs=1, batch_size=32, learning_rate=1e-4, loss_fn=nn.MSELoss(), device="hello")

        print(f"\033[91m[Test {num_tests}]: Model training on non-model should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for trying to train on non-model: {e}\n\033[0m")

        global num_passed
        num_passed += 1

def test_train_model_bad_dataset():
    global num_tests
    num_tests += 1

    try:
        model = p3d.models.TouchNet(load_pretrained=False)

        p3d.train_model(model, model, num_epochs=1, batch_size=32, learning_rate=1e-4, loss_fn=nn.MSELoss(), device="hello")

        print(f"\033[91m[Test {num_tests}]: Model training on non-TactileSensorDataset should have failed but didn't.\n\033[0m")

    except ValueError as e:
        print(f"\033[92m[Test {num_tests}]: Caught expected ValueError for trying to train on non-TactileSensorDataset: {e}\n\033[0m")

        global num_passed
        num_passed += 1

if __name__ == "__main__":
    # Dataset tests
    test_digit_dataset_download()
    test_gsmini_dataset_download()
    test_digit_dataset_no_root()
    test_gsmini_dataset_no_root()
    test_digit_dataset_invalid_root()
    test_gsmini_dataset_invalid_root()
    test_load_custom_mini_dataset()
    test_load_custom_tactile_sensor_dataset()
    test_load_custom_tactile_sensor_dataset_no_root()
    test_load_custom_tactile_sensor_dataset_invalid_root()
    test_add_coordinate_embeddings()
    test_split_dataset()

    # TouchNet model tests
    test_digit_load_model()
    test_gsmini_load_model()
    test_load_model_no_sensor_type()
    test_load_model_wrong_sensor_type()
    test_load_model_no_root()

    # Test model training
    # test_train_model()
    test_train_model_wrong_device()
    test_train_model_bad_model()
    test_train_model_bad_dataset()

    # Test depthmap generation
    test_digit_depthmap()
    test_gsmini_depthmap()
    test_get_depthmap_wrong_device()

    print(f"{num_passed}/{num_tests} TEST PASSED.")

    # p3d.train_model(model, dataset, num_epochs=1, batch_size=32, learning_rate=1e-4, loss_fn=nn.MSELoss(), device="mps")