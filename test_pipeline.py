import py3DCal as p3d
from py3DCal import datasets, models

# Data Collection
digit = p3d.DIGIT("D20966")
ender3 = p3d.Ender3("/dev/ttyUSB0")
calibrator = p3d.Calibrator(printer=ender3, sensor=digit)

calibrator.probe(calibration_file_path="misc/probe_points.csv", save_images=False)

# Model Training
my_dataset = datasets.TactileSensorDataset(root='./sensor_calibration_data')
dataset = datasets.DIGIT(root='.', download=True)
dataset = datasets.GelSightMini(root='.', download=True)

touchnet = models.TouchNet(load_pretrained=True, sensor_type=p3d.SensorType.DIGIT, root=".")

p3d.train_model(model=touchnet, dataset=my_dataset, num_epochs=1, batch_size=32, device="mps")

# Depthmap Generation
depthmap = p3d.get_depthmap(model=touchnet, image_path="pawn.png", blank_image_path="./sensor_calibration_data/blank_images/blank.png")