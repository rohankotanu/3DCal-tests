import py3DCal as p3d
from py3DCal import datasets, models
from torch import nn
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # dataset = datasets.DIGIT(root='.', download=True, add_coordinate_embeddings=True, subtract_blank=False, transform=None)
    dataset = datasets.TactileSensorDataset(root='digit_calibration_data/', add_coordinate_embeddings=False, subtract_blank=True, transform=None)

    # train_dataset, val_dataset = p3d.split_dataset(dataset, train_ratio=0.8)

    # dataset = datasets.GelSightMini(root='data', download=True, add_position_embeddings=True, transform=None)

    # model = models.TouchNet(load_pretrained=True, sensor_type=p3d.SensorType.DIGIT, root=".")

    # # p3d.train_model(model, dataset, num_epochs=1, batch_size=32, learning_rate=1e-4, loss_fn=nn.MSELoss(), device="mps")

    # depthmap = p3d.get_depthmap(model, image_path="misc/digit_pill.png", blank_image_path="./digit_calibration_data/blank_images/blank.png")

    # plt.imsave("test.png", depthmap, cmap='viridis')