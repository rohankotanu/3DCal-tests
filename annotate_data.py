import pandas as pd

df = pd.read_csv('sensor_calibration_data/annotations/probe_data.csv')
df['x_px'] = 10
df['y_px'] = 10

df.to_csv("sensor_calibration_data/annotations/annotations.csv", index=False)
print(df)