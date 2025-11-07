import py3DCal as p3d
from Calibrator import Calibrator
import cv2

digit = p3d.DIGIT("D20966")
ender3 = p3d.Ender3("/dev/ttyUSB0")

calibrator = Calibrator(printer=ender3, sensor=digit)
digit.connect()
print(digit.sensor._Digit__dev.get(cv2.CAP_PROP_BUFFERSIZE))

# calibrator.probe(home_printer=False, calibration_file_path="misc/probe_points.csv")