import py3DCal as p3d
from Calibrator import Calibrator

digit = p3d.DIGIT("D20966")
ender3 = p3d.Ender3("/dev/ttyUSB0")

calibrator = Calibrator(printer=ender3, sensor=digit)

calibrator.probe(home_printer=False, calibration_file_path="misc/probe_points.csv")