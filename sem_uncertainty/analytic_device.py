"""Declared square-law/subthreshold MOS surrogate for a self-contained setting."""

import numpy as np


class AnalyticDevice:
    """Geometry is width/length/oxide thickness in nm; device returns amperes."""

    def __init__(self, voltage):
        self.voltage = np.asarray(voltage, dtype=float)
        if self.voltage.ndim != 1 or len(self.voltage) < 2 or not np.all(np.diff(self.voltage) > 0):
            raise ValueError("Supply strictly increasing gate voltages")

    def geometry(self, theta):
        return np.asarray(theta, dtype=float)

    def device(self, theta):
        width, length, oxide = np.asarray(theta, dtype=float)
        if min(width, length, oxide) <= 0:
            raise ValueError("Geometry dimensions must be positive")
        threshold = .25 + .05 * (oxide - 2.)
        overdrive = self.voltage - threshold
        subthreshold = 1e-12 * np.exp(np.minimum(overdrive, 0.) / .07)
        return subthreshold + 2e-6 * width / length / oxide * np.maximum(overdrive, 0.) ** 2
