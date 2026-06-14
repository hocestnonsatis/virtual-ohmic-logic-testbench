#!/usr/bin/env python3
"""Minimal smoke test for the optional volt Python module."""

import sys

try:
    import volt
except ImportError as e:
    print("volt module not built; skip:", e, file=sys.stderr)
    sys.exit(0)

cfg = volt.Config()
cfg.iv_model = volt.IvModel.PowerLaw
cfg.iv_exponent = 1.5

W = [[0.8, -0.3], [-0.6, 0.9]]
x = [0.5, 0.7]
currents, levels = volt.forward(W, x, cfg)
assert len(currents) == 2
assert len(levels) == 2

W2 = [[0.5, 0.0], [0.0, 0.5]]
out = volt.two_layer_forward(W, W2, x, cfg, interlayer="diode_rectifier")
assert "mse" in out
print("python smoke test OK")
