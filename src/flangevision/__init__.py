"""Machine-vision inspection of a synthetic flange: rendering with ground truth, classical metrology and defect rules."""
from .render import render, CLASSES, NOMINAL, MM_PER_PX
from .measure import inspect, fit_circle, flat_field, classify, TOL

__all__ = ["render", "CLASSES", "NOMINAL", "MM_PER_PX", "inspect", "fit_circle", "flat_field", "classify", "TOL"]
