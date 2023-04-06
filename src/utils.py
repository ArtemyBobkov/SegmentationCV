from math import fabs


def assert_close(a, b, eps=1e-2):
    assert fabs(a - b) < eps
