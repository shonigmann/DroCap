import numpy as np


def get_inv_sqr_coefficients(x, y):
    """
    Returns coefficients for an inverse square function that intersects the two points specified by the camera range
    and score range
    :param x known x coordinates, x1 and x2
    :param y known y coordinates, y1 and y2

    :return: b and n corresponding to function f(x) = n / (x - b)^2
    """

    if x[0] != x[1] and y[0] != y[1]:
        a_ = (y[0] - y[1])
        b_ = 2 * (y[1] * x[1] - y[0] * x[0])
        c_ = y[0] * x[0] * x[0] - y[1] * x[1] * x[1]
        disc = b_ * b_ - 4 * a_ * c_
        if disc < 0:
            print("ERROR: INVALID PARAMETERS")
            print("x[0]: " + str(x[0]) + "; x[1]: " + str(x[1]) + "; y[0]: " + str(y[0]) + "; y[1]: " + str(y[1]))
            print("a: " + str(a_) + "; b: " + str(b_) + "; c: " + str(c_))
            b = -b_ / (2 * a_)
            n = y[0] * np.power(x[0] - b, 2)
            return b, n

        b = (-b_ - np.sqrt(disc)) / (2 * a_)
        n = y[0] * np.power(x[0] - b, 2)

        return b, n
    else:
        print("Warning: invalid inputs")
        return 0, 0
