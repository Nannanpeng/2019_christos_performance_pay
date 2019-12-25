import numpy as np

# transformation to comp domain -- range of [k_bar, k_up]
def box_to_cube(knext=[], params=None):
    n = len(knext)
    knext_box = knext[0:n]
    knext_dummy = knext[0:n]

    scaling_dept = (params.range_cube / (params.k_up - params.k_bar)
                    )  #scaling for kap

    #transformation onto cube [0,1]^d
    for i in range(n):
        #prevent values outside the box
        if knext[i] > params.k_up:
            knext_dummy[i] = params.k_up
        elif knext[i] < params.k_bar:
            knext_dummy[i] = params.k_bar
        else:
            knext_dummy[i] = knext[i]
        #transformation to sparse grid domain
        knext_box[i] = (knext_dummy[i] - params.k_bar) * scaling_dept

    return knext_box


def test_inf_nan(val):
    if sum(np.isinf(val)) > 0 or sum(np.isnan(val)) > 0:
        return True
    return False