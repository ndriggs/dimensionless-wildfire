

from sympy import Matrix
from sympy.physics import units
from sympy.physics.units import temperature


def dimensional_matrix(d_):
    # fix the columns and the rows
    cols = [d for d in d_]
    data = {d: units_log(d_[d]) for d in cols}  # create a dict of dicts mapping variable to their dimensions
    rows = list({key for d in data for key in data[d].keys()})


    # add zeros for the missing dimensions
    for d in data:
        for row in rows:
            if row not in data[d].keys():
                data[d].update({row: 0})

        # drop the unit row, so that dimensionless variables appear in the nullspace
        del data[d][1]

    rows.remove(1)

    # make the matrix
    return Matrix([[data[var][dim] for var in cols] for dim in rows]), cols, rows


def units_log(x):
    # map a dimension variable to a dictionary of the form {dim1: pow1, dim2: pow2, ...}
    return {power.as_base_exp()[0]: power.as_base_exp()[1] for power in x._name.as_ordered_factors()}


def dimensionless_vars(d_):
    mat, cols, rows = dimensional_matrix(d_)
    return mat.nullspace(), cols, rows

