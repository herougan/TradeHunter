ARGS must be a dict with these keys:

default: IVarType.CONTINUOUS or DISCRETE then some number in range,
        otherwise default index of range e.g. 1
range: IVarType.CONTINUOUS or DISCRETE then [min, max],
        otherwise it is an array of possible values [x_0, ..., x_n]
step_size: only if IVarType.CONTINUOUS or DISCRETE
        (treated as 1 for ARRAY, ENUM etc.)
type: IVarType