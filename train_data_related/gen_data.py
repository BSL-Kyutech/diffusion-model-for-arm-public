from simulator import definition as armdef
from .input_search import yamanobori, gen_random_inputs
import pandas as pd


def gen_data(x, y):
    df = pd.DataFrame()
    #input_, x_, y_, theta = yamanobori(armdef.arm, x, y, 100)
    input_ = gen_random_inputs(armdef.arm)
    df = pd.DataFrame([input_])
    armdef.arm.calc(input_)
    df['x'] = armdef.arm.last.x[0][0]
    df['y'] = armdef.arm.last.x[0][1]
    df['theta'] = armdef.arm.last.x[1]
    return df
