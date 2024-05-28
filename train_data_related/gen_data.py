from simulator import definition as armdef
from .input_search import yamanobori
import pandas as pd


def gen_data(x, y):
    df = pd.DataFrame()
    input_, x_, y_, theta = yamanobori(armdef.arm, x, y, 100)
    df = pd.DataFrame([input_])
    df['x'] = x_
    df['y'] = y_
    df['theta'] = theta
    return df
