import pandas as pd
import numpy as np


def points_near(x, y, point_array, n_points=10):
    df = pd.DataFrame(point_array).rename(
        columns={0: "x", 1: "y"}
    )
    print(df)
