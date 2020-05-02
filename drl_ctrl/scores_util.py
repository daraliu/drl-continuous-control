import pathlib
from typing import Union

import pandas as pd
import plotnine as gg

from banana_nav import config as cfg, path_util


def read_scores(p: Union[pathlib.Path, str]):
    return pd.read_csv(path_util.mk_path_scores(p), index_col=None)


def plot_scores(df, title=None, xlab=None, ylab=None):
    g = (
        gg.ggplot(df, gg.aes(x=cfg.SCORE_COLNAME_X, y=cfg.SCORE_COLNAME_Y)) +
        gg.geom_line())
    if title is not None:
        g += gg.ggtitle(title)
    if xlab is not None:
        g += gg.xlab(xlab)
    if ylab is not None:
        g += gg.ylab(ylab)
    return g
