import pathlib

from drl_ctrl import config as cfg


def mk_path_weights_actor(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_WEIGHTS_ACTOR)


def mk_path_weights_critic(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_WEIGHTS_CRITIC)


def mk_path_scores(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_SCORES)


def mk_path_metadata(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_METADATA)
