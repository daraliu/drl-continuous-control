import pathlib

import click
from unityagents import UnityEnvironment

from drl_ctrl import control


HERE = pathlib.Path(__file__).absolute().parent
DEFAULT_PATH_TO_REACHER20_ENV = HERE.parent.joinpath("Reacher20_Linux/Reacher.x86_64")


@click.group()
def cli():
    pass


@cli.command(
    "demo-reacher20",
    help="Run a demo of 20 Reacher agents - trained or random (if no model provided)")
@click.argument(
    "PATH_MODEL",
    required=False,
    type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.option(
    "--unity-reacher20-env", "-e",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    default=DEFAULT_PATH_TO_REACHER20_ENV,
    help=f"Path to Unity Reacher20 Environment executable, default: {DEFAULT_PATH_TO_REACHER20_ENV}")
def demo(path_model, unity_reacher_env):
    env = UnityEnvironment(file_name=str(unity_reacher_env))
    if path_model is None:
        click.echo("Using Random agent")
    else:
        click.echo(f"Loading trained agent model from {path_model.absolute()}")
    score = control.demo20(env, path_model)
    click.echo(
        f"Episode completed with {'random' if path_model is None else 'trained'} agent. "
        f"Score: {score:2f}")
    env.close()


