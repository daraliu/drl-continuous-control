from setuptools import setup, find_packages


setup(
    name="drl-continuous-control",
    version="0.1.0",
    packages=find_packages(),

    install_requires=[
        "numpy",
        "pandas",
        "plotnine>=0.2.0",
        "click>=7.0.0",
    ],
    author="Darius Aliulis",
    author_email="darius.aliulis@gmail.com",
    description="Deep Reinforcement Learning for Continuous Control",
    url="https://github.com/daraliu/drl-continuous-control",
    entry_points='''
        [console_scripts]
        drl-ctrl=drl_ctrl.cli:cli
    '''
)
