from setuptools import setup

setup(
    name='dsb3',
    version='0.1',
    entry_points={
        'console_scripts': [
            'dsb3 = dsb3.__main__:main',
        ],
    },
    packages=[
        'dsb3',
        'dsb3.steps',
    ],
)
