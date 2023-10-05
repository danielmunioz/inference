import os
from setuptools import setup, Extension

extensions = [
    Extension(
        "_compiler",
        [
            "_compiler.c",
        ]
    )
]


def main():
    setup(
        name="_compiler",
        version="1.0.0",
        description="package for ivy's graph compiler",
        ext_modules=extensions,
    )


if __name__ == "__main__":
    main()
