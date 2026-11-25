from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "nequip/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="nequip",
    version=version,
    description="NequIP is an open-source code for building E(3)-equivariant interatomic potentials.",
    download_url="https://github.com/mir-group/nequip",
    author="Simon Batzner, Albert Musealian, Lixin Sun, Anders Johansson, Mario Geiger, Tess Smidt",
    python_requires=">=3.7,<3.11",
    packages=find_packages(include=["nequip", "nequip.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "spingnn-train = nequip.scripts.train:main",
            "spingnn-evaluate = nequip.scripts.evaluate:main",
            "spingnn-benchmark = nequip.scripts.benchmark:main",
            "spingnn-deploy = nequip.scripts.deploy:main",
            "spingnn-data-delta-mag = nequip.scripts.vasp_mag_delta_data:main",
            "spingnn-data-mlff = nequip.scripts.vasp_mlff:main",
            "spingnn-data-mlff-bec = nequip.scripts.vasp_mlff_bec:main",
        ]
    },
    install_requires=[
        "numpy>=1.20,<2",
        "ase",
        "pandas",
        "tqdm",
        "torch>=2.6,<3",
        "e3nn>=0.5.0,<0.5.1",
        "pyyaml",
        "contextlib2;python_version<'3.7'",  # backport of nullcontext
        'contextvars;python_version<"3.7"',  # backport of contextvars for savenload
        "typing_extensions;python_version<'3.8'",  # backport of Final
        "torch-runstats>=0.2.0",
        "torch-ema>=0.3.0",
        "opt_einsum",
    ],
    extras_require={
        "plotting": ["matplotlib", "SciencePlots"],
        "wandb": ["wandb"],
        "test": ["pytest", "scikit-learn"],
    },
    zip_safe=True,
)
