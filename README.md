# Seminar Smart Grids WS26/27

This project contains the code developed while working on the Seminar Smart Grids WS26/27 at the Fernuniversität Hagen.

# Repo Setup

This repository is setup as a multirepo. In the `apps/` directory, multiple python projects, like `energy-consumption`, can be located. Everything is managed via `uv`, which needs to be installed on the machine ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)).

## Setup of one python project

Each python project in the `apps/` directory follows the same setup. The standard python module structure with `src/` and `tests/` folders can be found here, but also configuration files like the `pyproject.toml` file for each project are located here.

### Dependency Management

Dependencies are managed in the `pyproject.toml` file of each project. The list of dependencies is defined here. To support the dependency management, a `create_venv.sh` file is also present. Running the script in a terminal like 

```bash
./create_venv.sh
```

will create a virtual environment that contains all the pinned dependencies from the `requirements.txt` file. 

### Updating the dependencies

To add new dependencies, simply add the new package to the list of dependencies in the `pyproject.toml`. Afterwards, run the following command:

```bash
./create_venv.sh -u
```

Using the `-u` flag will cause the `requirements.txt` to be recompiled according to the dependencies defined in the `pyproject.toml`. Compiling the dependencies allows sharing of exact environments between developers through the git repository. Whenever updates are necessary, developers may recompile the depencies to allow propagating the update to others.

## Interactive `.py` files

Each python project may contain a `scripts/` directory, where interactive files are located. These interactive `.py` work similarly to jupyter notebooks in the sense that cells of code can be executed in a desired order, state is persisted and output is immediatly visible. Running these interactive files in VSCode requires the python and the jupyter extensions to be installed.

# ZTBus Dataset

Download the dataset from `https://www.research-collection.ethz.ch/entities/researchdata/61ac2f6e-2ca9-4229-8242-aed3b0c0d47c` and put it into the `datasets/` directory. The `metaData.csv` file contains metadata for each driving mission of the dataset. The zip files contain all values, recorded once per second in one csv file per mission. The paper can be found at `https://arxiv.org/pdf/2303.08667`, where e.g. the columns of the dataset are explained in detail.