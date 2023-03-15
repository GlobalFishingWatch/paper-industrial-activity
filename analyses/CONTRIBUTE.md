# Project Template Repository for Research

This guide contains best practices for coding in a way that makes collaboration easier and
is easier to maintain and automate. These rules are intended to get
you successfully through the proof-of-concept phase and ready for your datasets, or model
to be turned into an automated prototype.

Although you should try to stick to these conventions wherever possible, feel free to add,
remove, or modify folders or add subfolder structure within folders as needed. However,
when you do stray from this structure, be sure to document changes in your README to that
those new to your repo can still quickly understand where to find things.

This repo template is meant to be taken in the spirit of the following Pep8 guidelines:

>*“Consistency within a project is more important. Consistency within one module or function is the most important. ... However, know when to be inconsistent -- sometimes style guide recommendations just aren't applicable. When in doubt, use your best judgment. Look at other examples and decide what looks best. And don't hesitate to ask!”*

Future you and everyone else who comes into contact with your code will thank you!

This document has a lot of information, but the priority reads are highlighted [here](#what-do-i-need).

# What do I Need to Read?

- Priority reads for Everyone
   - [what goes in a readme](#what-goes-in-a-readmemd)
   - [coding best practices](#coding-best-practices)
   - [capturing dependencies](#capturing-dependencies)
   - [variable and file naming conventions](#variable-and-file-naming-conventions)
   - [using doc-strings](#using-doc-strings)
   - [Google Big Query and Databases](#google-big-query-and-databases)
   - [Branching Conventions and Git Workflow](#branching-conventions-and-git-workflow)
   - [Template Directory Tree](#template-directory-tree-and-structure-descriptions)
- Priority reads for Deployment and Consumed Projects
   - [refactoring code](#refactoring-code)
   - [Using a repository](#using-a-repository)
   - [typing in Python with `mypy`](#typing-in-python-with-mypy)
   - [Google Big Query and Databases](#google-big-query-and-databases)
   - [Installable Packages](#installable-python-packages-and-setupcfg)
- Priority reads for Release
   - [Dockerization](#dockerization)
   - [Zenodo](#zenodo)

# How to do Documentation

## What goes in a `README.md`?

All these sections are required

- What does the code do
- Who uses this/depends on this
- How to install or setup environment
- How to run things
- How to make changes to the code

## Using doc-strings

Add documentation to functions in a way that it can be picked up by IDEs and other
documentation compilers. In Python, this is doc-strings and we will be using the
[Google docstring](https://google.github.io/styleguide/pyguide.html) convention.
In R, use [`roxygen2`](https://cran.r-project.org/web/packages/roxygen2/vignettes/roxygen2.html)
formatting. [In some cases doc-strings are not required, discussion found here.](#exception-cases-for-doc-strings)

Example of function signatures with comments

```python
def round_to(precision: float, n: float) -> float:
    '''
    given a non decimal precision, round to that

    It turns out that `round` doesn't do mathematical rounding. This
      implementation is simple and mirrors python's expected behavior.
        https://realpython.com/python-rounding/

    Example:
      round_to(.02, .119) == .12
      round_to(.005, .114) == .115

      round_to_002 = functools.partial(round_to, .002)

    Args:
        precision: non-decimal bound precision
        n: the number to be rounded

    Raises:
        Exception: cannot have zero or negative precision

    Returns:
        the rounded number, rounded to the nearest of the given precision

    '''
    if precision <= 0:
        raise Exception:
    p = len(str(precision).split(".")[-1])
    return round(floor(n / precision + 0.5) * precision, p)
```

<details>
  <summary>Exception cases for doc-strings</summary>

discussion started [in this slack thread](https://globalfishingwatch.slack.com/archives/C02KM5XC9F0/p1655130327770189)

Some functions derive less value in conforming to strict spec. This is often found in
model building with tools like pandas and tensorflow. These functions are likely to
change regularly where type annotations and descriptions are non-informative.

As an example, the dimension of a tensor is not captured with the type system but is
arguably the most important thing in communicating restriction of use cases for a
network layer.

As a second example, the purpose of a layer is often very specific and regularly not
transferable to other networks through simple imports. These minor changes end up
describing bespoke network layers and their documentation upkeep ends up being a
non-beneficial burden.

Possible solutions:
- use `noqa:` flags for `darglint`
- delete doc-strings for functions that are subject to this concern

</details>

<details>
  <summary>Using `darglint` for doc-strings</summary>
Documentaiton for `darglint` found https://github.com/terrencepreilly/darglint .

```console
darglint --docstring-style google --strictness full ./pkg
```

This can also be executed through pre-commit hooks. It is configured in [setup.cfg](./setup.cfg)
</details>


## Typing in Python with `mypy`

You can find documentation for types in the [mypy cheatsheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
and affiliated documentation. Below is a contained example hitting many use cases.
In all cases where types are used in a project `mypy` should be used to verify. It
can be installed then triggered manually, or uncommennted to be included as a [`pre-commit` hook](#pre-commit-hooks)

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from dataclasses_json import DataClassJsonMixin

class CredentialType(Enum):
    EODMS = 'eodms.json'
    ESA = 'scihub-login.json'
    GOOGLE_AUTH = 'google-auth.json'
    ASF = 'asf.json'

class MissingCredentialsFile(Exception):
    pass

@dataclass
class Credentials(DataClassJsonMixin):
    username: str
    password: str
    url: Optional[str] = None

class CredentialsGetter:
    def __init__(self, directory_path: Path):
        self._directory_path = directory_path

    def _credential_path(self, ct: CredentialType) -> Path:
        # docstring left out for brevity
        return self._directory_path / ct.value

    def get(self, ct: CredentialType) -> Credentials:
        # docstring left out for brevity
        with self._credential_path(ct).open() as f:
            return Credentials.from_json(f.read())
```

# Coding Best Practices

## Variable and File Naming Conventions

You can find a lot of useful examples here:

PEP8: https://peps.python.org/pep-0008/

Google: https://google.github.io/styleguide/pyguide.html

Variables and functions:

```python
class LongClassName:
    var_one: int = 1
    var_two: float = 3.14
    var_three: str = "string"

def long_function_name(var_one, var_two, var_three):
    var_one += var_two * var_three
    return var_one

long_variable_name = "some string"

LONG_CONSTANT_NAME = 3.1415927
```

Files and directories:

```console
long_program_name.py

long_file_name.yaml

LongNotebookName.py

long_directory_name

long-repository-name
```

## Jupyter Notebooks

It is common for very early and exploratory steps to be captured through a `jupyter`
notebook. It is important to be aware of certain coding practices even in these early
stages. Place notebooks in the `./notebooks` directory and update `setup.cfg` with
dependencies as discussed [here](#capturing-dependencies).

<details>
<summary> Golden Rule of Sharing Notebooks </summary>

Any notebook you post on Slack or share in any other capacity should run out of the box
and in order. It should also be clear what the code is doing, so add enough descriptive
comments to allow someone else to follow your analysis/methodology.

*Best Practice Tip:* clear the kernel and run the code from top to bottom before sharing.
```
Kernel -> Restart & Run All
```
</details>

## Jupytext

We use jupytext because it works better with git. We use `.gitignore` to block all
`jupyter` notebooks from being added to the repository.

```bash
$ jupytext --from ipynb --to "py:percent" --sync --pipe black ./notebooks/*.ipynb
```

<details>
  <summary>With Jupyter Notebooks</summary>
  <br>

**Installing**

```bash
pip install jupytext
```

**Setup**

* Check: *File→Jupytext→Pair notebook with percent Script*
* Uncheck: *File→Jupytext→Include metadata*

</details>

Note that notebook ipynb files will not be included in git by default, but the paired
percent script files, which work better with git, will be. This is insured by the
`.gitignore` file.

# CLI Arguments and Configs

We use [Hydra](https://hydra.cc/docs/intro/) for both defining configuration parameters and passing command-line arguments. You have two options.

## Option 1: Using a YAML file

Define your parameters in a YAML file inside a directory:

`config/myconfig.yaml`

```python
# Model params
model_param1: 1
model_param2: 3.14
model_param3: False

# Data params
data_path: gs://bucket/file.zarr
data_split: train/0

# Dir paths
run_dir: path/to/run/dir
src_dir: path/to/src/dirRun pip install -e . to install the module. This will create a folder titled <module>.egg-info that will allow you to access the code within your <module> folder from outside of that folder by doing import <module> without any need to use paths.

```

Then in your code:

`program.py`

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="myconfig")
def main(cfg: DictConfig) -> None:

    # All your functions go inside main()
    # All you params can be accessed as

    print(cfg.model_param1)
    print(cfg.data_path)
    print(cfg.run_dir)

if __name__ == "__main__":
    main()
```

Now you can pass any params through the command line:

```console
python program.py \
    model_param1=123 \
    data_path=gs://bucket/new/file.zarr \
    run_dir=path/to/new/run/dir
```

## Option 2: Using a Dataclass

Define your parameters in a Dataclass inside your Python program:

`program.py`

```python
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

@dataclass
class MyConfig:
    # model params
    model_param1: 1
    model_param2: 3.14
    model_param3: False
    # data params
    data_path: 'gs://bucket/file.zarr'
    data_split: 'train/0'
    # dir paths
    run_dir: 'path/to/run/dir'
    src_dir: 'path/to/src/dir'

cs = ConfigStore.instance()
cs.store(name="myconfig", node=MyConfig)

@hydra.main(config_path=None, config_name="myconfig")
def main(cfg: MyConfig) -> None:

    # All works in the same way as above

if __name__ == "__main__":
    main()
```
<hr>

# Capturing Dependencies


Capture dependencies by updating the *env.yaml* file in this directory.
Every time you install a package, using either *pip* or *conda*, add it to the *env.yaml* file.
For example, after executing `pip install numpy` and `conda install gdal`
the *env.yaml* file would look like:

    dependencies:
    # ...
    - conda-forge::gdal
    - pip:
      # ...
      - numpy
    name: MY_PACKAGE_NAME

<details>
  <summary>Optional</summary>
  <br>

You may also capture the version number in the dependencies. For example after
executing `pip install numpy==1.22.4` and `conda install gdal=3.5.0` the *env.yaml*
file would look like:

    dependencies:
    # ...
    - conda-forge::gdal=3.5.0
    - pip:
      # ...
      - numpy==1.2.2
    name: MY_PACKAGE_NAME

Capturing the version number is **optional** because it is typically easy to determine
later. **Before release or hand off you will need to specify all versions**

</details>

<details>
  <summary>Details</summary>
  <br>

  It is important to capture the dependencies needed to run your code so that
  others can get it running easily and reliably. Someone
  will need to figure out the dependencies to run your code and it's much
  easier for you to capture them as you go along than for you – or someone else –
  to come back and figure them out later.

  While capturing the packages names is always important, it is not typically
  critical to capture the version numbers early in development. However, as you
  approach deployment the package names should be supplemented with their versions
  to ensure that future runs can recreate your current results.

  See the section on [Understanding Version Numbers](#understanding-library-versions) for
  more discussion on how version numbers behave

  See the section on [Too Many Libraries](#too-many-libraries) for more discussion
  of capturing dependencies.

</details>

<hr>

## Specifying Dependencies for Installation

When you are read to create and installable repo,
add the dependencies in *env.yaml* to [*./setup.cfg*](./setup.cfg)
under the *install_requires* tag using `~=` to indicate the version number:

```txt
[options]
install_requires =
    # ...
    gdal~=3.5.0
    # ...
    numpy~=1.2.2
```

<details>
  <summary>Details</summary>
  <br>

  There are three files that specify the dependencies for installation.

  - `pyproject.toml` : list of system dependencies
  - `setup.py` : stub for working with pip
  - `setup.cfg` : list of library dependencies

  You should not generally need to modify the first two files. The third, [`./setup.cfg`](  ./setup.cfg), specifies library and version dependencies.

  The procedure described above will pin the dependencies to an exact version. This is
  useful for ensuring you can recreate previous results. However for some uses, particularly
  for libraries, it is better to specify just the minimum version using `>=` to allow easier
  upgrading.

  Note: there are rare cases where *conda* and *pip* have different names for the same
  package. In that case, you'll need to translate the *conda* name to the *pip* name
  when copying it to *setup.cfg*.

</details>

<hr>


# Refactoring Code

## Leveraging the power of python modules
1. Refactor code in your notebook into functions and make sure everything still runs. Give functions descriptive names and follow [style conventions](#variable-and-file-naming-conventions).
2. Pull those functions out into one or more files. See [Turning your repo into a module](#turning-your-repo-into-a-module) for more information.
3. Pull long queries out into separate, parameterizeable files. For Python, use `jinja2`. For `R`, use `glue` or `gfw_query` from `fishwatchr`.
4. Move all parameter definitions to the top of your notebook. You can skip ahead to [a more advanced configuration](#cli-arguments-and-configs) file approach if you'd like.
5. Update your notebooks to call module functions and read queries from separate files and enjoy how clean your notebook looks now!


## Leaving notebooks behind
Once you have core functions out and into a module, you should think about how you can get your code to where notebooks are used for presentation purposes only where all real work is done elsewhere. This means moving things like running data pipelines and models into regular python scripts that can be called from the command line with parameters from the command line or from a config file (see [CLI Arguments and Configs](#cli-arguments-and-configs))

Refactoring is an iterative and ongoing process, so you can (and should) start as early as possible. Once code has moved past early phase development and is becoming ready for collaboration or hand-off, it is important that it is refactored and kept to higher development standards moving forward.

<details>
<summary>migrating functions for better scoping</summary>
  Start strong by defining functions within your notebook as you develop it rather than refactoring all at once at the end. As you find yourself using a chunk of code more than once, put it into a function. When you are ready to move to a fully fledge python module, your functions will then be easy to pull out into separate files. It's also a good idea to continuously be moving parameters to the top of your file so that they are easily identitied when you are ready to move to using a config file.
</details>

<details>
<summary>Get rid of hardcoded parameters</summary>

Move parameters, including table names and hyper-parameters, out of the code. A good
first step to is to define parameters together at the top of your notebook as soon as
possible. The next step is to define them in a configuration file or a [dataclass](https://docs.python.org/3/library/dataclasses.html),
which is covered in a later slide.

*Best Practice Tip:* create a habit of consistently moving parameters out of code as
you write it.

</details>

## Automatic Code Formatting

We use [Black](https://github.com/psf/black) to format Python code, both scripts and
Notebooks.

The simplest way is to run the `black` tool from the command line:

Install:
```console
pip install black
```

Then run:
```console
black your_program.py
```

Alternatively, you can add `black` as a plugin to your code editor and Notebook, and
format your code as you develop:

* [Editor integration](https://black.readthedocs.io/en/stable/integrations/editors.html)
* [Notebook integration](https://github.com/drillan/jupyter-black)

## BigQuery for Python

General rules:
1. Don't use command line calls to `os`, use the [pandas-gbq](https://pandas-gbq.readthedocs.io/en/latest/) or [BigQuery python API](https://cloud.google.com/bigquery/docs/reference/libraries) libraries instead.
2. Include schemas and table descriptions for all tables that will be distributed or used by anyone else (see `Example Snippets` below for how to implement these).

<details>
<summary>Google Authentication</summary>

The first time you use BigQuery from python, you will need to authenticate to your GFW Google account by running `gcloud auth application-default login` in the terminal. This will open a browser where you can log in to your GFW Google account. More information on Google BigQuery Authentication can be found [here](https://cloud.google.com/bigquery/docs/authentication).
</details>
<br>
<details>
<summary>Example Snippets</summary>

For examples of how to create/delete BQ tables, add schemas and descriptions, format `jinja2` queries and more, see [SNIPPETS.md](./SNIPPETS.md).
</details>

# Using a Repository

This section explains how the template repository is organized and conventions for changing it.

As soon as you want to, start sharing your code in `GitHub` and not just by passing
it informally through Slack, start a dedicated repo for your project.

*Best Practice Tip:* if you already know your work will be its own project, datasets,
or model, start a repo at the very beginning before you begin coding.


## Branching Conventions and Git Workflow

By default we use a `develop` and `main` branching strategy for projects that need versions, and a
branch from main strategy for small prototype projects. This means that (wlg)
contributors will typically branch off `develop` with the branch naming convention
`<username>.<description>`. If for your project you choose a different branching
strategy add a note in the ./README.md

**Do not put `/` characters in your branch name. In some cases this breaks `git` functionality**

<details>
<summary>details</summary>

```console
git checkout develop                         # checkout develop branch, if using develop
git checkout -b probinso.change-description  # checkout and create new branch
```

After changes are commited, push your branch to the host and submit a pull requests.
Documentation found https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request .

It is polite, but not nessicary, to merge from `develop` before issuing a pull request.

```console
git commit --all --message "made changes A and B"  # commit all changes and add message
git merge origin develop                           # Optional, makes pull requests easier
git push origin probinso.change-description        # push changes to host
```

Your pull request should be against the `develop` branch. Then notify the primary
contributor or who you would like to do code review that the pull request is ready
for them. **Delete the branch after a pull request is merged and completed.**

To reduce friction primary contributors may choose to make changes directly to `develop`

The primary contributor for a repository will be responsable for updating version
numbers during a merge from `develop` into `main` as part of the  process.
</details>

<details>
   <summary>If you want a stronger understanding of git.</summary>

   It is strongly suggested to read about **Git Flow** and **Trunk-based Development**. Although we do not employ these strategies directly, they will help you to understand how `git` can be used. If you aren't interested in the design patterns of use for `git` then this is overkill. No particular sources are provided.

</details>

## Directory Naming Conventions

We use `snake_case_name` for all directories with *exception* of the root folder,
which is the repository name:

```console
your-repository-name
    ├── README.md
    ├── assets
    ├── data
    │   └── sub_folder_1
    │       └── sub_folder_2
    ├── docs
    ├── notebooks
    │   ├── HelloNotebook.py
    │   └── utility_folder
    ├── outputs
    │   └── figures
    ├── pkg
    │   └── pkg_name
    ├── tests
    │   └── pkg_name
    └── scripts
```

## Template directory tree and structure descriptions

This section describes the current template layout. Although you may not use all of this,
as a project progresses you should be able to take advantage of the structure

```
$ tree -a --charset ascii -L 3
.
|-- .darglint                 # configuration for darglint
|-- .gitignore                # ignored file patterns for git
|-- .mypy.ini                 # configuration for mypy
|-- .pre-commit-config.yaml   # configuration for pre-commit
|-- CONTRIBUTE.md             # best practices for contribution
|-- INSTALL-ENV.sh            # setup a repository environment for development
|-- README.md                 # description of project repository
|-- pyproject.toml            # system level dependencies (not likely to change)
|-- radenv.yaml               # base conda environment, difficult dependencies
|-- setup.cfg                 # project specific requirements and other configurations
|-- setup.py                  # dummy file, required for pip (do not change)
|-- assets                    # non-data asset files needed for execution
|   |-- .gitkeep
|   |-- example_schema.json
|   |-- fishing_hours.schema.json
|   |-- fishing_hours.sql.j2
|   `-- validate.sql.j2
|-- data                      # data asset files needed for execution
|   `-- .gitkeep
|-- docker                    # folder for Dockerfile instructions
|   `-- Dockerfile
|-- docs
|   `-- .gitkeep
|-- ide_setup
|   `-- sublime.md
|-- notebooks                 # notebook files treated with jupytext
|   `-- HelloNotebook.py
|-- outputs                   # output files are untracked, default output for hydra
|   `-- figures
|       `-- .gitkeep
|-- pkg                       # code for all package modules (only modules have __init__.py)
|   |-- module_name_a         # named module, if only one typically name of repository
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- __pycache__
|   |   |-- models
|   |   |-- pipeline.py
|   |   `-- utils
|   `-- module_name_b         # named module, example of second module in one package
|       `-- __init__.py
|-- scripts                   # script files similar to notebooks directory
|   `-- .gitkeep
|-- tests                     # tests written using pytest if needed
|   |-- .gitkeep
|   `-- module_name_a         # same name as module being tested
|       `-- test_utils.py
`-- untracked                 # files here you don't want public (like passwords)
    `-- passwords.txt         # example of a file that is ignored by .gitignore
    `-- .gitkeep

X directories, Y files
```

## `pre-commit` hooks

Our chosen `pre-commit` hooks are specified in this [file](./pre-commit-config.yaml).
Find the documentation for `pre-commit` [here](https://pre-commit.com/)
We have several hooks that are commented out, but can be enabled as needed.

```console
$ pip install pre-commit      # install pre-commit if missing (found in setup.cfg)
$ pre-commit install          # install hooks (found in pre-commit-config.yaml)
$ pre-commit run --all-files  # run all hooks on files in their current state
```

## Installable python packages and `setup.cfg`

<details>
    <summary>Why make installable packages?</summary>
        <br>
        Turning your repo into an installable python package allows for your core functions to be accessed throughout the repo without having to define paths. It will make your imports very clean and store core functionality in a central place so that multiple users of the repo can benefit from functions that may have previously lived only in one person's notebook.
        <br>
        <br>
        This also allows for collaborators to ensure analysis integrity by all using the same functions and having changes to those functions cascade to everyone at once so you are in sync.
</details>
<br>

Follow these steps to turn your repo into an installable python package:
1. Refactor some or all of repo (See [Refactoring code](#refactoring-code)).
2. Update your dependencies as needed (See [Capturing dependencies](#capturing-dependencies)).
3. Setup and activate the conda environment (See [Building conda environment](#building-conda-environment))
4. If you aren't using `INSTALL-ENVIRONMENT.sh`, build your environment then use `pip install -e .[all]`. This will create a folder titled `pkg.egg-info` and will allow you to access the code within your `pkg` folder from outside of that folder.

## `radenv` repository

There are a bunch of packages that are very difficult to install. We use `conda` over
`pip` to define dependency versions for these packages through a `radenv.yaml` file.
This allows us to share a common base environment with expected versions across
packages. Do not change this file if you can avoid it.

## Building conda environment

The executable setup script `./INSTALL-ENVIRONMENT.sh` will attempt to build the correct environments
for your development. It uses timestamps for most recent modification comparing recipie
files (like `setup.cfg` and `radenv.yaml`) and the creation dates of their respective
envioronments to choose which build steps are nessicary. This defaults to the base
environment specified by `radenv.yaml`.

If you are unable to run the `./INSTALL-ENVIRONMENT.sh` script, then you can read the script to
identify the critical steps. Please report errors to @probinso, so they can improve
functionality.

```bash
$ ./INSTALL-ENVIRONMENT.sh
environment [rad] is ready
environment [research_python_template] is ready
to activate: conda activate research_python_template

$ conda activate research_python_template
```

<details>
<summary> Using custom conda yaml files </summary>

You can override the base copnda environment by passing it to the script that has the
`name` variable set

```bash
$ cat tensorflow-env.yaml
name: tf-env

channels:
  - conda-forge
  - defaults

dependencies:
  - cudatoolkit~=11.2
  - tensorboard
  - tensorflow~=2.6.2
  - gdal~=3.5.0
  - geos~=3.11.0

$ ./INSTALL-ENVIRONMENT.sh tensforflow-env.yaml
environment [tf-env] is ready
environment [research_python_template] is ready
to activate: conda activate research_python_template
```
</details>

## Importing packages

Do not use local imports in a package's modules, always import from the base module

```python
import module_name_a.utils.datetime.as_date_str
# OR
from module_name_a.utils.datetime import as_date_str
# OR
import numpy as np
```

# Public Release Story

## Dockerization

There is an important slack discussion about research position on dockerization found
[here](https://globalfishingwatch.slack.com/archives/C02KM5XC9F0/p1657720775540319).

The goal of dockerization is to make handing a working project from one team to another
easy. It is most useful when the recieving team needs to run the project but has no
interest in making changes themself. This handoff can instead be done through making your
project an installable package, and that may be a useful step regardless.

<details>
<summary>details</summary>

If you are not yet handing a project off, then just capturing dependencies should be
sufficient for collaborative work.

The [Makefile](./Makefile) includes instructions for building a `project-dev` environment
in case you want to develop inside the docker image. This development strategy makes it
easier to verify that the project will work in the docker image that you share.

The build instructions for the docker image are found in the [Dockerfile](./docker/Dockerfile).
Here there are instructions for both a `project-dev` and a `project-exec`, based on a
`project-base`. This strategy allows you to capture image and project execution instructions
to the `project-base` section that will propogate to `-dev` and `-exec` cleanly.

</details>

Best Practice Tip: At minimum, Docker must be set up before the project can go into
the Prototype phase. However, the earlier you set it up, the better.

# Zenodo

GFW strives to make code publicly available whenever possible. This may include our custom software tools (e.g. 4Wings), data pipelines, or code underlying our scientific publications. To this end, GFW uses [Zenodo](https://zenodo.org/), a free and open digital archive for data and code, to formally publish a stable codebase. Zenodo provides all uploads with a Digital Object Identifier (DOI), to make them citable and trackable, and easily integrates with GitHub to preserve repositories. This will allow the public to cite - and GFW to track the use of - specific GFW codebases and datasets.

<details>
  <summary>Details</summary>
  Follow these steps to publish a repository on Zenodo:

  1. Follow [these instructions](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content) to log into Zenodo using your GitHub account and enable the target repository for archiving on Zenodo.
  2. Follow [these instructions](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) to create a "release" of the repository on GitHub. Zenodo will automatically download a `.zip-ball` of each release and register a DOI.
  3. Return to your [Zenodo GitHub Settings](https://zenodo.org/account/settings/github/) page. You should now see the repository listed under the **Enabled Repositories** section with a DOI badge.
  4. Click on the repository DOI badge and copy the provided Markdown.
  5. Return to the repository GitHub page and add the badge to the top of the repo README.md using the provided markdown.

  </details>


# Understanding Library Versions

Versions are often defined by `MajorRevision.MinorRevision.Bugfix`. Unfortunately this
is done by convention, not as a requirement, so its useful to know if this is the
versioning system being used. An easy way to relax requirements for dependencies, in
order to to capture all `bugfixes` without grabbing versions that may break expected
functionality is to use `numpy~=1.19.2` which bounds the library to `x>=1.19.2` and
`x<1.20.0`.

<details>
<summary>Dangers of being too specific with dependencies</summary>

If a project over-specifies by including too many libraries then this makes the install
too rigid. This rigidity may result in
- versions that aren't available for specific operating systems
- unknown downstream dependency versions, can prevent updates for all other libraries
</details>

<details>
<summary>Why its better to write your own dependencies file?</summary>

To automatically capture the dependencies of your `conda` environment the command

```
conda env export -n environment-name -f environment-name.yml
```

will extract all dependencies and save them out to the `environment-name.yml` file,
however this includes additional build information that is operating system specific.
For handing a project to engineering or between team members, it is possible the
operating system of the recipient will be different than your own. Anaconda's `export`
command will not provide the correct build numbers for your recipient's operating system.

You can attempt using the `--no-builds` flag

```
conda env export -n environment-name -f environment-name.yml --no-builds
```

which sometimes is sufficient, but may still cause problems because packages in `conda`
repositories may not have the same versions for every operating system setup.

</details>

# Google Big Query and Databases

GFW's other resource for GBQ is [here](https://github.com/GlobalFishingWatch/bigquery-documentation-wf827), and has many additional examples.

For research projects, append `.concept`, `.prot`, or `.prod` to new tables in order to
communicate to engineering the level of stability.

## Database Changes

It is likely that during early phases of development/analysis that tables are either
embedded in code, notebooks, or through personal wizard craft. In order to assure
reproducible for others be sure to document and include these changes.

## Creating Date Indexable Tables

It is suggested to partition large tables by month, to increase useability and search
time.

<details>
<summary>history</summary>
It used to be policy to shard tables by name through use of appending shard boundary
information as a suffix to the table name `examlple_table.10-2017`. As `google-big-query`
infrustructure has changed for most large date indexed data sets engineering has advised
that partitions are a better dividing strategy and that imperically partitioning by month
gives a very healthy size/speed use compromize.

This decision also helps engineering with `copy`, `move`, and other administrative
operations.

</details>


## Create Schemas for all Tables

Fields should have good descriptions and the table description should include all
input parameters and the name of the repo that generates the table. Once you have it
running in the command line, also include the command that was run to generate that
table.
