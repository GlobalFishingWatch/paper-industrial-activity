This project provides a template and [cooresponding contribution and best practices guide](./CONTRIBUTE.md) for `python` projects. This repo is a python project template from
which you can create a new repo [instructions found in Setup](./CONTRIBUTE.md/#using-a-repository).

# What does this project do?

This template repo are meant to:

1. Make starting a new project easier. With a framework to start from, we hope this will make.
2. Help us be more consistent across projects. This allows new people (and your future self) to more efficiently orient themselves to your repo.
3. M∂ake hand-offs from research teams simpler
4. Be a gentle reminder of best practices at different stages of a project. Haven't used the `queries/` folder yet? Perhaps it's time to pull some queries out of those notebooks! Haven't put any code into your `pkg/<module>/` folder yet? Schedule some time to modularize your code.

The repo structure and a description of the intended use of each folder/file is provided in the *directory tree and structure and descriptions* section.

A slack channel [#prj-innovation-best-practices](https://globalfishingwatch.slack.com/archives/C02KM5XC9F0)
is available for any and all discussion around best practices, including repo
management. This channel is meant to be a place where people can ask for advice,
suggest a change in best practices, share a clever solution they came up with, share
a new tool they found... anything related to best practices in any language or
application!

# Who Uses this project?

The primary customers or users of this project are the Research team.

# How is this project setup for development?

The current iteration has dependencies and `pre-commit` hooks managed through
- `setup.cfg`
- `pyproject.toml`
- `setup.py`
- `.pre-commit-config.yaml`

# Setup this project

Unless you have changes, the instructions can be found [in contributions guide](./CONTRIBUTIONS.md/#building-conda-environment).

## Using the best practices template

This repository is intended to be used as a `template` or an `upstream`. The following steps will help you get started.

1. Create new repo with template by going to [the repo in GitHub](https://github.com/GlobalFishingWatch/research-python-template)
and clicking the `Use this template` button. You can also use the GitHub command line
tool, `gh repo create --template https://github.com/GlobalFishingWatch/research-python-template.git`
option. More documentation on this library is available [here](https://cli.github.com/manual/gh_repo_create).

**Expert `git` users may want to use this as an `upstream` instead of a `template`.**

2. Rename `research_pipeline_template` to your module name. The convention is to name
it the same as your repo, swapping any dashes for underscore.

1. Update the `[metadata]` section of `setup.cfg`. The `name` is typically the name of your new repository. You should only have to change the leading lines to get started.

```txt
[metadata]
name = research_python_template
version = 0.1.0
author = Jenn Van Osdel
author_email = jenn@globalfishingwatch.org
description = A python template repo for the Research and Innovation team at Global Fishing Watch
```

4. Note [CONTRIBUTE](./CONTRIBUTE.md) for best practices and instructions for how to setup your python environment

5. Update the `README.md` to what is relevant for your project.

6. Delete the `Using the best practices template` section of this `README.md` document. Finally, you can fill out the rest of this `README.md` to fit your project. Also, feel free to delete the contents of the `pkg` and `tests` directories, from your repository. Eventually your project should supplant this code.
