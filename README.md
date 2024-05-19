# Python

This is a personal template for Python projects. This template expects to use asdf and Poetry to manage Python version and dependencies.

## Prerequests

You must setup asdf in advance following [this guide](https://asdf-vm.com/guide/getting-started.html).

To install Python in asdf, you would need additional packages. If you are an ubuntu user, type the following command to install all required packages:
```bash
$ sudo apt update
$ sudo apt install make build-essential wget curl llvm \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev  \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Finally, install Python and Poetry in asdf as follows:
```bash
# add python
asdf plugin-add python
asdf install python 3.12.2

# add poetry
asdf plugin-add poetry
asdf install poetry 1.8.2
```

## How to start development

Take the following steps to start your development.
1. Edit pyproject.toml, e.g., rename the project name.
2. Run `poetry install --no-root` to install some common libraries.
3. [Optional] Run `git config commit.template .gitmessage` to use gitmessage for commit.
