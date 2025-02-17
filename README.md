# QML VIP Spring 25 Developer Guide

## Setting up your environment

Before you being, make sure that you have pipx and Poetry installed:

Instructions to install pipx: https://pipx.pypa.io/stable/installation/

Instructions to install Poetry: https://python-poetry.org/docs/

Once you have Poetry installed, clone this repo to your machine w/ `git clone`

Activate your environment: `poetry shell`

Once you have activated your environment, install the necessary requirements:
`poetry install --no-root` 

Once you have done all these steps, you are done setting up your environment

## Updating your environment

Once you pull from main, you will see that there have been new dependencies added to your pyproject.toml. 

Activate your environment and then run `poetry install`. Then run `poetry lock`. This should install all the new dependencies.

## Adding dependencies

Due to the nature of this project, you will likely test and add many new dependencies. In order to make sure that *everyone* has the correct dependencies installed, make sure to push only the updates you make to the `pyproject.toml` file to the `virtualenv` branch and then make a pull request. Do not push any code that you write to the virtualenv branch. It **will** be discarded.

## Project Structure:
