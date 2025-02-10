# QML VIP Spring 25 Developer Guide

## Setting up your environment

Before you begin, make sure that you have pyenv installed. If you don't have it installed check it out [here]{https://github.com/pyenv/pyenv?tab=readme-ov-file#installation}. 

Before you begin setting up this project's environment, we will first need to install the correct version of python. We will be using python version 3.10.12. You install this with `pyenv install 3.10.12`.

Once you have pyenv installed, clone this repo to your machine w/ `git clone`

Initialize a virtual environment. You can name it whatever (ex: vipenv): `pyenv virtualenv 3.10.12 vipenv`

Activate your environment: `pyenv activate vipenv`

Once you have activated your environment, install the necessary requirements (which can be found in requirements.txt):
`pip install -r /path/to/requirements.txt`
Your `requirements.txt` file will be found in the main directory for this project. 

Once you have done all these steps, you are done setting up your environment

## Project Structure:
