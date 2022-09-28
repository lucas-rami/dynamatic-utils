#!/bin/bash

# Name of virtual environment
ENV_NAME="dynamatic-tools"

# Name of python executable that creates/launches the environment
# This will be the python version used within the environment too
PYTHON_EXEC="python3"

# Make sure pip and virtualenv (not conda!) are installed
if [ ! -d "$HOME/.virtualenvs/$ENV_NAME" ]; then
    $PYTHON_EXEC -m ensurepip
    $PYTHON_EXEC -m pip install virtualenv virtualenvwrapper
    mkdir -p "$HOME/.virtualenvs"
fi

# Creates environment variables for the virtual environent
export WORKON_HOME="$HOME/.virtualenvs"
export VIRTUALENVWRAPPER_PYTHON="$(which $PYTHON_EXEC)"
export VIRTUALENVWRAPPER_VIRTUALENV="$HOME/.local/bin/virtualenv"
. "$HOME/.local/bin/virtualenvwrapper.sh"

# Create (if doesn't exist) or launch (if exists) the virtual environment
if [ -d "$HOME/.virtualenvs/$ENV_NAME" ]; then
    workon $ENV_NAME
else
    mkvirtualenv -p $PYTHON_EXEC $ENV_NAME
fi