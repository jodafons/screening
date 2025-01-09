#!/bin/bash

mode=$1

# docker
export DOCKER_NAMESPACE=screening

# virtualenv name
export VIRTUALENV_NAMESPACE=screening-env

# lab storage environs
export PROJECT_DIR=/mnt/brics_data
export DOCKER_STORAGE=$PROJECT_DIR

# data input
export DATA_DIR=$PROJECT_DIR/datasets

# repo
export REPO_DIR=$PWD

export DATABASE_SERVER_URL=$POSTGRES_SERVER_URL

# logger level
export LOGURO_LEVEL="INFO"

# dorothy token
export DOROTHY_TOKEN="b16fe0fc92088c4840a98160f3848839e68b1148"



if [ -d "$VIRTUALENV_NAMESPACE" ]; then
    echo "$VIRTUALENV_NAMESPACE exists."
    source $VIRTUALENV_NAMESPACE/bin/activate
else
    make
    source $VIRTUALENV_NAMESPACE/bin/activate
fi

export PATH=$PATH:$REPO_DIR/scripts
export PYTHONPATH=$PYTHONPATH:$REPO_DIR

if [ "$mode" == "jupyter" ]; then
    jupyter-lab --no-browser --port ${DOCKER_EXPOSE} --NotebookApp.token='' --NotebookApp.password=''
fi




