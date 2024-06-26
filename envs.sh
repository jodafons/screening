
# docker
export DOCKER_NAMESPACE=screening

#expose to jupyter
export DOCKER_EXPOSE=8080

# virtualenv name
export VIRTUALENV_NAMESPACE=screening-env

# git control
export GIT_SOURCE_COMMIT=$(git rev-parse HEAD)

# lab storage environs
export PROJECT_DIR=/mnt/brics_data
export DOCKER_STORAGE=$PROJECT_DIR

# output 
export TARGET_DIR=$PWD/targets

# mlflow tracking
export TRACKING_DIR=$PROJECT_DIR/tracking

# data input
export DATA_DIR=$PROJECT_DIR/datasets

# repo
export REPO_DIR=$PWD

# maestro environs
export DATABASE_SERVER_URL=$POSTGRES_SERVER_URL

# logger level
export LOGURO_LEVEL="INFO"

# dorothy token
export DOROTHY_TOKEN="b16fe0fc92088c4840a98160f3848839e68b1148"

#
# maestro
#
export SLURM_RESERVATION=joao.pinto_9



