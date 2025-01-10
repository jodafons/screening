
all: build_local
SHELL := /bin/bash


run_jupyter:
    jupyter-lab --no-browser --port ${DOCKER_EXPOSE} --NotebookApp.token='' --NotebookApp.password=''

clean:
	docker system prune -a
