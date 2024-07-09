# Flood Prediction Project

Flood prediction using LSTM and Deep Learning approaches.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

# Setup

## Virtual environment

Create a python virtual environment to install all the necessary packages:

```bash
python -m venv .venv
```

or 

```bash
conda create -n flood_proj python=3.10.12
```

## Installation

Install the required packages into the created virtual environment

```bash
pip install -r requirements.txt
```

## Data

To prepare the dataset see the [dataset readme file](https://github.com/LuftWaffe99/flood-prediction/tree/main/dataset/README.md).


# Server

## Transfer data and start training with docker

Transfer data from the host machine to the remote (Note: run command in the host's terminal!)

```bash
rsync -azP /local/path/to/source/file user_name@server_ip:/remote/path/to/destination
```

Example:

```bash
rsync -azP /Users/abzal/Desktop/issai-srp/php03V9iD.png abzal_nurgazy@10.10.25.13:/raid/abzal_nurgazy/flood-prediction
```

List available gpu index and its unique id

```bash
nvidia-smi --query-gpu=index,uuid --format=csv
```

## Training with Docker

To run the Docker container, use the following command pattern (Note: run using tmux!):

```bash
tmux new -s session_name
docker run --name container_name --gpus '"device=GPU-id"' --rm -v /local/path:/container/path --workdir /container/path image_name command
```

Example:

```bash
docker run --name test_run1 --gpus '"device=GPU-a6535fb0-896f-edf3-632a-c44f49ad8600"' --rm -v /raid/abzal_nurgazy/flood-prediction:/workspace \
--workdir /workspace flood-prediction python3 test_run.py
```

To see running processes in tmux. Use CTRL+B D to detach from the current session 

``` bash
tmux list-sessions
tmux attach-session -t session_name
```
