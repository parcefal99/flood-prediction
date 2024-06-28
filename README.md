# flood-prediction

Flood prediction using LSTM and Deep Learning approaches

# Setup

## Virtual environment

Create a python virtual environment to install all the necessary packages:

```bash
python -m venv .venv
```

## Installation

Install the required packages into the created virtual environment

```bash
pip install -r requirements.txt
```

# Transfer data and start training with docker
Transfer data from the host machine to the remote (Note: run command in the host's terminal!)
```bash
rsync -azP /local/path/to/source/file user_name@server_ip:/remote/path/to/destination
```
Example:
```bash
rsync -azP /Users/abzal/Desktop/issai-srp/php03V9iD.png abzal_nurgazy@10.10.25.13:/raid/abzal_nurgazy/flood-prediction```
```
List available gpu index and its unique id
```bash
nvidia-smi --query-gpu=index,uuid --format=csv
```
To run the Docker container, use the following command pattern:
```bash
docker run --name container_name --gpus '"device=GPU-id"' --rm -v /local/path:/container/path --workdir /container/path image_name command
```
Example:
```bash
docker run --name test_run1 --gpus '"device=GPU-a6535fb0-896f-edf3-632a-c44f49ad8600"' --rm -v /raid/abzal_nurgazy/flood-prediction:/workspace \
--workdir /workspace flood-prediction python3 test_run.py```






