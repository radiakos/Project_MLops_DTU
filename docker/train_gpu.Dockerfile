FROM  nvcr.io/nvidia/pytorch:23.02-py3
#change above with cuda to run with gpu
# Set working directory

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy necessary files to container
WORKDIR /
COPY requirements_gpu.txt /requirements.txt
COPY setup.py /setup.py
COPY src/ /src/
COPY data/ /data/
COPY dtumlops-406109-3703b69ca83d.json /dtumlops-406109-3703b69ca83d.json
COPY dtumlops-406109-ac8fa5c6b516.json /dtumlops-406109-ac8fa5c6b516.json
# RUN python setup.py

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -U dvc 
RUN pip install -U dvc[gs]

ENV WANDB_API_KEY=a1582d7e00e1d4c88d9f547b9a755237ffa63871

# DVC-adapted
RUN mkdir -p /data/external

COPY .dvc/ /.dvc/
COPY data/external.dvc /data/external.dvc
RUN dvc config core.no_scm true
RUN dvc remote modify --local remote_storage credentialpath "dtumlops-406109-ac8fa5c6b516.json"
RUN dvc pull

RUN python src/data/data_cleaning.py


ENTRYPOINT [ "python", "-u", "/src/models/model_run.py"]
# ENTRYPOINT [ "python", "-u", "src/models/model_run.py", "--arg1", "value1"]
#docker build -f docker/train_gpu.Dockerfile . -t train_local:test0
