# FROM  nvcr.io/nvidia/pytorch:22.12-py3
FROM python:3.11.4
#change above with cuda to run with gpu

# Set working directory
WORKDIR /

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY setup.py setup.py
# RUN python setup.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
COPY data/ data/

ENV WANDB_API_KEY=a1582d7e00e1d4c88d9f547b9a755237ffa63871

# # or, DVC-adapted (mitsos)
# RUN cd /app && mkdir -p src/data && mkdir -p data/raw
# COPY .dvc /app/.dvc
# COPY data.dvc /app/data.dvc
# RUN dvc config core.no_scm true
# RUN dvc pull
# RUN python src/data/make_dataset.py

ENTRYPOINT [ "python", "-u", "src/models/model_run.py", "train=False", "--config-name", "models_config.yaml" ]
# ENTRYPOINT [ "python", "-u", "src/models/model_run.py", "--arg1", "value1"]
