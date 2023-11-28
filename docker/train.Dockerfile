FROM  nvcr.io/nvidia/pytorch:23.02-py3
#change above with cuda to run with gpu
# Set working directory

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY data/ data/
COPY src/ src/
COPY setup.py setup.py
# RUN python setup.py
COPY requirements.txt requirements.txt

WORKDIR /
#check cuda version

RUN pip install -r requirements.txt --no-cache-dir

ENV WANDB_API_KEY=a1582d7e00e1d4c88d9f547b9a755237ffa63871

RUN python src/data/data_cleaning.py
RUN python src/data/make_dataset.py

ENTRYPOINT [ "python", "-u", "src/models/model_run.py"]
# ENTRYPOINT [ "python", "-u", "src/models/model_run.py", "--arg1", "value1"]
