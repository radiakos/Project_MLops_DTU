FROM python:3.11.4
#change above with cuda to run with gpu
# Set working directory
WORKDIR /app

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

# or, DVC-adapted
RUN mkdir -p data/external
COPY .dvc/ .dvc/
COPY external.dvc /external.dvc
RUN dvc config core.no_scm true
RUN dvc pull
RUN python src/data/make_dataset.py
RUN python src/data/data_cleaning.py

# RUN dvc init --no-scm
# RUN dvc remote add -d myremote gs://birds_bucket/
# RUN dvc remote modify myremote url gs://birds_bucket/
# RUN export GOOGLE_APPLICATION_CREDENTIALS='dtumlops-374716-d8e76837973a.json'

# RUN dvc pull

ENTRYPOINT [ "python", "-u", "src/models/model_run.py"]
# ENTRYPOINT [ "python", "-u", "src/models/model_run.py", "--arg1", "value1"]



