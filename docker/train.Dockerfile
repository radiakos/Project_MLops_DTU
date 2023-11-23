FROM python:3.11.4

# Set working directory
WORKDIR /app

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY ../src/ src/
COPY ../setup.py setup.py
# RUN python setup.py
COPY ../requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
COPY ../data/ data/

# # or, DVC-adapted (mitsos)
# RUN cd /app && mkdir -p src/data && mkdir -p data/raw
# COPY .dvc /app/.dvc
# COPY data.dvc /app/data.dvc
# RUN dvc config core.no_scm true
# RUN dvc pull
# RUN python src/data/make_dataset.py

ENTRYPOINT [ "python", "-u", "src/models/model_train.py"]
# ENTRYPOINT [ "python", "-u", "src/models/model_train.py", "--arg1", "value1"]
# ENTRYPOINT [ "python", "-u", "src/models/test_docker.py"]