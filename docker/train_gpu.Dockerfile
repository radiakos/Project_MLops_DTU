FROM  nvcr.io/nvidia/pytorch:23.02-py3
#change above with cuda to run with gpu
# Set working directory

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_gpu.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# RUN python setup.py


RUN pip install --upgrade pip

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -U dvc 
RUN pip install -U dvc[gs]

WORKDIR /
ENV WANDB_API_KEY=a1582d7e00e1d4c88d9f547b9a755237ffa63871

# or, DVC-adapted
RUN mkdir -p data/external
#COPY dtumlops-406109-ac8fa5c6b516.json dtumlops-406109-ac8fa5c6b516.json
#RUN dvc init --no-scm
#RUN dvc remote add -d remote_storage gs://dtu_mlops_special/
#RUN dvc remote modify remote_storage url gs://dtu_mlops_special/
#RUN export GOOGLE_APPLICATION_CREDENTIALS='dtumlops-406109-ac8fa5c6b516.json'

COPY .dvc/ .dvc/
COPY data/external.dvc data/external.dvc
RUN dvc config core.no_scm true
<<<<<<< HEAD
RUN dvc remote modify --local remote_storage credentialpath "dtumlops-406109-3703b69ca83d.json"
=======
RUN dvc remote modify --local remote_storage credentialpath "dtumlops-406109-3703b69ca83d.json'"
>>>>>>> 5f2e688e045d5a5d7a025818f66f90a939703eb7

RUN dvc pull

# RUN python src/data/data_cleaning.py
# RUN python src/data/make_dataset.py

ENTRYPOINT [ "python", "-u", "src/models/model_run.py"]
# ENTRYPOINT [ "python", "-u", "src/models/model_run.py", "--arg1", "value1"]
