FROM continuumio/miniconda3:latest

COPY environment.yml /environment.yml
RUN conda env create -f /environment.yml
ENV PATH="/opt/conda/envs/bayes_spec-dev/bin:$PATH"
ENV CONDA_DEFAULT_ENV="bayes_spec-dev"
RUN echo "conda activate bayes_spec-dev" >> ~/.bashrc
RUN pip install bayes_spec
