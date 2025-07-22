FROM apache/airflow:2.7.3

USER airflow

COPY requirements.txt .
ARG AIRFLOW_VERSION=2.7.3
ARG PYTHON_VERSION=3.8.20
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

#RUN pip install --no-cache-dir -r requirements.txt -c ${CONSTRAINT_URL}

RUN pip install --no-cache-dir -r requirements.txt 


#RUN pip install --force-reinstall "pydantic<2.0.0"
# 2. Force install Altair and typing-extensions after Airflow is installed
#RUN pip install --upgrade altair typing-extensions