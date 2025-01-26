# Dockerfile for inference
# Start from a base Python image
FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Changing the path to the working directory in the container
WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy data processing folder to Docker
COPY data_process/ /app/data_process

# Copy models folder to Docker
COPY models/ /app/models

# Copy code for torch NN model creation
COPY modeling/ /app/modeling

# Copy the code itself
COPY inference /app/inference

COPY unittests/preinf_test.py /app/unittests/preinf_test.py
COPY unittests/after_inf_test.py /app/unittests/after_inf_test.py

# Copy useful scripts and configs
COPY utils.py /app
COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -m unittest discover -s unittests -p preinf_test.py

# Train the model
RUN python3 inference/run.py

RUN python3 -m unittest discover -s unittests -p after_inf_test.py
