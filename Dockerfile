FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.3-cuda12.1.1

# Set the working directory inside the container
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Update package lists
RUN apt-get update

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the container
COPY . /app

# Set the command to run when the container starts
ENTRYPOINT ["python"]