# Use an official Python image as the base image
FROM python:3.12-slim

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary tools: curl and sed
RUN apt-get update && apt-get install -y \
    curl \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Install img2dataset
RUN pip install img2dataset

# Set the working directory in the container
WORKDIR /app

COPY ./download_cc3m.sh /app/download_cc3m.sh

# Set default command (optional, change if you want a specific entry point)
CMD ["bash", "download_cc3m.sh"]
