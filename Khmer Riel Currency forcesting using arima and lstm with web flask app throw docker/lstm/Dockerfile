# Use an official Python runtime as the base image
FROM python:3.10.6-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the container
COPY . .

# Expose the port on which your Flask app will run
EXPOSE 5000

# Set the command to run your Flask app when the container starts
CMD ["python", "app_lstm.py", "--host=0.0.0.0"]
