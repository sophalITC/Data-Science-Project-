# Use a base image with Python pre-installed
FROM python:3.10.6-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt requirements.txt

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port that your Flask app is running on
EXPOSE 8888

# Run the Flask app when the container launches
CMD ["python", "app.py"]