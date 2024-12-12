FROM python:3.12.4-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory into the /app directory in the container
COPY . /app

# Install the required Python packages from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for the application
EXPOSE 8000

# Set the default command to run the application using uvicorn
ENTRYPOINT ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
