# 1. Use an official Python base image
FROM python:3.12-slim

# 2. Set environment variables
# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install system dependencies (including Tesseract as requested earlier)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory in the container
# This represents your 'veradoc-back' folder
WORKDIR /app

# 5. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire 'app' folder into the container
# We copy it into a folder named 'app' so 'from app.routers' works
COPY . .

# 7. Copy default database to final name with default data
RUN cp /app/db/veradoc-def.db /app/db/veradoc.db

# 8. Expose the port FastAPI runs on
EXPOSE 8808

# 9. Command to run the application
# We use the module path 'app.main:app' just like the -m flag
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8808"]