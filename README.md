# X-123 Face Recognition Service

A FastAPI-based face recognition and avatar management service system.
Have fun!

## Project Overview

This project provides a complete face recognition and avatar management solution, including:
- Face registration and recognition
- Avatar management and processing
- RESTful API interface
- Support for various image processing features

## Tech Stack

- FastAPI
- OpenCV
- Milvus Vector Database
- Redis Cache
- MySQL Database
- DeepFace Recognition
- Python 3.x

## Prerequisites

Ensure your system has the following components installed:
- Python 3.x
- pip package manager
- Redis service
- MySQL database
- Milvus vector database

## Installation

1. Clone the project:
```bash
git clone [repository URL]
cd x-123
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Starting the Service

Use the following command to start the service:
```bash
uvicorn app:app --reload
```

After startup, access the service at:
- API Documentation: http://127.0.0.1:8000/docs
- Service Interface: http://127.0.0.1:8000

## API Documentation

### Avatar Management
- GET `/avatars/{filename}` - Get specific avatar
- GET `/users/{user_id}/avatars` - Get all avatars for a user
- POST `/users/{user_id}/avatar` - Add user avatar

### Face Registration and Recognition
- POST `/register/face/{user_id}` - Register user face
- POST `/register/avatar/{user_id}` - Register user avatar
- POST `/unregister/{user_id}` - Unregister user
- POST `/recognize/face/{user_id}` - Face recognition

## Project Structure

```
.
├── app.py              # Main application entry
├── service.py          # Core service logic
├── config.py           # Configuration file
├── requirements.txt    # Project dependencies
├── entity/            # Entity models
├── dao/               # Data access layer
├── server/            # Server-related code
└── volumes/           # Data storage directory
```

## Important Notes

- Ensure all required services (Redis, MySQL, Milvus) are properly configured and running
- Verify configuration file parameters are set correctly
- Be aware of image upload size limits and format requirements
