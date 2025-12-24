# 🚗 Vehicle Insurance MLOps Project

**End to End Production Grade Machine Learning Pipeline**

This project demonstrates a complete **MLOps driven machine learning system** built for vehicle insurance data. It showcases how real world ML systems are designed, trained, validated, deployed, and automated using modern industry tools and best practices.

The goal of this project is to present a **production ready ML pipeline** that reflects how ML systems operate in real organizations, making it ideal for recruiters, hiring managers, and MLOps engineers reviewing the work.

---

## 📌 Key Highlights

* Modular and scalable project architecture
* End to end ML lifecycle from data ingestion to deployment
* Cloud native design using AWS services
* CI/CD automation using GitHub Actions and Docker
* MongoDB based data ingestion
* Production level logging, exception handling, and validation
* Model versioning and S3 based model registry
* Deployed prediction service with web interface

---

## 🧱 Project Architecture Overview

```
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Deployment
      ↓
CI/CD Automation (GitHub Actions + Docker + AWS)
```

---

## 📁 Project Setup and Structure

### Step 1: Project Template Generation

* Run `template.py` to generate the complete project structure
* Creates folders for components, configuration, pipelines, utilities, and artifacts

### Step 2: Package Management

* Local package setup using:

  * `setup.py`
  * `pyproject.toml`
* Detailed explanations available in `crashcourse.txt`

### Step 3: Environment Setup

Create and activate a virtual environment:

```
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```

Verify installed packages:

```
pip list
```

---

## 📊 MongoDB Setup and Data Management

### Step 4: MongoDB Atlas Configuration

* Create a MongoDB Atlas account
* Set up a free M0 cluster
* Create database user and allow network access from `0.0.0.0/0`
* Obtain MongoDB connection string for Python

### Step 5: Data Upload to MongoDB

* Create `notebook/` directory
* Add dataset and `mongoDB_demo.ipynb`
* Push data to MongoDB using PyMongo
* Verify data in Atlas under **Browse Collections**

---

## 📝 Logging, Exception Handling, and EDA

### Step 6: Logging and Exception Handling

* Centralized logging and custom exception modules
* Tested using `demo.py`

### Step 7: EDA and Feature Engineering

* Exploratory Data Analysis notebook
* Feature engineering pipeline aligned with model training flow

---

## 📥 Data Ingestion Pipeline

### Step 8: Data Ingestion

* MongoDB connection handled in `configuration/mongo_db_connections.py`
* Ingestion logic implemented in:

  * `data_access`
  * `components/data_ingestion.py`
* Configuration and artifact tracking via:

  * `entity/config_entity.py`
  * `entity/artifact_entity.py`

#### Environment Variable Setup

**Linux / Mac**

```
export MONGODB_URL="mongodb+srv://<username>:<password>@..."
```

**Windows PowerShell**

```
$env:MONGODB_URL="mongodb+srv://<username>:<password>@..."
```

---

## 🔍 Data Validation, Transformation, and Model Training

### Step 9: Data Validation

* Schema defined in `config/schema.yaml`
* Validation logic implemented in `utils/main_utils.py`

### Step 10: Data Transformation

* Feature transformation pipeline in `components/data_transformation.py`
* Custom estimator logic in `entity/estimator.py`

### Step 11: Model Training

* Training pipeline implemented in `components/model_trainer.py`
* Uses transformed datasets and custom estimators

---

## ☁️ AWS Setup for Model Evaluation and Deployment

### Step 12: AWS Configuration

* Create IAM user with required permissions
* Configure AWS credentials as environment variables

```
export AWS_ACCESS_KEY_ID="YOUR_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET"
```

* S3 bucket configuration stored in constants

### Step 13: Model Evaluation and S3 Integration

* S3 used as model registry
* Model push and pull logic implemented in:

  * `aws_storage`
  * `entity/s3_estimator.py`

---

## 🚀 Model Evaluation, Deployment, and Prediction

### Step 14: Model Evaluation and Model Pusher

* Best model selected based on evaluation metrics
* Deployed model stored in S3
* Prediction pipeline created for inference

### Step 15: Web Application

* Flask based API in `app.py`
* UI supported via `static/` and `template/` directories

---

## 🔄 CI/CD Automation

### Step 16: Docker and GitHub Actions

* Dockerized application using `Dockerfile`
* CI/CD pipeline implemented using GitHub Actions
* GitHub Secrets used:

  * `AWS_ACCESS_KEY_ID`
  * `AWS_SECRET_ACCESS_KEY`
  * `AWS_DEFAULT_REGION`
  * `ECR_REPO`

### Step 17: AWS EC2 and ECR

* EC2 instance used for deployment
* Docker installed on EC2
* EC2 configured as a self hosted GitHub Actions runner

### Step 18: Final Deployment

* Port `5080` opened in EC2 security group
* Application accessible at:

```
http://<public_ip>:5080
```

---

## 🛠️ Tools and Technologies Used

* Python
* MongoDB Atlas
* Scikit Learn
* Docker
* GitHub Actions
* AWS EC2
* AWS S3
* AWS ECR
* Flask

---

## 🎯 Project Workflow Summary

* Data Ingestion from MongoDB
* Schema Based Data Validation
* Feature Engineering and Transformation
* Model Training and Evaluation
* Model Versioning using AWS S3
* CI/CD Automation
* Cloud Deployment on AWS EC2

---

## 💬 Connect

If you found this project useful or want to discuss MLOps, ML engineering, or deployment strategies, feel free to connect.

⭐ If this repository helped you, consider giving it a star!

---



This README provides a structured walkthrough of the MLOps project, showcasing the end-to-end pipeline, cloud integration, CI/CD setup, and robust data handling capabilities.
