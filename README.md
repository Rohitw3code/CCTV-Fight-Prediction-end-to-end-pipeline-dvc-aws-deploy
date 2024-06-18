
**CCTV Fight Prediction End-to-End Pipeline**

**Project Overview**
=====================

This project aims to detect violence in CCTV camera footage using deep learning and video vision techniques. It uses a dataset of 100 videos each of non-violence and violence, with each video being 3 seconds long.

**System Requirements**
=====================

1. **Clone the Git Repository**
```bash
git clone https://github.com/rohitcode005/CCTV-Fight-Prediction-end-to-end-pipeline-dvc-aws-deploy.git
```
2. **Create Virtual Environment**
```bash
python -m venv venv
```
3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
4. **Run Setup Script**
```python
python setup.py install
```
5. **Place Kaggle API Key**
```bash
mkdir ~/.kaggle/
mv kaggle.json ~/.kaggle/
```
**Setup and Running the Pipeline**
================================

1. **Initialize DVC**
```bash
dvc init
```
2. **Run the Pipeline**
```python
dvc repro
```
or
```python
python run_pipeline.py
```
**Configuration Options**
=====================

* **MLFLOW_TRACKING_URI**: https://dagshub.com/rohitcode005/CCTV-Fight-Prediction-end-to-end-pipeline-dvc-aws-deploy.mlflow
* **MLFLOW_TRACKING_USERNAME**: rohitcode005
* **MLFLOW_TRACKING_PASSWORD**: (Leave blank)

**Additional Notes**
=====================

* This project uses Keras and MLflow for tracking model accuracy and hyperparameter tuning.
* This project uses DVC to automate the workflow.
* This project is designed to be deployed on AWS using DVC.