# InsurePredict-ML 🏥🤖

An end-to-end MLOps project for insurance cross-sell prediction using Machine Learning. Built with industry-standard tools and best practices.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.7-orange.svg)](https://mlflow.org/)

## 🎯 Project Overview

This project predicts whether insurance customers are likely to purchase additional insurance products (cross-selling) using machine learning with complete MLOps implementation.

**Business Problem:** Insurance companies want to identify customers most likely to buy additional policies to optimize marketing efforts.

**Solution:** ML model with 76% ROC-AUC score, deployed as REST API with full experiment tracking and data versioning.

## 🛠️ Tech Stack

### Core ML & Data
- **Python 3.9+** - Programming language
- **Scikit-learn** - Machine learning algorithms
- **Pandas & NumPy** - Data processing
- **RandomForest** - Classification model

### MLOps Tools
- **MLflow** - Experiment tracking & model registry
- **DVC** - Data version control
- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server

### Development & CI/CD
- **Git & GitHub** - Version control
- **GitHub Actions** - CI/CD automation
- **Docker** - Containerization
- **Pytest** - Unit testing

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 68.18% |
| Precision | 25.92% |
| Recall | 87.05% |
| F1 Score | 39.94% |
| ROC-AUC | 76.31% |

*Note: High recall prioritized to catch potential customers (imbalanced dataset: 88% class 0, 12% class 1)*

## 🚀 Getting Started

### Prerequisites
```bash
python --version  # 3.9 or higher
git --version
```

### Installation

1. **Clone repository**
```bash
git clone https://github.com/YOUR_USERNAME/InsurePredict-ML.git
cd InsurePredict-ML
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Pull data (DVC)**
```bash
dvc pull
```

## 📁 Project Structure
```
InsurePredict-ML/
├── src/
│   ├── components/          # ML components
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   └── model_trainer.py
│   ├── pipeline/            # Training & prediction pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── utils/               # Utility functions
│   │   └── common.py
│   ├── logger.py            # Logging configuration
│   └── exception.py         # Custom exception handling
├── data/                    # Dataset (tracked by DVC)
├── models/                  # Trained models
├── logs/                    # Application logs
├── mlruns/                  # MLflow experiments
├── tests/                   # Unit tests
├── app.py                   # FastAPI application
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
└── Dockerfile              # Container configuration
```

## 💻 Usage

### 1. Train Model
```bash
python src/pipeline/training_pipeline.py
```

**Output:**
- Trained model saved to `models/model.pkl`
- Preprocessor saved to `models/preprocessor.pkl`
- Metrics logged to MLflow

### 2. View Experiments (MLflow)
```bash
mlflow ui
```
Visit: `http://localhost:5000`

### 3. Run API Server
```bash
python app.py
```
Visit: `http://localhost:8000/docs` for interactive API documentation

### 4. Make Predictions

**Using Python:**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "Gender": "Male",
    "Age": 35,
    "HasDrivingLicense": 1,
    "RegionID": 28,
    "Switch": 0,
    "VehicleAge": "1-2 Year",
    "PastAccident": "No",
    "AnnualPremium": 30000,
    "SalesChannelID": 152,
    "DaysSinceCreated": 200
}

response = requests.post(url, json=data)
print(response.json())
```

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Age": 35,
    "HasDrivingLicense": 1,
    "RegionID": 28,
    "Switch": 0,
    "VehicleAge": "1-2 Year",
    "PastAccident": "No",
    "AnnualPremium": 30000,
    "SalesChannelID": 152,
    "DaysSinceCreated": 200
  }'
```

## 🐳 Docker Deployment
```bash
# Build image
docker build -t insurepredict-ml .

# Run container
docker run -p 8000:8000 insurepredict-ml
```

## 🧪 Testing
```bash
pytest tests/
```

## 📈 MLOps Features

### 1. Experiment Tracking (MLflow)
- Automatic logging of parameters, metrics, and models
- Compare multiple experiments
- Model versioning and registry

### 2. Data Version Control (DVC)
- Track dataset changes
- Reproducible data pipelines
- Collaborate on data like code

### 3. Logging
- Comprehensive logging at each step
- Timestamped log files in `logs/`
- Error tracking with line numbers

### 4. Error Handling
- Custom exception classes
- Detailed error messages with context
- Automatic error logging

## 🔄 CI/CD Pipeline

GitHub Actions workflow:
- Automated testing on push
- Code quality checks
- Docker image building

## 📚 Dataset

**Source:** Insurance Cross-Sell Prediction Dataset

**Features (10):**
- `Gender` - Male/Female
- `Age` - Customer age
- `HasDrivingLicense` - 0/1
- `RegionID` - Region code
- `Switch` - Previously switched policies (0/1)
- `VehicleAge` - < 1 Year / 1-2 Year / > 2 Years
- `PastAccident` - Yes/No
- `AnnualPremium` - Premium amount
- `SalesChannelID` - Channel code
- `DaysSinceCreated` - Customer vintage

**Target:**
- `Result` - 1 (will buy) / 0 (won't buy)

**Size:** 100,000 train + 100,000 test records

## 🏆 Key Learnings & Skills

- End-to-end ML pipeline development
- MLOps tools implementation (MLflow, DVC)
- API development with FastAPI
- Docker containerization
- CI/CD with GitHub Actions
- Handling imbalanced datasets
- Production-ready code structure
- Error handling and logging best practices

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset source: Insurance cross-sell prediction
- Inspired by real-world MLOps practices
- Built as part of MLOps learning journey

---

⭐ **If you found this helpful, please star the repo!**