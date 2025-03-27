# Data Scientist - AI Acceleration (AIA) Technical Assessment

This repository contains my solutions for the **Data Scientist - AI Acceleration (AIA) Technical Assessment** from Zero Margin Limited.

---

## Part 1: Structured Data & ML Challenge

### Dataset: `retail_sales_data.csv`
The dataset consists of daily sales records from a multi-store retail chain with features like store ID, product category, weather, promotions, customer demographics, and sales metrics.

### Tasks:
- **Data Preprocessing:** Cleaning, handling missing values, exploratory data analysis (EDA), and feature engineering.
- **Sales Forecasting:** Developing a model to predict daily `total_sales` for each `store-category` combination and generating a 14-day forecast for January 2024.
- **Customer Segmentation:** Clustering stores into meaningful groups and providing actionable insights.

### Deliverables:
- `structured_data_and_ml_challenge/Part_1.ipynb` – Jupyter notebook with well-documented code.
- `structured_data_and_ml_challenge/Part_1.doc` – Summary of findings and insights.

---

## Part 2: LLM & Vector Database Challenge

### Dataset: `product_reviews.csv` & `product_reviews.jsonl`
Contains customer reviews for electronic products with features like product name, category, ratings, review text, and sentiment labels.

### Tasks:
- **Document Processing Pipeline:** Extracting key information, generating embeddings, and implementing a vector search system.
- **LLM Application:** Utilizing a Large Language Model (LLM) for summarization, Q&A, and feature extraction.
- **Sentiment Analysis:** Building a classifier for sentiment detection and comparing results with labeled data.

### Deliverables:
- `llm_vector_database/document_processing_pipeline.py` – Code for extracting key information, generating embeddings, and implementing a vector search system.
- `llm_vector_database/llm_reviews_analysis.py` – Code for utilizing a Large Language Model (LLM) for summarization, Q&A, and feature extraction.
- `llm_vector_database/sentiment_analysis_and_classification.py` – Code for building a classifier for sentiment detection and comparing results with labeled data.
- `llm_vector_database/sentiment_visualization.py` – Code for visualizing sentimnet over time.


---

## Part 3: MLOps Design Exercise

### Scenario:
Design an MLOps architecture for deploying and managing three AI models:
1. **Demand Forecasting Model** – Predicts sales for product categories.
2. **Customer Segmentation Model** – Groups customers by purchasing behavior.
3. **Product Recommendation Engine** – Suggests items based on past purchases.

### Tasks:
- **MLOps Architecture:** Designing a scalable, reliable, and secure deployment strategy.
- **Data & Model Versioning:** Implementing version control for datasets and models.
- **Monitoring & Maintenance:** Setting up performance monitoring and alerting.
- **Documentation & Governance:** Ensuring compliance and tracking model changes.

### Deliverables:
- `mlops_design/mlops_design.md` – Detailed architecture and system design.


---

## Setup Instructions

### Prerequisites
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Python 3.11 is required

### Installation Steps
1. **Create a Conda environment**
   ```bash
   conda create --name aia_assessment python=3.11 -y
   conda activate aia_assessment
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**
   - Copy `env.example` to `.env`
   ```bash
   cp llm_vector_database/env.example llm_vector_database/.env
   ```
   - Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
