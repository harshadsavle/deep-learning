# 100 Days of Machine Learning - Complete Notes

> Comprehensive video-wise notes from CampusX's 100 Days of Machine Learning playlist  
> Language: Hinglish (Hindi + English mix)

---

## Table of Contents

1. [Section 1: Introduction & Fundamentals (Videos 1-14)](#section-1-introduction--fundamentals)
2. [Section 2: Data Handling & EDA (Videos 15-22)](#section-2-data-handling--eda)
3. [Section 3: Feature Engineering (Videos 23-34)](#section-3-feature-engineering)
4. [Section 4: Missing Data & Outliers (Videos 35-45)](#section-4-missing-data--outliers)
5. [Section 5: Dimensionality Reduction - PCA (Videos 46-49)](#section-5-dimensionality-reduction---pca)
6. [Section 6: Linear Models & Regression (Videos 50-68)](#section-6-linear-models--regression)
7. [Section 7: Logistic Regression & Classification (Videos 69-79)](#section-7-logistic-regression--classification)
8. [Section 8: Decision Trees (Videos 80-88)](#section-8-decision-trees)
9. [Section 9: Ensemble Methods (Videos 89-105)](#section-9-ensemble-methods)
10. [Section 10: Clustering (Videos 106-114)](#section-10-clustering)
11. [Section 11: SVM (Videos 115-118)](#section-11-svm)
12. [Section 12: Naive Bayes (Videos 119-125)](#section-12-naive-bayes)
13. [Section 13: Advanced Topics (Videos 126-134)](#section-13-advanced-topics)

---

## Section 1: Introduction & Fundamentals

### Video 1: What is Machine Learning?

#### Definition
Machine Learning ek aisa approach hai jahan computer explicitly program kiye bina data se automatically patterns learn karta hai.

**Traditional Programming vs Machine Learning:**

| Traditional Programming | Machine Learning |
|------------------------|------------------|
| Rules + Data = Output | Data + Output = Rules |
| Programmer rules define karta hai | Computer khud rules discover karta hai |
| Static logic | Adaptive logic |

#### Key Concepts

**Machine Learning kab use karte hain:**
1. **Complex Rules** - Jab problem ke rules bahut complex ho (e.g., spam detection)
2. **Fluctuating Environment** - Jab rules time ke saath change ho (e.g., recommendation systems)
3. **Data Mining** - Jab large datasets se hidden patterns nikalne ho

**Examples:**
- Spam Classification - Email ke content se spam ya non-spam decide karna
- Image Classification - Photos ko categorize karna
- Recommendation Systems - Netflix, Amazon suggestions

#### History of ML's Rise
1. **Data Availability** - Internet se massive data generation
2. **Hardware Advancement** - GPUs, cloud computing
3. **Algorithm Development** - Deep Learning breakthroughs

---

### Video 2: AI vs ML vs DL

#### Relationship Hierarchy

```
┌─────────────────────────────────────────┐
│         Artificial Intelligence          │
│   ┌─────────────────────────────────┐   │
│   │       Machine Learning           │   │
│   │   ┌─────────────────────────┐   │   │
│   │   │     Deep Learning        │   │   │
│   │   └─────────────────────────┘   │   │
│   └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### Definitions

| Term | Definition | Example |
|------|------------|---------|
| **AI** | Machines jo intelligent behavior exhibit karti hain | Chatbots, Self-driving cars |
| **ML** | AI ka subset jo data se seekhta hai | Spam filters, Recommendations |
| **DL** | ML ka subset jo neural networks use karta hai | Image recognition, NLP |

#### Key Differences

**Symbolic AI (Expert Systems) Limitations:**
- Domain experts ki zaroorat
- Rules manually define karne padte hain
- Scalability issues
- Knowledge acquisition bottleneck

**ML Advantages over Symbolic AI:**
- Data se automatically learn karta hai
- Complex patterns detect kar sakta hai
- Scale kar sakta hai with more data

**DL Advantages:**
- Automatic feature extraction
- Better performance with large datasets
- Works well with unstructured data (images, text, audio)

---

### Video 3: Types of Machine Learning (Based on Supervision)

#### 1. Supervised Learning

**Definition:** Labeled data se seekhna - input aur expected output dono available hain.

**Sub-types:**

| Type | Output | Example |
|------|--------|---------|
| **Regression** | Continuous value | House price prediction |
| **Classification** | Discrete categories | Email spam/not spam |

**Key Points:**
- Training data mein labels (correct answers) hote hain
- Model label predict karna seekhta hai
- Most common type of ML

---

#### 2. Unsupervised Learning

**Definition:** Unlabeled data se patterns discover karna.

**Sub-types:**

| Type | Purpose | Example |
|------|---------|---------|
| **Clustering** | Similar items group karna | Customer segmentation |
| **Dimensionality Reduction** | Features reduce karna | PCA for visualization |
| **Anomaly Detection** | Unusual patterns find karna | Fraud detection |
| **Association Rule Learning** | Item relationships find karna | Market basket analysis |

**Key Points:**
- No labels in data
- Model khud patterns discover karta hai
- Exploratory analysis ke liye useful

---

#### 3. Semi-Supervised Learning

**Definition:** Kuch labeled + bahut saara unlabeled data use karna.

**Use Case:** Jab labeling expensive ya time-consuming ho.

**Example:** Medical imaging jahan expert annotation costly hai.

---

#### 4. Reinforcement Learning

**Definition:** Agent environment se interact karke rewards/penalties se seekhta hai.

**Components:**
- **Agent** - Learning entity
- **Environment** - Where agent operates
- **Actions** - What agent can do
- **Rewards** - Feedback signal

**Examples:**
- Game playing (AlphaGo, Chess)
- Robotics
- Autonomous vehicles

---

### Video 4: Batch (Offline) Machine Learning

#### Definition
Poora dataset ek saath use karke model train karna, then production mein deploy karna.

#### Process Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Collect   │ --> │    Train    │ --> │   Deploy    │
│    Data     │     │   (Offline) │     │  (Server)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

#### Characteristics

| Aspect | Description |
|--------|-------------|
| Training | Complete dataset pe offline |
| Deployment | Trained model server pe |
| Update | Periodic retraining required |
| Resource | Training time pe high compute |

#### Disadvantages
1. **Model becomes stale** - Time ke saath data distribution change hoti hai
2. **Periodic retraining needed** - Manual intervention required
3. **Not adaptive** - Real-time changes capture nahi kar paata

#### When to Use
- Data slowly changes
- Training can be done offline
- Production has limited compute

---

### Video 5: Online Machine Learning

#### Definition
Model ko incrementally train karna - small batches of data se continuously learn karna.

#### Process Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  New Data   │ --> │  Partial    │ --> │   Updated   │
│  (Batch)    │     │   Fit       │     │    Model    │
└─────────────┘     └─────────────┘     └─────────────┘
        ↑                                      │
        └──────────────────────────────────────┘
                    (Continuous Loop)
```

#### Key Concepts

**Learning Rate:**
- Controls how fast model adapts to new data
- High learning rate = rapid adaptation (may forget old patterns)
- Low learning rate = slow adaptation (stable)

**Out-of-Core Learning:**
- Jab data RAM mein fit nahi hota
- Data ko chunks mein process karna
- Online learning techniques use karte hain

#### Advantages
1. Continuous improvement
2. Adapts to changing data (concept drift handle kar sakta hai)
3. Memory efficient (purana data store karne ki zaroorat nahi)
4. Cost effective for large datasets

#### Challenges
1. **Stability** - Model unstable ho sakta hai
2. **Biased Data** - Bad data se model degrade ho sakta hai
3. **Monitoring Required** - Continuous performance tracking
4. **Concept Drift** - Data distribution change hoti hai

#### Libraries for Online Learning
- **Scikit-learn**: `partial_fit()` method
- **River**: Python library for online ML
- **Vowpal Wabbit**: Fast online learning

```python
# Example using partial_fit
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
for batch in data_batches:
    X_batch, y_batch = batch
    model.partial_fit(X_batch, y_batch, classes=[0, 1])
```

---

### Video 6: Instance-Based vs Model-Based Learning

#### Instance-Based Learning (Memorization)

**Definition:** Training examples yaad rakhna aur prediction time pe similarity measure karna.

**Process:**
1. Store all training data
2. Find similar instances for new input
3. Predict based on neighbors

**Example: K-Nearest Neighbors (KNN)**
- Query point ke k nearest neighbors find karo
- Majority voting (classification) ya average (regression)

**Characteristics:**
- Lazy learning (training fast, prediction slow)
- Memory intensive
- Works well with small datasets
- No explicit model

---

#### Model-Based Learning (Generalization)

**Definition:** Data se underlying pattern/rule seekhna as a mathematical model.

**Process:**
1. Select a model (e.g., linear equation)
2. Define cost function
3. Optimize parameters
4. Use model for predictions

**Examples:**
- Linear Regression: y = mx + b
- Logistic Regression
- Decision Trees
- Neural Networks

**Characteristics:**
- Eager learning (training takes time, prediction fast)
- Memory efficient (only model stored)
- Generalizes better
- Explicit model parameters

---

#### Comparison

| Aspect | Instance-Based | Model-Based |
|--------|---------------|-------------|
| Training | Fast (just store) | Slow (optimization) |
| Prediction | Slow (search) | Fast (compute) |
| Memory | High (store data) | Low (store model) |
| Interpretability | Low | Varies |
| Example | KNN | Linear Regression |

---

### Video 7: Challenges in Machine Learning

#### 1. Data-Related Challenges

**Insufficient Data:**
- ML models need lots of data
- Solution: Data augmentation, transfer learning

**Poor Quality Data:**
- Missing values
- Outliers
- Inconsistent data
- Solution: Data cleaning, preprocessing

**Non-Representative Data:**
- Sampling bias
- Training data doesn't represent real-world
- Solution: Stratified sampling, diverse data collection

**Irrelevant Features:**
- Too many features (curse of dimensionality)
- Noise features
- Solution: Feature selection, dimensionality reduction

#### 2. Model-Related Challenges

**Overfitting:**
- Model memorizes training data
- Poor generalization to new data
- Solution: Regularization, cross-validation, simpler models

```
Training Accuracy: 99%
Test Accuracy: 60%  ← Overfitting!
```

**Underfitting:**
- Model too simple
- Can't capture underlying patterns
- Solution: Complex models, more features

```
Training Accuracy: 55%
Test Accuracy: 50%  ← Underfitting!
```

#### 3. Deployment Challenges

**Model Drift:**
- Data distribution changes over time
- Model performance degrades
- Solution: Continuous monitoring, retraining

**Scalability:**
- Handling large-scale predictions
- Solution: Distributed computing, model optimization

---

### Video 8: Applications of Machine Learning

#### Industry-wise Applications

| Industry | Application | ML Type |
|----------|-------------|---------|
| **Healthcare** | Disease diagnosis, Drug discovery | Classification, Regression |
| **Finance** | Fraud detection, Credit scoring | Anomaly detection, Classification |
| **E-commerce** | Recommendations, Price optimization | Clustering, Regression |
| **Transportation** | Self-driving cars, Route optimization | Deep Learning, Reinforcement Learning |
| **Entertainment** | Content recommendations, Game AI | Clustering, Reinforcement Learning |
| **Manufacturing** | Quality control, Predictive maintenance | Anomaly detection, Regression |

#### Common ML Tasks

1. **Image Classification** - Photos categorize karna
2. **Object Detection** - Objects locate karna in images
3. **Natural Language Processing** - Text understanding
4. **Speech Recognition** - Audio to text
5. **Recommendation Systems** - Personalized suggestions
6. **Time Series Forecasting** - Future values predict karna

---

### Video 9: Machine Learning Development Life Cycle (MLDLC)

#### Phases

```
┌─────────────────────────────────────────────────────────┐
│                    MLDLC Phases                         │
├─────────────────────────────────────────────────────────┤
│  1. Problem Definition                                  │
│  2. Data Collection                                     │
│  3. Data Preparation (EDA + Feature Engineering)        │
│  4. Model Building                                      │
│  5. Model Evaluation                                    │
│  6. Model Deployment                                    │
│  7. Monitoring & Maintenance                            │
└─────────────────────────────────────────────────────────┘
```

#### Detailed Phases

**1. Problem Definition:**
- Business problem ko ML problem mein convert karna
- Success metrics define karna
- Feasibility analysis

**2. Data Collection:**
- Relevant data sources identify karna
- Data acquisition
- Data storage

**3. Data Preparation:**
- EDA (Exploratory Data Analysis)
- Data cleaning
- Feature engineering
- Data transformation

**4. Model Building:**
- Algorithm selection
- Training
- Hyperparameter tuning

**5. Model Evaluation:**
- Performance metrics calculate karna
- Cross-validation
- Comparison with baseline

**6. Model Deployment:**
- Model export
- API creation
- Integration with existing systems

**7. Monitoring & Maintenance:**
- Performance tracking
- Model retraining
- Handling model drift

---

### Video 10: Data Science Job Roles

#### Role Comparison

| Role | Focus | Skills |
|------|-------|--------|
| **Data Engineer** | Data infrastructure | SQL, ETL, Big Data tools |
| **Data Analyst** | Business insights | SQL, Excel, Visualization |
| **Data Scientist** | ML models | Python, Statistics, ML |
| **ML Engineer** | Production systems | MLOps, Software Engineering |

#### Detailed Descriptions

**Data Engineer:**
- Data pipelines build karta hai
- Data warehousing
- ETL processes manage karta hai
- Tools: Apache Spark, Airflow, Kafka

**Data Analyst:**
- Data se business insights nikalta hai
- Dashboards and reports create karta hai
- Stakeholders ko communicate karta hai
- Tools: SQL, Tableau, Power BI

**Data Scientist:**
- ML models develop karta hai
- Statistical analysis
- Experimental design
- Tools: Python, Scikit-learn, TensorFlow

**ML Engineer:**
- Models ko production mein deploy karta hai
- MLOps practices
- Model optimization
- Tools: Docker, Kubernetes, MLflow

---

### Video 11: What are Tensors?

#### Definition
Tensor ek multi-dimensional array hai jo numerical data store karta hai.

#### Tensor Types

| Dimensions | Name | Example |
|------------|------|---------|
| 0-D | Scalar | 5 |
| 1-D | Vector | [1, 2, 3] |
| 2-D | Matrix | [[1,2], [3,4]] |
| 3-D | 3D Tensor | Image (H x W x C) |
| n-D | n-D Tensor | Video (T x H x W x C) |

#### Visual Representation

```
Scalar (0D):  5

Vector (1D):  [1, 2, 3, 4, 5]

Matrix (2D):  [[1, 2, 3],
               [4, 5, 6]]

3D Tensor:    [[[1,2], [3,4]],
               [[5,6], [7,8]]]
```

#### Tensor Properties

**Shape:** Dimensions of tensor
```python
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)
```

**Rank/ndim:** Number of dimensions
```python
print(arr.ndim)  # 2
```

**Dtype:** Data type of elements
```python
print(arr.dtype)  # int64
```

#### Why Tensors in ML?
- Data representation (images, text, audio)
- Mathematical operations
- GPU acceleration
- Neural network computations

---

### Video 12: End-to-End Toy Project (Day 13)

#### Project Overview
Ek simple ML project banana - data collection se deployment tak.

#### Steps

**1. Problem Definition:**
- Kya predict karna hai?
- Dataset available hai?
- Success metric kya hai?

**2. Data Loading:**
```python
import pandas as pd

df = pd.read_csv('data.csv')
df.head()
```

**3. EDA:**
```python
# Basic statistics
df.describe()

# Check missing values
df.isnull().sum()

# Data visualization
import matplotlib.pyplot as plt
df['column'].hist()
plt.show()
```

**4. Feature Engineering:**
```python
# Handle missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['category'])
```

**5. Model Building:**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
```

**6. Evaluation:**
```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2: {r2_score(y_test, y_pred)}")
```

---

### Video 13: How to Frame a Machine Learning Problem

#### Steps to Frame ML Problem

**1. Understand Business Problem:**
- Stakeholders se baat karo
- Actual problem samjho
- Expected outcome define karo

**2. Convert to ML Problem:**
- Supervised vs Unsupervised?
- Classification vs Regression?
- What to predict (target variable)?

**3. Define Success Metrics:**

| Problem Type | Metrics |
|--------------|---------|
| Classification | Accuracy, Precision, Recall, F1 |
| Regression | MSE, RMSE, MAE, R² |
| Clustering | Silhouette Score |

**4. Data Requirements:**
- Kya data available hai?
- Kitna data chahiye?
- Data quality kaisi hai?

**5. Feasibility Check:**
- Is it solvable with ML?
- Do we have enough data?
- Is the ROI worth it?

#### Example: E-commerce Customer Churn

```
Business Problem: Customers leaving the platform
         ↓
ML Problem: Binary Classification
         ↓
Target: Will customer churn? (Yes/No)
         ↓
Features: Purchase history, engagement, demographics
         ↓
Metric: Recall (want to catch all churners)
```

---

### Video 14: Installing Anaconda & Jupyter Notebook Setup

#### Anaconda Installation

**Steps:**
1. Go to anaconda.com
2. Download Anaconda Individual Edition
3. Run installer
4. Follow installation wizard

#### Tools in Anaconda

| Tool | Purpose |
|------|---------|
| **Jupyter Notebook** | Interactive coding |
| **Spyder** | IDE for Python |
| **Anaconda Navigator** | GUI for managing packages |

#### Jupyter Notebook Basics

**Starting Jupyter:**
```bash
jupyter notebook
```

**Cell Types:**
- **Code Cell** - Python code execute karna
- **Markdown Cell** - Documentation likhna

**Keyboard Shortcuts:**
- `Shift + Enter` - Run cell and move to next
- `Enter` - Edit mode
- `Esc` - Command mode
- `A` - Insert cell above
- `B` - Insert cell below
- `M` - Convert to Markdown
- `Y` - Convert to Code

#### Virtual Environments

**Why Virtual Environments:**
- Project-specific dependencies
- Avoid package conflicts
- Easy deployment

**Creating Environment:**
```bash
conda create -n myenv python=3.9
conda activate myenv
```

**Installing Packages:**
```bash
conda install pandas numpy scikit-learn
# or
pip install package_name
```

#### Google Colab Alternative
- Free cloud-based Jupyter
- Free GPU access
- No installation required
- Direct Google Drive integration

```python
# Mount Google Drive in Colab
from google.colab import drive
drive.mount('/content/drive')
```

#### Kaggle Notebooks
- Free notebooks with datasets
- Community learning
- Competitions

---

## Section 2: Data Handling & EDA

### Video 15: Working with CSV Files

#### Loading CSV Files

**Basic Loading:**
```python
import pandas as pd

# From local file
df = pd.read_csv('data.csv')

# From URL
df = pd.read_csv('https://example.com/data.csv')
```

#### Important Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `sep` | Column separator | `sep='\t'` for TSV |
| `header` | Row to use as column names | `header=0` or `header=None` |
| `names` | Custom column names | `names=['col1', 'col2']` |
| `index_col` | Column for index | `index_col='id'` |
| `usecols` | Columns to load | `usecols=['col1', 'col2']` |
| `nrows` | Number of rows to read | `nrows=100` |
| `skiprows` | Rows to skip | `skiprows=[0, 2]` |
| `encoding` | File encoding | `encoding='latin-1'` |
| `na_values` | Values to treat as NaN | `na_values=['NA', '?']` |
| `dtype` | Column data types | `dtype={'col': 'int64'}` |
| `parse_dates` | Parse date columns | `parse_dates=['date']` |
| `chunksize` | Read in chunks | `chunksize=1000` |

#### Handling TSV Files
```python
# Tab-separated values
df = pd.read_csv('data.tsv', sep='\t')
```

#### When No Header Exists
```python
df = pd.read_csv('data.csv', header=None, 
                 names=['id', 'name', 'age'])
```

#### Reading Specific Columns
```python
df = pd.read_csv('data.csv', usecols=['name', 'age'])
```

#### Handling Large Files with Chunks
```python
# Process in chunks
chunks = pd.read_csv('large_data.csv', chunksize=10000)
for chunk in chunks:
    # Process each chunk
    process(chunk)
```

#### Encoding Issues
```python
# If UTF-8 doesn't work
df = pd.read_csv('data.csv', encoding='latin-1')
```

#### Handling Bad Lines
```python
df = pd.read_csv('data.csv', on_bad_lines='skip')
```

#### Data Type Specification
```python
df = pd.read_csv('data.csv', dtype={'target': 'int32'})
```

#### Parsing Dates
```python
df = pd.read_csv('data.csv', parse_dates=['date_column'])
```

#### Custom Converters
```python
def get_short_name(name):
    if name == 'Royal Challengers Bangalore':
        return 'RCB'
    return name

df = pd.read_csv('ipl.csv', converters={'team': get_short_name})
```

#### Handling Missing Values
```python
df = pd.read_csv('data.csv', na_values=['NA', '?', '-', 'missing'])
```

---

### Video 16: Working with JSON and SQL

#### JSON Files

**What is JSON:**
- JavaScript Object Notation
- Universal data format
- Similar to Python dictionary

**Reading JSON:**
```python
import pandas as pd

# From file
df = pd.read_json('data.json')

# From URL
df = pd.read_json('https://api.example.com/data')
```

**JSON Structure Example:**
```json
{
  "results": [
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"}
  ]
}
```

**Nested JSON Handling:**
```python
import json

with open('data.json') as f:
    data = json.load(f)
    
df = pd.DataFrame(data['results'])
```

---

#### SQL Databases

**Setup (Using XAMPP + MySQL):**
1. Install XAMPP
2. Start Apache and MySQL
3. Access phpMyAdmin at `localhost/phpmyadmin`
4. Create database and import data

**Connecting Python to MySQL:**
```python
# Install connector
# pip install mysql-connector-python

import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='mydb'
)
```

**Reading SQL Data:**
```python
import pandas as pd

# Using read_sql_query
df = pd.read_sql_query('SELECT * FROM users', conn)

# With filters
df = pd.read_sql_query('''
    SELECT * FROM users 
    WHERE age > 25
''', conn)
```

**Common SQL Operations:**
```python
# Select specific columns
df = pd.read_sql_query('SELECT name, age FROM users', conn)

# Join tables
df = pd.read_sql_query('''
    SELECT u.name, o.product
    FROM users u
    JOIN orders o ON u.id = o.user_id
''', conn)

# Aggregation
df = pd.read_sql_query('''
    SELECT country, COUNT(*) as count
    FROM users
    GROUP BY country
''', conn)
```

---

### Video 17: Fetching Data from an API

#### What is an API?
- Application Programming Interface
- Software-to-software communication
- Data pipeline between systems

**Real-World Example:**
```
Railway Ticket Booking:
IRCTC Database ←→ API ←→ MakeMyTrip App
                 ↕
              Yatra.com
```

#### Making API Requests

**Using requests library:**
```python
import requests
import pandas as pd

# API endpoint
url = 'https://api.example.com/movies?api_key=YOUR_KEY'

# Make request
response = requests.get(url)

# Check status
print(response.status_code)  # 200 = Success

# Get JSON data
data = response.json()
```

**Creating DataFrame from API:**
```python
# Extract relevant data
movies = data['results']

# Create DataFrame
df = pd.DataFrame(movies)

# Select columns
df = df[['id', 'title', 'release_date', 'vote_average']]
```

#### Pagination Handling
```python
all_data = pd.DataFrame()

for page in range(1, 429):  # Total pages
    url = f'https://api.example.com/movies?page={page}&api_key=KEY'
    response = requests.get(url)
    data = response.json()
    
    temp_df = pd.DataFrame(data['results'])
    all_data = pd.concat([all_data, temp_df], ignore_index=True)
```

#### API Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 404 | Not Found |
| 500 | Server Error |
| 401 | Unauthorized |

#### Saving Data
```python
# Save to CSV
df.to_csv('movies.csv', index=False)
```

---

### Video 18: Fetching Data using Web Scraping

#### What is Web Scraping?
Web scraping se aap kisi bhi website se data extract kar sakte ho jab API available nahi ho.

**When to Use:**
- API not available
- Data directly accessible nahi hai
- Website robots.txt mein allowed hai

#### Tools Required

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
```

#### Basic Web Scraping Process

**Step 1: Make HTTP Request**
```python
url = 'https://example.com/companies'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

response = requests.get(url, headers=headers)
print(response.status_code)  # 200 = Success
```

**Step 2: Parse HTML with BeautifulSoup**
```python
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.prettify())  # Formatted HTML
```

**Step 3: Extract Data**
```python
# Find all elements with specific tag
all_h2 = soup.find_all('h2')

# Find by class
ratings = soup.find_all('p', class_='rating')

# Find single element
name = soup.find('h2', class_='company-name')

# Extract text
company_name = name.text.strip()
```

#### Scraping Multiple Pages

```python
all_companies = []

for page in range(1, 334):  # Total pages
    url = f'https://example.com/companies?page={page}'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find company containers
    companies = soup.find_all('div', class_='company-card')
    
    for company in companies:
        name = company.find('h2').text.strip()
        rating = company.find('p', class_='rating').text.strip()
        
        all_companies.append({
            'name': name,
            'rating': rating
        })

# Create DataFrame
df = pd.DataFrame(all_companies)
```

#### Handling Access Denied (403 Error)

**robots.txt Check:**
- Visit `website.com/robots.txt`
- Check if scraping is allowed

**Solution: Add User-Agent Header**
```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
response = requests.get(url, headers=headers)
```

#### Using Browser Inspect Tool

1. Right-click on element → Inspect
2. Find HTML structure
3. Identify class names and tags
4. Use in BeautifulSoup selectors

#### Best Practices
- Respect robots.txt
- Add delays between requests (`time.sleep(1)`)
- Use proper headers
- Don't overload servers

---

### Video 19: Understanding Your Data

#### First Questions When You Get Data

Jab bhi naya dataset mile, yeh questions poocho:

**1. Data Size:**
```python
df.shape  # (rows, columns)
```

**2. Data Preview:**
```python
df.head()  # First 5 rows
df.tail()  # Last 5 rows
df.sample(5)  # Random 5 rows (better for biased data)
```

**3. Data Types:**
```python
df.info()
# Shows: column names, non-null counts, dtypes, memory usage
```

**4. Missing Values:**
```python
df.isnull().sum()  # Missing count per column
```

**5. Basic Statistics:**
```python
df.describe()  # Statistical summary for numerical columns
```

**6. Duplicate Check:**
```python
df.duplicated().sum()  # Number of duplicate rows
```

**7. Correlation:**
```python
df.corr()  # Correlation matrix
df['target'].corr(df)  # Correlation with target
```

#### Titanic Dataset Example

```python
# Data types
df.info()
# Numerical: PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare
# Categorical: Name, Sex, Ticket, Cabin, Embarked

# Missing values
df.isnull().sum()
# Age: 177 missing
# Cabin: 687 missing
# Embarked: 2 missing

# Statistics
df.describe()
# Age: mean=29.7, min=0.42, max=80

# Correlation with Survival
df.corr()['Survived']
# Pclass: -0.34 (negative correlation)
# Fare: 0.26 (positive correlation)
```

---

### Video 20: EDA using Univariate Analysis

#### What is Univariate Analysis?
Ek column ko independently analyze karna - ek variable ka distribution samajhna.

#### Data Types Review

| Type | Example | Analysis Method |
|------|---------|-----------------|
| **Numerical** | Age, Price | Histogram, Box plot |
| **Categorical** | Gender, City | Count plot, Pie chart |

---

#### Categorical Data Analysis

**Count Plot (Frequency):**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Count plot
sns.countplot(x='Survived', data=df)
plt.show()

# Alternative with value_counts
df['Survived'].value_counts().plot(kind='bar')
```

**Pie Chart (Percentage):**
```python
df['Pclass'].value_counts().plot(kind='pie', autopct='%.1f%%')
plt.show()
```

**Insights from Titanic:**
```
Survived: 0 (died) = 549, 1 (survived) = 342
More people died than survived

Pclass Distribution:
- Class 3: 55.1% (most passengers)
- Class 1: 24.2%
- Class 2: 20.7%
```

---

#### Numerical Data Analysis

**Histogram (Distribution):**
```python
import matplotlib.pyplot as plt

plt.hist(df['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Using seaborn
sns.histplot(df['Age'], bins=20)
```

**Distribution Plot (Histogram + KDE):**
```python
sns.displot(df['Age'], kde=True)
```

**KDE (Kernel Density Estimation):**
- Probability density function
- Shows probability of any value
- Smooth continuous curve

**Box Plot (5-Number Summary):**
```python
sns.boxplot(x=df['Age'])
```

**5-Number Summary:**
- Minimum (Q1 - 1.5*IQR)
- Q1 (25th percentile)
- Median (50th percentile)
- Q3 (75th percentile)
- Maximum (Q3 + 1.5*IQR)

**Detecting Outliers:**
- Points outside whiskers are potential outliers
- IQR = Q3 - Q1

**Statistics:**
```python
df['Age'].min()     # Minimum
df['Age'].max()     # Maximum
df['Age'].mean()    # Mean
df['Age'].median()  # Median
df['Age'].std()     # Standard Deviation
df['Age'].skew()    # Skewness (0 = symmetric)
```

---

### Video 21: EDA using Bivariate and Multivariate Analysis

#### Bivariate Analysis
Do columns ke beech relationship analyze karna.

**Three Combinations:**
1. Numerical vs Numerical
2. Categorical vs Numerical
3. Categorical vs Categorical

---

#### 1. Numerical vs Numerical

**Scatter Plot:**
```python
sns.scatterplot(x='total_bill', y='tip', data=tips)
```

**With Additional Dimension (Multivariate):**
```python
# Color by category
sns.scatterplot(x='total_bill', y='tip', hue='sex', data=tips)

# Color + Style
sns.scatterplot(x='total_bill', y='tip', hue='sex', style='smoker', data=tips)

# Color + Style + Size
sns.scatterplot(x='total_bill', y='tip', hue='sex', style='smoker', 
                size='size', data=tips)
```

---

#### 2. Categorical vs Numerical

**Bar Plot:**
```python
sns.barplot(x='Pclass', y='Age', data=titanic)
# Shows mean Age for each Pclass
```

**With Hue (Multivariate):**
```python
sns.barplot(x='Pclass', y='Age', hue='Sex', data=titanic)
```

**Box Plot Across Categories:**
```python
sns.boxplot(x='Sex', y='Age', data=titanic)

# With hue
sns.boxplot(x='Sex', y='Age', hue='Survived', data=titanic)
```

**Dist Plot Comparison:**
```python
# Survived vs Not Survived age distribution
sns.displot(titanic[titanic['Survived']==0]['Age'], kde=True, hist=False)
sns.displot(titanic[titanic['Survived']==1]['Age'], kde=True, hist=False)
```

**Insight:** Young children (Age < 10) had higher survival probability than dying probability.

---

#### 3. Categorical vs Categorical

**Cross-tabulation:**
```python
pd.crosstab(titanic['Pclass'], titanic['Survived'])
```

**Heatmap:**
```python
sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']), annot=True)
```

**Cluster Map:**
```python
sns.clustermap(pd.crosstab(titanic['SibSp'], titanic['Survived']))
# Groups similar values together
```

**Percentage Calculation:**
```python
# Survival rate by Pclass
titanic.groupby('Pclass')['Survived'].mean() * 100
# Class 1: 62.96%, Class 2: 47.28%, Class 3: 24.24%

# Survival rate by Sex
titanic.groupby('Sex')['Survived'].mean() * 100
# Female: 74.2%, Male: 18.9%
```

---

#### Pair Plot (Multiple Scatter Plots)

```python
# All numerical columns scatter plots
sns.pairplot(iris)

# With hue for categories
sns.pairplot(iris, hue='species')
```

**What Pair Plot Shows:**
- Diagonal: Histogram of each variable
- Off-diagonal: Scatter plot between pairs
- Color coding for different classes

---

#### Line Plot (Time Series)

**When to Use:** X-axis is time-based (date, year, month)

```python
# Group by year
yearly = flights.groupby('year')['passengers'].sum().reset_index()

# Line plot
sns.lineplot(x='year', y='passengers', data=yearly)
```

**Heatmap for Time Series:**
```python
# Pivot table
pivot = flights.pivot_table(values='passengers', index='month', columns='year')

# Heatmap
sns.heatmap(pivot, cmap='YlGnBu')

# Cluster map (groups similar patterns)
sns.clustermap(pivot)
```

---

### Video 22: Pandas Profiling (Automated EDA)

#### What is Pandas Profiling?
Ek library jo automatically EDA report generate karti hai - ek line of code mein.

#### Installation

```bash
pip install pandas-profiling
# or
pip install ydata-profiling  # Newer version
```

#### Basic Usage

```python
from pandas_profiling import ProfileReport
# or
from ydata_profiling import ProfileReport

# Generate report
profile = ProfileReport(df, title="Titanic Dataset Report")

# Save as HTML
profile.to_file("report.html")

# Display in notebook
profile.to_notebook_iframe()
```

#### Report Sections

**1. Overview:**
- Number of variables (columns)
- Number of observations (rows)
- Missing cells count
- Duplicate rows
- Memory size
- Variable types (numerical/categorical)

**2. Variables (Univariate Analysis):**
For each column:
- Distinct values
- Missing values
- Histogram/Distribution
- Statistics (mean, median, std, min, max)
- Quantile statistics
- Common values

**3. Interactions (Bivariate Analysis):**
- Scatter plots between pairs
- Correlation visualization

**4. Correlations:**
- Pearson correlation matrix
- Spearman correlation
- Heatmap visualization

**5. Missing Values:**
- Count per column
- Missing values matrix
- Missing values heatmap

**6. Sample:**
- First 5 rows
- Last 5 rows

#### Warnings Generated

```
- High cardinality (many unique values in categorical)
- Missing values percentage
- High correlation between features
- Uniform distribution
- Unique values (ID columns)
```

#### Example Output

```python
# Overview
Variables: 12
Observations: 891
Missing cells: 866 (8.1%)
Duplicate rows: 0

# Warnings
- Name has high cardinality (891 distinct values)
- Cabin has 77.1% missing values
- Age has 19.9% missing values
```

#### Best Practice
Jab bhi naya dataset mile, pehle pandas profiling report generate karo - bohot saara time bachega!

---

## Section 3: Feature Engineering

### Video 23: What is Feature Engineering?

#### Definition
Feature Engineering wo process hai jisme raw data ko ML model ke liye better input features mein transform karte hain.

#### Why Feature Engineering?
- Raw data directly use nahi ho sakta
- Better features = Better model performance
- Domain knowledge se insights extract karna
- Data ko ML-ready banana

#### Types of Feature Engineering

```
┌───────────────────────────────────────────────────┐
│             Feature Engineering                    │
├────────────────┬───────────────┬─────────────────┤
│  Transformation│   Selection   │   Creation      │
│  - Scaling     │  - Filter     │  - Domain-based │
│  - Encoding    │  - Wrapper    │  - Mathematical │
│  - Imputation  │  - Embedded   │  - Time-based   │
└────────────────┴───────────────┴─────────────────┘
```

#### Feature Engineering Pipeline

| Step | Process | Example |
|------|---------|---------|
| 1 | Missing Value Handling | Imputation |
| 2 | Feature Scaling | Standardization, Normalization |
| 3 | Encoding | Label, One-Hot |
| 4 | Transformation | Log, Power transform |
| 5 | Feature Creation | Polynomial features |
| 6 | Feature Selection | Remove irrelevant features |

---

### Video 24: Feature Scaling - Standardization

#### Why Feature Scaling?

ML algorithms jo distance-based hain (KNN, SVM, Neural Networks) unhe scaled features chahiye:
- Different features ki different ranges hoti hain
- Large values wale features dominate kar sakte hain
- Gradient descent faster converge karta hai

**Example:**
```
Age: 0-100 (small range)
Income: 0-10,00,000 (large range)
```
Without scaling, Income feature will dominate!

#### Standardization (Z-Score Normalization)

**Formula:**
```
z = (x - μ) / σ

Where:
- x = original value
- μ = mean of feature
- σ = standard deviation
```

**Result:**
- Mean = 0
- Standard Deviation = 1

#### When to Use Standardization?
- Data follows Gaussian (Normal) distribution
- Algorithms that assume normal distribution
- When outliers present (more robust than MinMax)

#### Implementation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler for test data!

# Check results
print(f"Mean: {X_scaled.mean():.2f}")  # ~0
print(f"Std: {X_scaled.std():.2f}")     # ~1
```

#### Important Points

**Fit and Transform:**
```python
# Training data - fit_transform
scaler.fit_transform(X_train)

# Test data - only transform (NEVER fit on test data!)
scaler.transform(X_test)
```

**Why?** Test data ke statistics use karne se data leakage hoti hai.

---

### Video 25: Feature Scaling - Normalization

#### Types of Normalization

| Method | Formula | Range | Use Case |
|--------|---------|-------|----------|
| **MinMax Scaling** | (x - min)/(max - min) | [0, 1] | Neural Networks |
| **MaxAbs Scaling** | x / max(\|x\|) | [-1, 1] | Sparse data |
| **Robust Scaling** | (x - median)/IQR | No fixed | Outliers present |

---

#### 1. Min-Max Scaling (Normalization)

**Formula:**
```
x_scaled = (x - x_min) / (x_max - x_min)
```

**Properties:**
- Values range: [0, 1]
- Preserves original distribution shape
- Sensitive to outliers

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

# Custom range [0, 5]
scaler = MinMaxScaler(feature_range=(0, 5))
```

**When to Use:**
- Data doesn't follow normal distribution
- Neural Networks (bounded activation functions)
- Image data (pixel values 0-255 → 0-1)

---

#### 2. Max Absolute Scaling

**Formula:**
```
x_scaled = x / max(|x|)
```

**Properties:**
- Values range: [-1, 1]
- Doesn't shift/center data
- Preserves sparsity (zeros remain zeros)

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
```

**When to Use:**
- Sparse data (many zeros)
- Data already centered at zero

---

#### 3. Robust Scaling

**Formula:**
```
x_scaled = (x - median) / IQR

Where:
- IQR = Q3 - Q1 (Interquartile Range)
```

**Properties:**
- Uses median and IQR (not mean/std)
- Robust to outliers
- No fixed range

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

**When to Use:**
- Outliers present in data
- When StandardScaler performs poorly

---

#### Choosing the Right Scaler

| Scenario | Recommended Scaler |
|----------|-------------------|
| Normal distribution | StandardScaler |
| Neural Networks | MinMaxScaler |
| Outliers present | RobustScaler |
| Sparse data | MaxAbsScaler |
| Tree-based models | No scaling needed! |

---

### Video 26: Encoding Categorical Data - Ordinal & Label Encoding

#### Why Encoding?
ML algorithms sirf numbers samajhte hain. Categorical data (text) ko numbers mein convert karna padta hai.

#### Types of Categorical Data

| Type | Has Order? | Example |
|------|------------|---------|
| **Nominal** | No | Gender (M/F), City names |
| **Ordinal** | Yes | Education (High School < Bachelor < Master) |

---

#### 1. Label Encoding

**What:** Har category ko ek integer assign karna.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Gender'])

# Female → 0, Male → 1
```

**Problem:** Creates false ordering (algorithm thinks Male > Female)

**When to Use:**
- Target variable encoding
- Tree-based models (can handle this)

---

#### 2. Ordinal Encoding

**What:** Categories ko meaningful order ke saath encode karna.

```python
from sklearn.preprocessing import OrdinalEncoder

# Define order
education_order = [['High School', 'Bachelor', 'Master', 'PhD']]

oe = OrdinalEncoder(categories=education_order)
df['Education_encoded'] = oe.fit_transform(df[['Education']])

# High School → 0, Bachelor → 1, Master → 2, PhD → 3
```

**When to Use:**
- Ordinal categorical data
- When order matters

#### Handling Unknown Categories

```python
oe = OrdinalEncoder(handle_unknown='use_encoded_value', 
                    unknown_value=-1)
```

---

### Video 27: One Hot Encoding

#### Why One Hot Encoding?
Label encoding mein false ordering aata hai. One Hot Encoding se har category ek separate column ban jata hai.

#### How It Works

```
Original:          One Hot Encoded:
┌────────────┐     ┌───────┬───────┬───────┐
│  Color     │     │ Red   │ Green │ Blue  │
├────────────┤     ├───────┼───────┼───────┤
│  Red       │  →  │   1   │   0   │   0   │
│  Green     │  →  │   0   │   1   │   0   │
│  Blue      │  →  │   0   │   0   │   1   │
│  Red       │  →  │   1   │   0   │   0   │
└────────────┘     └───────┴───────┴───────┘
```

#### Implementation

**Using Pandas:**
```python
df_encoded = pd.get_dummies(df, columns=['Color'])
```

**Using Scikit-learn:**
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, drop='first')
encoded = ohe.fit_transform(df[['Color']])

# drop='first' removes one column (dummy variable trap)
```

#### Dummy Variable Trap
- If you have n categories, you need only n-1 columns
- One column can be inferred from others
- Causes multicollinearity in linear models

```python
# Red=1, Green=1, Blue=1 → Only need 2 columns
# If Red=0 and Green=0, then Blue must be 1
ohe = OneHotEncoder(drop='first')
```

#### When NOT to Use One Hot Encoding

| Issue | Explanation |
|-------|-------------|
| High Cardinality | 100+ categories = 100+ columns |
| Tree-based models | Can use Label Encoding |
| Sparse data issues | Too many zeros |

#### Alternatives for High Cardinality
- Target Encoding
- Frequency Encoding
- Feature Hashing

---

### Video 28: Column Transformer

#### Problem
Different columns need different transformations:
- Numerical → Scaling
- Categorical → Encoding

#### Solution: ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define transformers
ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'salary']),  # Numerical cols
        ('cat', OneHotEncoder(), ['gender', 'city'])   # Categorical cols
    ],
    remainder='passthrough'  # Keep other columns as-is
)

X_transformed = ct.fit_transform(X)
```

#### Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `transformers` | List of (name, transformer, columns) | - |
| `remainder` | What to do with unspecified columns | 'drop', 'passthrough' |
| `sparse_threshold` | When to use sparse matrix | 0.3 (default) |

#### Example with Multiple Transformations

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

ct = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), ['age', 'income']),
        ('ohe', OneHotEncoder(drop='first'), ['gender']),
        ('ordinal', OrdinalEncoder(), ['education'])
    ],
    remainder='drop'
)
```

#### Accessing Feature Names (After Transformation)

```python
# Get feature names
ct.get_feature_names_out()
```

---

### Video 29: Machine Learning Pipelines

#### What is Pipeline?
Pipeline ek sequence hai of data processing steps + model training.

#### Why Pipelines?
1. **Clean Code** - Organized workflow
2. **No Data Leakage** - Proper fit/transform handling
3. **Easy Deployment** - Single object to save/load
4. **Cross-validation** - Works seamlessly
5. **Hyperparameter Tuning** - All steps in one place

#### Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Training
pipe.fit(X_train, y_train)

# Prediction
predictions = pipe.predict(X_test)
```

#### Pipeline with ColumnTransformer

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'fare']),
        ('cat', OneHotEncoder(), ['sex', 'embarked'])
    ]
)

# Full pipeline
pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier())
])

pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

#### Accessing Pipeline Steps

```python
# Access scaler
pipe.named_steps['scaler']

# Access model parameters
pipe.named_steps['classifier'].feature_importances_
```

#### Pipeline with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Parameter grid (use step_name__param_name)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
```

#### Saving and Loading Pipeline

```python
import pickle

# Save
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Load
with open('pipeline.pkl', 'rb') as f:
    loaded_pipe = pickle.load(f)
```

---

### Video 30: Function Transformer (Mathematical Transformations)

#### Why Mathematical Transformations?

**Goal:** Data ko Normal Distribution mein convert karna.

Statistical algorithms (Linear Regression, Logistic Regression) perform better on normally distributed data.

#### How to Check if Data is Normal?

1. **Histogram/PDF Plot:**
```python
import seaborn as sns
sns.histplot(df['column'], kde=True)
```

2. **Skewness Check:**
```python
df['column'].skew()
# 0 = symmetric (normal)
# Positive = right skewed
# Negative = left skewed
```

3. **Q-Q Plot (Most Reliable):**
```python
import scipy.stats as stats
stats.probplot(df['column'], dist="norm", plot=plt)
```
- Points on diagonal line = Normal distribution

---

#### Types of Transformations

| Transform | Formula | Use Case |
|-----------|---------|----------|
| Log | log(x) | Right-skewed data |
| Reciprocal | 1/x | Specific distributions |
| Square | x² | Left-skewed data |
| Square Root | √x | Count data |

---

#### Log Transform

**When to Use:** Right-skewed data

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Create transformer
log_transformer = FunctionTransformer(np.log1p)  # log(1+x) handles zeros

# Apply
X_transformed = log_transformer.fit_transform(X)
```

**Why log1p instead of log?**
- `np.log(0)` = -inf (error!)
- `np.log1p(0)` = log(1) = 0 ✓

**Effect:**
- Large values get compressed
- Skewed distribution becomes more symmetric

---

#### Square Root Transform

```python
sqrt_transformer = FunctionTransformer(np.sqrt)
X_transformed = sqrt_transformer.fit_transform(X)
```

**When to Use:** 
- Count data
- Left-skewed data (sometimes)

---

#### Reciprocal Transform

```python
def reciprocal(x):
    return 1 / (x + 0.01)  # Add small value to avoid division by zero

reciprocal_transformer = FunctionTransformer(reciprocal)
```

---

#### Custom Transformation

```python
# Any custom function
def custom_transform(x):
    return x ** 2 + 2*x

transformer = FunctionTransformer(custom_transform)
```

---

#### Example: Comparing Transformations

```python
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# No transformation
model = LogisticRegression()
scores_original = cross_val_score(model, X, y, cv=5)
print(f"Original: {scores_original.mean():.3f}")

# Log transformation
log_t = FunctionTransformer(np.log1p)
X_log = log_t.fit_transform(X)
scores_log = cross_val_score(model, X_log, y, cv=5)
print(f"Log: {scores_log.mean():.3f}")
```

---

### Video 31: Power Transformer (Box-Cox & Yeo-Johnson)

#### What is Power Transformer?

Power Transformer automatically finds the best transformation to make data more Gaussian-like.

**Formula:**
```
y = (x^λ - 1) / λ,  if λ ≠ 0
y = log(x),         if λ = 0
```

Where λ (lambda) is learned from data.

---

#### Box-Cox Transform

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='box-cox')
X_transformed = pt.fit_transform(X)
```

**Limitations:**
- Only works with **strictly positive** values (x > 0)
- No zeros, no negatives

---

#### Yeo-Johnson Transform

```python
pt = PowerTransformer(method='yeo-johnson')  # Default
X_transformed = pt.fit_transform(X)
```

**Advantages:**
- Works with positive, negative, and zero values
- More flexible than Box-Cox
- Default in sklearn

---

#### Comparing Box-Cox and Yeo-Johnson

| Feature | Box-Cox | Yeo-Johnson |
|---------|---------|-------------|
| Positive values | ✓ | ✓ |
| Zero values | ✗ | ✓ |
| Negative values | ✗ | ✓ |
| Default in sklearn | No | Yes |

---

#### How Power Transformer Works

1. For each feature, find optimal λ
2. Apply transformation with that λ
3. Optionally standardize output

```python
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_transformed = pt.fit_transform(X)

# Get lambda values for each feature
print(pt.lambdas_)
```

---

#### Example

```python
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Load data
X, y = load_data()

# Without transformation
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Original R²: {scores.mean():.3f}")  # 0.46

# With Box-Cox (if all positive)
pt = PowerTransformer(method='box-cox')
X_transformed = pt.fit_transform(X + 0.01)  # Add small value for zeros
scores = cross_val_score(model, X_transformed, y, cv=5, scoring='r2')
print(f"Box-Cox R²: {scores.mean():.3f}")  # 0.60

# With Yeo-Johnson
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)
scores = cross_val_score(model, X_transformed, y, cv=5, scoring='r2')
print(f"Yeo-Johnson R²: {scores.mean():.3f}")  # 0.61
```

---

#### When to Use

| Algorithm | Need Transformation? |
|-----------|---------------------|
| Linear Regression | Yes |
| Logistic Regression | Yes |
| SVM | Yes |
| Neural Networks | Yes |
| Decision Trees | No |
| Random Forest | No |
| Gradient Boosting | No |

---

### Video 32 & 33: Binning and Binarization (Discretization)

#### What is Discretization?

Numerical data ko categorical mein convert karna by creating bins (intervals).

**Use Cases:**
- Outlier handling
- Non-linear relationships capture karna
- Feature simplification

---

#### Types of Binning

| Type | Method | Use Case |
|------|--------|----------|
| **Uniform/Equal Width** | Same width intervals | General use |
| **Quantile** | Equal frequency | Skewed data |
| **K-Means** | Cluster-based | Clustered data |
| **Custom** | Domain knowledge | Specific requirements |

---

#### 1. Uniform (Equal Width) Binning

Each bin has same width: `width = (max - min) / n_bins`

```python
from sklearn.preprocessing import KBinsDiscretizer

# Equal width binning
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_binned = kbd.fit_transform(X)
```

**Example:**
```
Data: [1, 5, 10, 15, 20, 25, 30, 35, 40, 100]
Bins: [1-20], [20-40], [40-60], [60-80], [80-100]

Result: Most values in first bin, last value alone in last bin
```

**Pros:** Simple
**Cons:** Doesn't handle outliers well

---

#### 2. Quantile (Equal Frequency) Binning

Each bin has approximately same number of samples.

```python
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_binned = kbd.fit_transform(X)
```

**Example:**
```
Data: 10 values, 5 bins → Each bin ~2 values
Bin edges based on percentiles: 20th, 40th, 60th, 80th, 100th
```

**Pros:** 
- Handles outliers better
- Uniform value spread
- Default in sklearn

**Cons:** Bin widths vary

---

#### 3. K-Means Binning

Uses K-Means clustering to find optimal bin boundaries.

```python
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
X_binned = kbd.fit_transform(X)
```

**When to Use:** Data has natural clusters

---

#### KBinsDiscretizer Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `n_bins` | int | Number of bins |
| `encode` | 'ordinal', 'onehot' | Output encoding |
| `strategy` | 'uniform', 'quantile', 'kmeans' | Binning strategy |

---

#### Example: Age Binning

```python
# Age categories
kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
df['Age_binned'] = kbd.fit_transform(df[['Age']])

# Result: 0=Young, 1=Adult, 2=Middle-aged, 3=Senior
```

#### Viewing Bin Edges

```python
print(kbd.bin_edges_)
# [array([0.42, 20.125, 28., 38., 80.])]
```

---

#### Binarization

Converting numerical data to binary (0 or 1) based on threshold.

```python
from sklearn.preprocessing import Binarizer

# Values > 30 become 1, others 0
binarizer = Binarizer(threshold=30)
X_binary = binarizer.fit_transform(X)
```

**Use Cases:**
- Image processing (pixel thresholding)
- Creating binary features (e.g., is_adult: age > 18)

**Example:**
```python
# Family size > 0 means not traveling alone
binarizer = Binarizer(threshold=0)
df['has_family'] = binarizer.fit_transform(df[['family_size']])
```

---

#### Custom Binning (Domain Knowledge)

```python
# Age categories based on domain knowledge
bins = [0, 18, 35, 60, 100]
labels = ['Child', 'Young', 'Adult', 'Senior']
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)
```

**Note:** Custom binning not available in sklearn, use pandas.

---

### Video 34: Handling Date and Time Variables

#### Why Date/Time Features Matter?

Date/Time columns mein hidden information hota hai:
- Day of week (Monday, Tuesday...)
- Month
- Year
- Quarter
- Weekend vs Weekday
- Time of day

#### Converting to DateTime

```python
# By default, date columns are loaded as string
df['date'] = pd.to_datetime(df['date'])

# Check dtype
df.info()  # Should show datetime64
```

---

#### Extracting Date Components

```python
# Year
df['year'] = df['date'].dt.year

# Month (number)
df['month'] = df['date'].dt.month

# Month (name)
df['month_name'] = df['date'].dt.month_name()

# Day of month
df['day'] = df['date'].dt.day

# Day of week (0=Monday, 6=Sunday)
df['day_of_week'] = df['date'].dt.dayofweek

# Day name
df['day_name'] = df['date'].dt.day_name()

# Week number
df['week'] = df['date'].dt.isocalendar().week

# Quarter
df['quarter'] = df['date'].dt.quarter
```

---

#### Creating Derived Features

**Is Weekend:**
```python
df['is_weekend'] = df['date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)
```

**Semester:**
```python
df['semester'] = df['quarter'].isin([1, 2]).astype(int) + 1
# Q1, Q2 → Semester 1
# Q3, Q4 → Semester 2
```

---

#### Time Difference Calculation

```python
from datetime import datetime

# Current date
today = datetime.today()

# Days since order
df['days_since_order'] = (today - df['order_date']).dt.days

# Months difference
df['months_diff'] = (today - df['order_date']) / np.timedelta64(1, 'M')
```

---

#### Extracting Time Components

```python
# Hour
df['hour'] = df['datetime'].dt.hour

# Minute
df['minute'] = df['datetime'].dt.minute

# Second
df['second'] = df['datetime'].dt.second

# Time only (without date)
df['time'] = df['datetime'].dt.time
```

---

#### Time-Based Features

```python
# Part of day
def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['part_of_day'] = df['hour'].apply(get_part_of_day)
```

---

#### Time Difference in Specific Units

```python
# Difference between two times
time_diff = df['end_time'] - df['start_time']

# In seconds
df['duration_seconds'] = time_diff.dt.total_seconds()

# In minutes
df['duration_minutes'] = time_diff / pd.Timedelta(minutes=1)

# In hours
df['duration_hours'] = time_diff / pd.Timedelta(hours=1)
```

---

#### Complete Example

```python
# E-commerce order data
df['order_date'] = pd.to_datetime(df['order_date'])

# Extract features
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['day_of_week'] = df['order_date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['quarter'] = df['order_date'].dt.quarter

# Days since first order
first_order = df['order_date'].min()
df['days_since_start'] = (df['order_date'] - first_order).dt.days
```

---

## Section 4: Missing Data & Outliers

### Video 35: Handling Missing Data - Complete Case Analysis

#### Types of Missing Data

| Type | Abbreviation | Description | Can Fill? |
|------|--------------|-------------|-----------|
| **Missing Completely At Random (MCAR)** | MCAR | Data collection hi nahi hua (random) | Maybe |
| **Missing At Random (MAR)** | MAR | Missing due to other observed variables | Yes |
| **Missing Not At Random (MNAR)** | MNAR | Intentionally hidden/removed | Difficult |

**Example:**
- MCAR: Survey form lost accidentally
- MAR: Men less likely to disclose weight (can predict from gender)
- MNAR: Rich people don't disclose income (can't predict from other variables)

---

#### Complete Case Analysis (CCA)

**What:** Simply drop all rows with any missing values.

```python
# Drop rows with missing values
df_clean = df.dropna()

# Drop rows where specific column is missing
df_clean = df.dropna(subset=['Age'])
```

#### When to Use CCA

✅ Use when:
- Missing data is MCAR
- Very small percentage of missing data (< 5%)
- Enough data remains after dropping

❌ Don't use when:
- High percentage of missing data
- Missing data has a pattern (MAR/MNAR)
- Loss of important information

---

#### Disadvantages of CCA

1. **Data Loss** - Lose valuable information
2. **Bias** - If data not MCAR, introduces bias
3. **Reduced Sample Size** - Less data for training

---

### Video 36: Handling Missing Numerical Data - Simple Imputer

#### What is Imputation?
Missing values ko estimated values se fill karna.

#### Simple Imputer (Univariate)

Ek column ke statistics use karke usi column ko fill karna.

**Strategies:**

| Strategy | Formula | Use Case |
|----------|---------|----------|
| **mean** | Column ka mean | Normal distribution |
| **median** | Column ka median | Outliers present |
| **mode** | Most frequent value | Categorical |
| **constant** | Fixed value | When you know fill value |

---

#### Mean Imputation

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])
```

**Pros:**
- Simple and fast
- Works well for MCAR

**Cons:**
- Reduces variance
- Doesn't preserve relationships
- Bad for MAR data

---

#### Median Imputation

```python
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])
```

**When to Use:**
- Outliers present in data
- Skewed distribution

---

#### Arbitrary Value Imputation

```python
imputer = SimpleImputer(strategy='constant', fill_value=-1)
df['Age'] = imputer.fit_transform(df[['Age']])
```

**Common choices:**
- 0, -1, 999
- Value outside normal range (to distinguish)

---

### Video 37: Handling Missing Categorical Data

#### Most Frequent (Mode) Imputation

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer.fit_transform(df[['Embarked']])
```

---

#### Missing Category Approach

**Treat missing as a separate category.**

```python
df['Cabin'].fillna('Missing', inplace=True)
```

**Advantages:**
- Preserves missingness information
- Sometimes missingness itself is predictive
- Works well for tree-based models

---

### Video 38: Missing Indicator & Random Sample Imputation

#### Missing Indicator

**What:** Create new binary column indicating where values were missing.

```python
from sklearn.impute import SimpleImputer, MissingIndicator

# Missing indicator
indicator = MissingIndicator()
missing_flags = indicator.fit_transform(df)

# Or use SimpleImputer with add_indicator
imputer = SimpleImputer(strategy='mean', add_indicator=True)
result = imputer.fit_transform(df[['Age']])
# Result has: imputed values + indicator column
```

**Why Use:**
- Missingness pattern can be predictive
- Preserve information about missing data
- Use alongside any imputation method

---

#### Random Sample Imputation

**What:** Fill missing values with random samples from existing values.

```python
import numpy as np

# Get non-missing values
valid_values = df['Age'].dropna()

# Count missing
n_missing = df['Age'].isnull().sum()

# Generate random samples
random_samples = valid_values.sample(n=n_missing, replace=True).values

# Fill missing
df.loc[df['Age'].isnull(), 'Age'] = random_samples
```

**Advantages:**
- Preserves original distribution
- Maintains variance
- Good for MCAR data

---

### Video 39 & 40: KNN Imputer (Multivariate Imputation)

#### Univariate vs Multivariate Imputation

| Type | Method | Use Other Columns? |
|------|--------|-------------------|
| **Univariate** | SimpleImputer (mean, median) | No |
| **Multivariate** | KNNImputer, IterativeImputer | Yes |

---

#### KNN Imputer

**How it works:**
1. Find k nearest neighbors for row with missing value
2. Use neighbors' values to impute
3. For numerical: Average of k neighbors
4. Distance metric: Nan Euclidean distance

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

---

#### Nan Euclidean Distance

Normal Euclidean distance modified to handle missing values:
- Only calculate distance for present values
- Apply weight based on how many values present

**Formula:**
```
d = sqrt(weight * sum((x_i - y_i)²))
weight = total_features / present_features
```

---

#### KNNImputer Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_neighbors` | Number of neighbors | 5 |
| `weights` | 'uniform' or 'distance' | 'uniform' |
| `metric` | Distance metric | 'nan_euclidean' |
| `add_indicator` | Add missing indicator | False |

---

#### Uniform vs Distance Weighting

**Uniform:**
```
Imputed value = (neighbor1 + neighbor2) / 2
```

**Distance:**
- Closer neighbors have more weight
- Weight = 1/distance
```
Imputed value = (v1/d1 + v2/d2) / (1/d1 + 1/d2)
```

```python
# Distance-weighted KNN
imputer = KNNImputer(n_neighbors=5, weights='distance')
```

---

#### Advantages & Disadvantages

**Advantages:**
- Uses relationships between features
- Generally better results than univariate

**Disadvantages:**
- Slow for large datasets (O(n²) distance calculations)
- Needs to store entire training data
- Memory intensive

---

### Video 41: MICE (Multivariate Imputation by Chained Equations)

#### What is MICE/Iterative Imputer?

**Multiple Imputation by Chained Equations:**
1. Fill missing values with simple imputation (mean)
2. For each column with missing:
   - Treat as target
   - Use other columns as features
   - Train model, predict missing
3. Repeat until convergence

---

#### How MICE Works

```
Step 1: Initial fill with mean
     Col1  Col2  Col3
     100   NaN   500
      50    20   600
      NaN   30   NaN
After: 75, 25, 550 (means)

Step 2: For Col1 missing
- Make Col1 NaN again
- Train model: Col1 ~ Col2 + Col3
- Predict Col1 missing values

Step 3: For Col2 missing
- Make Col2 NaN again
- Train model: Col2 ~ Col1 + Col3
- Predict Col2 missing values

Step 4: Repeat until values converge
```

---

#### Implementation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = imputer.fit_transform(df)
```

---

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_iter` | Maximum iterations | 10 |
| `estimator` | Model to use | BayesianRidge |
| `initial_strategy` | Initial fill method | 'mean' |
| `random_state` | For reproducibility | None |

---

#### Custom Estimator

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Using Ridge
imputer = IterativeImputer(estimator=Ridge(), max_iter=10)

# Using Random Forest
imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10)
```

---

#### When to Use

| Use | Don't Use |
|-----|-----------|
| MAR data | MCAR (simple imputation enough) |
| Important relationships between features | Large datasets (slow) |
| Better accuracy needed | Quick baseline models |

---

### Video 42: What are Outliers?

#### Definition
Outlier wo data point hai jo baaki data se bahut alag ho - unusually high ya low value.

**Example:** Sharma ji ka beta - class mein sabko kam marks aaye, usko 100% aya!

---

#### Why Outliers are Dangerous?

```
Class salaries: 20K, 15K, 18K, 25K, 22K, Bill Gates: 10 Crore

Mean salary without Bill Gates: ~20K
Mean salary with Bill Gates: Crores!
```

**Impact on ML models:**
- Bias predictions
- Distort relationships
- Linear models especially affected

---

#### Visual Example: Linear Regression

```
Normal data → Good fit line
With outlier → Line pulled towards outlier

Original fit: y = 2x + 1
With outlier: y = 0.5x + 10 (bad fit!)
```

---

#### Which Algorithms are Affected?

**Highly Affected:**
- Linear Regression
- Logistic Regression
- SVM (Support Vector Machines)
- Neural Networks
- Principal Component Analysis

**Less Affected (Tree-based):**
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost

**Why?** Tree-based models split data, don't calculate weighted sums.

---

#### Should You Always Remove Outliers?

**Remove when:**
- Data entry error (Age = 838 instead of 38)
- Measurement error
- Outlier doesn't represent real scenario

**Keep when:**
- Anomaly detection task (fraud detection)
- Outlier is legitimate data
- Outlier adds information

---

#### Outlier Treatment Methods

| Method | Description |
|--------|-------------|
| **Trimming** | Remove outliers completely |
| **Capping** | Replace with threshold value |
| **Treat as Missing** | Replace with NaN, then impute |
| **Discretization** | Bin the data |

---

### Video 43: Outlier Detection using Z-Score Method

#### What is Z-Score?

**Formula:**
```
z = (x - μ) / σ

Where:
- x = data point
- μ = mean
- σ = standard deviation
```

Z-score tells how many standard deviations away from mean.

---

#### When to Use

**Condition:** Data should be normally distributed (or close to normal)

**Check:** Histogram, QQ-plot, Skewness (~0)

```python
# Check distribution
import seaborn as sns
sns.histplot(df['column'], kde=True)

# Check skewness
df['column'].skew()  # Should be close to 0
```

---

#### Detection Rule

For normal distribution:
- 68% data within ±1 std
- 95% data within ±2 std
- 99.7% data within ±3 std

**Rule:** If Z-score > 3 or < -3 → Outlier

---

#### Implementation

**Step 1: Calculate bounds**
```python
mean = df['column'].mean()
std = df['column'].std()

upper_limit = mean + 3 * std
lower_limit = mean - 3 * std
```

**Step 2: Find outliers**
```python
outliers = df[(df['column'] > upper_limit) | (df['column'] < lower_limit)]
print(f"Found {len(outliers)} outliers")
```

---

#### Treatment: Trimming

```python
# Remove outliers
df_trimmed = df[(df['column'] <= upper_limit) & (df['column'] >= lower_limit)]
```

---

#### Treatment: Capping

```python
# Cap outliers at limits
df['column'] = np.where(
    df['column'] > upper_limit,
    upper_limit,
    np.where(
        df['column'] < lower_limit,
        lower_limit,
        df['column']
    )
)

# Or using pandas
df['column'] = df['column'].clip(lower=lower_limit, upper=upper_limit)
```

---

#### Using Z-Score Column

```python
from scipy import stats

# Calculate z-scores
df['zscore'] = stats.zscore(df['column'])

# Filter outliers
df_clean = df[df['zscore'].abs() <= 3]
```

---

### Video 44: Outlier Detection using IQR Method

#### When to Use

**Condition:** When data is skewed (not normal distribution)

```python
# Check skewness
df['column'].skew()  # If not close to 0, use IQR
```

---

#### What is IQR?

**IQR = Q3 - Q1**

Where:
- Q1 = 25th percentile
- Q3 = 75th percentile

---

#### Box Plot Review

```
                 IQR
            ┌─────────┐
  Whisker   │ ┌─────┐ │   Whisker
←─────────→ │ │     │ │ ←─────────→
Min        Q1 Median Q3          Max
            └─────────┘

Points outside whiskers = Outliers
```

---

#### IQR Proximity Rule

```
Lower Limit = Q1 - 1.5 × IQR
Upper Limit = Q3 + 1.5 × IQR
```

**Implementation:**

```python
# Calculate percentiles
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1

# Calculate limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['column'] < lower_limit) | (df['column'] > upper_limit)]
```

---

#### Treatment: Trimming

```python
df_trimmed = df[
    (df['column'] >= lower_limit) & 
    (df['column'] <= upper_limit)
]
```

---

#### Treatment: Capping

```python
df['column'] = np.where(
    df['column'] > upper_limit,
    upper_limit,
    np.where(
        df['column'] < lower_limit,
        lower_limit,
        df['column']
    )
)
```

---

#### Example: Placement Exam Marks

```python
# Skewed data (not normal)
df['placement_marks'].skew()  # 0.83 (right skewed)

# Calculate IQR
Q1 = df['placement_marks'].quantile(0.25)  # 17
Q3 = df['placement_marks'].quantile(0.75)  # 44
IQR = Q3 - Q1  # 27

# Limits
lower_limit = Q1 - 1.5 * IQR  # -23.5 (no lower outliers possible)
upper_limit = Q3 + 1.5 * IQR  # 84.5

# 15 outliers found above upper limit
outliers = df[df['placement_marks'] > upper_limit]
```

---

### Video 45: Percentile Method & Winsorization

#### Percentile Method

**Approach:** Define your own threshold percentiles.

```
Below 1st percentile → Outlier
Above 99th percentile → Outlier
```

---

#### Implementation

```python
# Calculate percentiles
lower_limit = df['column'].quantile(0.01)  # 1st percentile
upper_limit = df['column'].quantile(0.99)  # 99th percentile

# Find outliers
outliers = df[
    (df['column'] < lower_limit) | 
    (df['column'] > upper_limit)
]
```

---

#### Choosing Percentile Threshold

| Threshold | Cutoff | Data Removed |
|-----------|--------|--------------|
| 1%, 99% | Extreme only | ~2% |
| 5%, 95% | Moderate | ~10% |
| 2.5%, 97.5% | Common choice | ~5% |

---

#### Treatment: Trimming

```python
# Remove outliers
df_trimmed = df[
    (df['column'] >= lower_limit) & 
    (df['column'] <= upper_limit)
]
```

---

#### Winsorization (Capping)

**Winsorization:** Capping outliers using percentile method.

```python
from scipy.stats import mstats

# Winsorize (cap at 1st and 99th percentile)
df['column_winsorized'] = mstats.winsorize(df['column'], limits=[0.01, 0.01])
```

---

#### Manual Winsorization

```python
lower = df['column'].quantile(0.01)
upper = df['column'].quantile(0.99)

df['column'] = np.where(
    df['column'] > upper,
    upper,
    np.where(
        df['column'] < lower,
        lower,
        df['column']
    )
)
```

---

#### Comparison of Methods

| Method | Distribution Required | Pros | Cons |
|--------|----------------------|------|------|
| **Z-Score** | Normal | Uses mean, std | Sensitive to extreme outliers |
| **IQR** | Any | Robust | Fixed formula |
| **Percentile** | Any | Flexible | Subjective threshold |

---

#### When to Use Which?

```
Is data normally distributed?
├── Yes → Z-Score method
└── No (Skewed)
    ├── Use IQR method (fixed 1.5×IQR rule)
    └── Or Percentile method (custom thresholds)
```

---

## Section 5: Dimensionality Reduction - PCA

### Video 46: Curse of Dimensionality

#### What is Dimensionality?
Dimensionality matlab aapke dataset mein kitne columns (features) hain.

#### What is Curse of Dimensionality?
Jab features bohot zyada ho jaate hain, toh ML models ki performance girti hai.

**Problem:**
```
┌─────────────────────────────────────────────────────┐
│  As Number of Features ↑                            │
│                                                     │
│  • Model Performance ↑ (initially)                  │
│  • After optimal point → Performance ↓              │
│  • Computational Cost ↑                             │
│  • Data becomes sparse                              │
└─────────────────────────────────────────────────────┘
```

#### Why High Dimensions are Bad?

**1. Sparsity Problem:**
- More dimensions → Data points spread out
- Distance between points increases
- Models can't find patterns easily

**Example - Lost Wallet Analogy:**
```
1D (Line): Easy to find wallet
2D (Room): Harder to find
3D (Building): Much harder
100D Space: Nearly impossible!
```

**2. Distance Becomes Meaningless:**
- In high dimensions, all points are equally far
- KNN and other distance-based algorithms fail
- Euclidean distance loses meaning

**3. Computational Complexity:**
- More features = More computation
- Training time increases exponentially
- Memory requirements increase

---

#### Solutions to Curse of Dimensionality

| Method | Type | Description |
|--------|------|-------------|
| **Feature Selection** | Reduce dimensions | Pick best subset of features |
| **Feature Extraction** | Transform dimensions | Create new features from existing |

**Feature Selection:**
- Keep original features
- Remove irrelevant/redundant features
- Examples: Filter methods, Wrapper methods

**Feature Extraction:**
- Create new features
- Combine existing features
- Examples: PCA, t-SNE, LDA

---

#### Optimal Number of Features

```
Performance
    │     
    │     ╭──────╮
    │    ╱        ╲
    │   ╱          ╲
    │  ╱            ╲
    │ ╱              ╲
    │╱                ╲
    └───────────────────────
            Features →
              ↑
         Optimal Point
```

**Rule of Thumb:** Use as many features as needed, but not more!

---

### Video 47: Principal Component Analysis (PCA) - Geometric Intuition

#### What is PCA?

PCA ek **unsupervised** feature extraction technique hai jo high-dimensional data ko low-dimensional mein convert karta hai while keeping the essence of data.

**Analogy - Football Photographer:**
```
3D Football Match → 2D Photo
                 ↓
Photographer finds best angle to capture essence

PCA does the same:
High-D Data → Low-D Data
           ↓
Finds best "angle" (principal components) to preserve information
```

---

#### Why Use PCA?

**Benefit 1: Faster Algorithm Execution**
- Reduced features = Faster training
- Less computational cost
- Smaller model size

**Benefit 2: Data Visualization**
- Can't visualize beyond 3D
- PCA converts 784D → 2D or 3D
- Can now plot and see patterns

---

#### Feature Selection vs Feature Extraction

**Feature Selection (Simple Example):**
```
Data: Rooms, Grocery_Shops → House_Price

Which to keep? → Rooms (more variance, more predictive)

Method: Project data on each axis, keep axis with more spread
```

**Problem with Selection:**
What if both features have equal variance?

**Feature Extraction (PCA):**
```
Instead of choosing between existing features:
Create NEW features by combining existing ones

Rooms + Washrooms → Size (new feature!)
```

---

#### Geometric Intuition

**Original Space:**
```
        Y (Washrooms)
        │    • 
        │   •  •
        │  •    •
        └──────────── X (Rooms)
```

**PCA Process:**
1. Find direction of maximum variance
2. Create new axis (Principal Component 1)
3. Find perpendicular direction (Principal Component 2)
4. Project data onto new axes

**Result:**
```
        PC2
        │     
        │    •••
        │  •••••
        │ •••••
        └────────────── PC1 (Maximum Variance)
```

---

#### Key Concepts

**1. Variance = Spread of Data**
- More variance → More information
- PCA maximizes variance along new axes

**2. Principal Components:**
- New axes created by PCA
- PC1 has maximum variance
- PC2 has second maximum variance
- PCs are orthogonal (perpendicular)

**3. Number of PCs = Number of Original Features**
- 3 features → 3 PCs (but can use fewer)
- Choose top k PCs with most variance

---

#### Why Variance is Important?

**Example - Green vs Red Points:**
```
Original 2D Space:
        │ • •
     R• │• G •
        │  • • 
        └──────
           
If we project on wrong axis:
Red and Green points overlap → Can't distinguish!

If we project on correct axis (max variance):
Red and Green points separated → Can distinguish!
```

**Conclusion:** Maximum variance → Better separation → Better ML performance

---

### Video 48: PCA - Mathematical Formulation

#### Problem Formulation

**Goal:** Find unit vector **u** such that variance of projected data is maximum.

**Steps:**
1. Data points as vectors: **x₁, x₂, ..., xₙ**
2. Find projection onto unit vector **u**
3. Maximize variance of projections

---

#### Projection Formula

For a data point **x** projected onto unit vector **u**:

```
Projection = u^T · x (dot product)
```

Since **u** is unit vector (|u| = 1):
```
Projection scalar = u^T · x
```

---

#### Variance of Projections

```
Variance = (1/n) Σ (u^T·xᵢ - u^T·x̄)²
```

**Objective:** Maximize this variance by finding optimal **u**

---

#### What is Covariance?

**Problem with Variance:**
- Variance only tells spread on single axis
- Doesn't tell relationship between axes

**Covariance:**
- Measures how two variables change together
- Positive: Both increase together
- Negative: One increases, other decreases

**Formula:**
```
Cov(X, Y) = (1/n) Σ (xᵢ - x̄)(yᵢ - ȳ)
```

---

#### Covariance Matrix

For data with multiple features:

```
        [ Var(X1)      Cov(X1,X2)   Cov(X1,X3) ]
C =     [ Cov(X2,X1)   Var(X2)      Cov(X2,X3) ]
        [ Cov(X3,X1)   Cov(X3,X2)   Var(X3)    ]
```

**Properties:**
- Diagonal: Variances
- Off-diagonal: Covariances
- Symmetric: Cov(X,Y) = Cov(Y,X)
- Contains complete spread and orientation info

---

#### Eigenvectors and Eigenvalues

**Eigenvector:** Special vector that doesn't change direction under transformation (only scales)

```
A · v = λ · v

Where:
- A = Matrix (transformation)
- v = Eigenvector
- λ = Eigenvalue (scaling factor)
```

**Intuition:**
```
When you apply transformation A:
- Most vectors change direction
- Eigenvectors ONLY scale (stretch/shrink)
- Eigenvalue = how much it scales
```

---

#### Solution to PCA

**When we solve the optimization problem:**

The eigenvectors of covariance matrix give principal components!

```
Covariance Matrix → Eigen Decomposition → Eigenvectors & Eigenvalues
                                         ↓              ↓
                                    Principal      Variance explained
                                    Components     by each PC
```

**Key Insight:**
- Largest eigenvalue → PC1 (most variance)
- Second largest → PC2
- And so on...

---

#### PCA Steps (Summary)

```
Step 1: Mean Centering
        X_centered = X - mean(X)

Step 2: Compute Covariance Matrix
        C = (1/n) X^T · X

Step 3: Eigen Decomposition
        C → Eigenvectors + Eigenvalues

Step 4: Sort by Eigenvalue (descending)
        Keep top k eigenvectors

Step 5: Transform Data
        X_new = X · Eigenvectors_top_k
```

---

### Video 49: PCA - Code Implementation & Visualization

#### Implementation from Scratch

**Step 1: Mean Centering**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Step 2: Compute Covariance Matrix**
```python
import numpy as np

cov_matrix = np.cov(X_scaled.T)  # Transpose for correct shape
print(cov_matrix.shape)  # (n_features, n_features)
```

**Step 3: Eigen Decomposition**
```python
from numpy.linalg import eig

eigenvalues, eigenvectors = eig(cov_matrix)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors shape: {eigenvectors.shape}")
```

**Step 4: Select Top k Components**
```python
# Sort by eigenvalue (descending)
sorted_idx = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_idx]
sorted_eigenvectors = eigenvectors[:, sorted_idx]

# Select top k
k = 2
top_k_eigenvectors = sorted_eigenvectors[:, :k]
```

**Step 5: Transform Data**
```python
X_transformed = X_scaled.dot(top_k_eigenvectors)
print(f"Original shape: {X_scaled.shape}")
print(f"Transformed shape: {X_transformed.shape}")
```

---

#### Using Scikit-learn PCA

```python
from sklearn.decomposition import PCA

# Create PCA object
pca = PCA(n_components=2)

# Fit and transform
X_pca = pca.fit_transform(X_scaled)

# Access components
print(pca.components_)        # Eigenvectors (principal components)
print(pca.explained_variance_)  # Eigenvalues
print(pca.explained_variance_ratio_)  # Percentage of variance explained
```

---

#### MNIST Dataset Example

**Loading Data:**
```python
# MNIST: 28x28 pixel images → 784 features
# Each pixel = 1 feature

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target
```

**Without PCA:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(f"Accuracy without PCA: {accuracy:.4f}")  # ~96.7%
```

**With PCA:**
```python
# Scale first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=100)  # 784 → 100 features
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train model
knn = KNeighborsClassifier()
knn.fit(X_train_pca, y_train)
accuracy = knn.score(X_test_pca, y_test)
print(f"Accuracy with PCA (100 components): {accuracy:.4f}")  # ~95%+
```

**Observation:** Similar accuracy with 7x fewer features!

---

#### Visualizing High-Dimensional Data

**2D Visualization:**
```python
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

import plotly.express as px
fig = px.scatter(x=X_2d[:, 0], y=X_2d[:, 1], color=y)
fig.show()
```

**3D Visualization:**
```python
pca = PCA(n_components=3)
X_3d = pca.fit_transform(X_scaled)

fig = px.scatter_3d(x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2], color=y)
fig.show()
```

---

#### Explained Variance Ratio

**What it tells:** How much information each PC captures.

```python
pca = PCA()  # All components
pca.fit(X_scaled)

# Variance explained by each PC
print(pca.explained_variance_ratio_)
# [0.10, 0.08, 0.05, ...] → PC1: 10%, PC2: 8%, etc.

# Cumulative variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
```

---

#### Finding Optimal Number of Components

**Rule:** Keep enough components to explain ~90% variance.

```python
pca = PCA()
pca.fit(X_scaled)

# Plot cumulative variance
cumsum = np.cumsum(pca.explained_variance_ratio_)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.9, color='r', linestyle='--')  # 90% line
plt.show()
```

**Finding exact number:**
```python
n_components_90 = np.argmax(cumsum >= 0.90) + 1
print(f"Components needed for 90% variance: {n_components_90}")
```

---

#### When PCA Doesn't Work Well

**Case 1: Equal Variance in All Directions**
```
        │  • • •
        │• • • •
        │  • • •
        └──────────
        
Circle-like data → No clear PC1 direction
PCA won't help reduce dimensions
```

**Case 2: Non-Linear Patterns**
```
        │    • •
        │  •   •
        │ •     •
        │  •   •
        │    • •
        └──────────
        
Parabola or circular patterns
Linear PCA loses this structure
Use: Kernel PCA, t-SNE
```

**Case 3: Similar Variance on All Axes**
- No dimension has more information
- All features equally important
- PCA reduction will lose information

---

#### Summary: PCA Workflow

```python
# Complete PCA workflow
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))  # Keep 95% variance
])

# 2. Fit and transform
X_reduced = pipeline.fit_transform(X_train)

# 3. Transform test data
X_test_reduced = pipeline.transform(X_test)

# 4. Train any model on reduced data
model.fit(X_reduced, y_train)
```

---


---

## Section 6: Linear Models & Regression (Videos 50-68)

### Video 50: Simple Linear Regression - Intuition & Basics

#### What is Linear Regression?

Linear Regression ek **supervised learning algorithm** hai jo **regression problems** solve karta hai.

```
Regression Problem = Output column NUMERICAL hota hai (continuous values)
```

**Simple Linear Regression (SLR):**
- Sirf **1 input column** (feature) aur **1 output column** (target)
- Example: CGPA → Package

**Multiple Linear Regression (MLR):**
- **Multiple input columns** aur **1 output column**
- Example: CGPA + IQ + Gender → Package

---

#### Geometric Intuition

**Goal:** Best fit line dhundhna jo data ke through jaaye.

```
        Package (y)
            │
         •  │      •
            │   •
         •  │  /────────  Best Fit Line
            │ /   •
         •  │/  •
            └───────────── CGPA (x)
```

**Line Equation:**
```
y = mx + b

where:
- m = slope (line ka angle)
- b = intercept (line kahan se y-axis ko cross karta hai)
```

---

#### Loss Function (Cost Function)

**Goal:** Actual values aur Predicted values ka difference minimize karna.

```
Loss = Σ (yᵢ - ŷᵢ)²
     = Σ (yᵢ - (mx + b))²
```

**Visual:**
```
         │    •  ← Actual Point
         │    |   (y - ŷ) = Error/Residual
         │    |
    ─────┼────●──────── Prediction Line
         │    ↑
         │  Predicted Point
```

---

#### Best Fit Line Selection

| Line Type | Loss | Selection |
|-----------|------|-----------|
| Line 1 | 25 | ❌ |
| Line 2 | 15 | ❌ |
| Line 3 | 5 | ✅ Minimum Loss |

**Selection Rule:** Jo line ka total squared error (loss) minimum ho, wahi best fit line.

---

### Video 51: Simple Linear Regression - Mathematical Formulation

#### Ordinary Least Squares (OLS) Method

**Objective:** Find m and b such that loss is minimized.

**Loss Function:**
```
L(m, b) = Σ (yᵢ - mx - b)²
```

**Taking Partial Derivatives:**
```
∂L/∂m = 0  →  Solve for m
∂L/∂b = 0  →  Solve for b
```

**Closed Form Solution:**
```
         Σ(xᵢ - x̄)(yᵢ - ȳ)
m = ─────────────────────────
         Σ(xᵢ - x̄)²


b = ȳ - m·x̄
```

where:
- x̄ = mean of x values
- ȳ = mean of y values

---

#### Code from Scratch

```python
import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.m = None
        self.b = None
    
    def fit(self, X, y):
        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope (m)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        self.m = numerator / denominator
        
        # Calculate intercept (b)
        self.b = y_mean - self.m * x_mean
    
    def predict(self, X):
        return self.m * X + self.b

# Usage
model = SimpleLinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### Video 52: Multiple Linear Regression

#### Extension from Simple to Multiple

**Simple Linear Regression:**
```
y = mx + b
```

**Multiple Linear Regression:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

---

#### Geometric Intuition

| Dimensions | Input Features | Model Shape |
|------------|----------------|-------------|
| 2D | 1 feature | Line |
| 3D | 2 features | Plane |
| nD | n features | Hyperplane |

```
3D Example (2 features):

    Package │    
            │   ╱ ← Hyperplane
            │  ╱
            │ ╱
            │╱_________ IQ
           ╱│
        CGPA
```

---

#### Matrix Formulation

**Equation in Matrix Form:**
```
Y = Xβ + ε

Where:
- Y = Target vector (n × 1)
- X = Feature matrix (n × p+1) with 1s column for intercept
- β = Coefficients vector (p+1 × 1)
- ε = Error vector (n × 1)
```

**OLS Solution:**
```
β = (XᵀX)⁻¹ Xᵀ Y
```

---

#### Code Example

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

### Video 53-56: Gradient Descent

#### Why Gradient Descent?

**Problem with OLS:**
```
β = (XᵀX)⁻¹ Xᵀ Y

Matrix inverse (XᵀX)⁻¹ is computationally expensive!
Time Complexity: O(n³)
```

**Solution:** Gradient Descent - Iterative optimization

---

#### Gradient Descent Intuition

**Analogy:** Blindfolded person finding lowest point in a valley.

```
        Loss (L)
           │╲
           │ ╲
           │  ╲
           │   ╲___
           │       ╲___/  ← Global Minimum
           └─────────────── Parameter (β)
                   
Step 1: Start at random point
Step 2: Calculate slope (gradient)
Step 3: Move in direction of steepest descent
Step 4: Repeat until minimum reached
```

---

#### Mathematical Formulation

**Update Rule:**
```
βₙₑw = βₒₗd - η · ∂L/∂β

Where:
- η = Learning Rate (step size)
- ∂L/∂β = Gradient of Loss w.r.t. β
```

**For Simple Linear Regression:**
```
∂L/∂m = -2/n · Σ xᵢ(yᵢ - ŷᵢ)
∂L/∂b = -2/n · Σ (yᵢ - ŷᵢ)
```

---

#### Types of Gradient Descent

| Type | Data Used | Pros | Cons |
|------|-----------|------|------|
| **Batch GD** | Entire dataset | Stable, Smooth convergence | Slow for large data |
| **Stochastic GD** | 1 sample | Fast updates | Noisy, May not converge |
| **Mini-Batch GD** | Batch of samples | Balance of both | Requires batch size tuning |

---

#### Code: Batch Gradient Descent

```python
import numpy as np

class GradientDescentRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Store loss for visualization
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Usage
model = GradientDescentRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

#### Learning Rate Effects

```
Too Small LR:              Too Large LR:           Optimal LR:
    │                          │                      │
    │\                         │ ↗                    │\
    │ \                        │↗ ↘                   │ \_
    │  \                       │    ↗↘                │   \__/
    │   \___                   │      Diverges!       │   Converges!
    Slow convergence           Oscillates             
```

**Rule of Thumb:** Start with 0.01, adjust based on loss curve.

---

### Video 57: Regression Metrics

#### Mean Absolute Error (MAE)

```
        Σ |yᵢ - ŷᵢ|
MAE = ─────────────
            n
```

**Pros:**
- Same unit as target variable
- Robust to outliers

**Cons:**
- Not differentiable at 0
- Can't use as loss function for optimization

---

#### Mean Squared Error (MSE)

```
        Σ (yᵢ - ŷᵢ)²
MSE = ───────────────
            n
```

**Pros:**
- Differentiable → Can use as loss function
- Penalizes large errors more

**Cons:**
- Unit is squared → Hard to interpret
- Sensitive to outliers

---

#### Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √(Σ(yᵢ - ŷᵢ)² / n)
```

**Pros:**
- Same unit as target
- Differentiable
- Most commonly used

---

#### R² Score (Coefficient of Determination)

**Intuition:** How much variance is explained by the model?

```
           SSᵣₑₛ         Σ(yᵢ - ŷᵢ)²
R² = 1 - ────── = 1 - ─────────────
           SSₜₒₜ         Σ(yᵢ - ȳ)²

Where:
- SSᵣₑₛ = Residual Sum of Squares (model error)
- SSₜₒₜ = Total Sum of Squares (baseline error using mean)
```

**Interpretation:**
| R² Value | Meaning |
|----------|---------|
| 1.0 | Perfect fit |
| 0.8 - 1.0 | Very good |
| 0.5 - 0.8 | Good |
| < 0.5 | Poor |
| < 0 | Worse than mean baseline |

---

#### Adjusted R²

**Problem with R²:** Adding more features always increases R² (even irrelevant ones).

**Solution:** Adjusted R² penalizes adding useless features.

```
                          n - 1
Adjusted R² = 1 - (1-R²) · ─────
                          n - p - 1

Where:
- n = number of samples
- p = number of features
```

---

#### Code Example

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate all metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Adjusted R²
n = len(y_test)
p = X_test.shape[1]  # number of features
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
```

---

### Video 58: Polynomial Regression

#### When Linear Doesn't Fit

**Problem:** Data relationship is non-linear.

```
Linear Fit (Bad):           Polynomial Fit (Good):

    │     •                     │     •
    │   •   •                   │   •   •
    │ •       •                 │ •  ╭───╮  •
    │───────────                │   ╯     ╰
    │•         •                │•           •
```

---

#### How Polynomial Regression Works

**Transform features to polynomial:**
```
Original: [x]
Degree 2: [x, x²]
Degree 3: [x, x², x³]
```

**Then apply Linear Regression on transformed features!**

```
y = β₀ + β₁x + β₂x² + β₃x³ + ...
```

---

#### Code Example

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Method 1: Manual transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Method 2: Using Pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

#### Choosing Polynomial Degree

| Degree | Effect |
|--------|--------|
| Too Low | Underfitting - Model too simple |
| Too High | Overfitting - Fits noise |
| Optimal | Best generalization |

**Use Cross-Validation to find optimal degree!**

---

### Video 59-60: Bias-Variance Tradeoff

#### What is Bias?

**Bias:** Error from wrong assumptions in learning algorithm.

```
High Bias (Underfitting):
    │     •    •
    │ •  •────────•  ← Simple line misses pattern
    │   •      •
    │•      •
    
Model is TOO SIMPLE
```

---

#### What is Variance?

**Variance:** Error from sensitivity to fluctuations in training data.

```
High Variance (Overfitting):
    │  •  ╭╮
    │ •╭╯╰╮  •
    │ ╯    ╰╮ ╭╯  ← Wiggly line fits noise
    │       ╰╯
    
Model is TOO COMPLEX
```

---

#### The Tradeoff

```
Error
  │
  │╲               ╱
  │ ╲    Total    ╱
  │  ╲   Error   ╱
  │   ╲__╱─────╲╱
  │    ╲  /     
  │  Bias╲/Variance
  │    ↓
  │  Optimal
  └─────────────────── Model Complexity
      ↑               ↑
  Underfitting    Overfitting
```

**Key Points:**
- Simple models: High Bias, Low Variance
- Complex models: Low Bias, High Variance
- Goal: Find sweet spot (minimum total error)

---

### Video 61-63: Ridge Regression (L2 Regularization)

#### What is Regularization?

**Problem:** Overfitting in Linear Regression (large coefficients).

**Solution:** Add penalty term to loss function to constrain coefficients.

---

#### Ridge Regression Formula

**Standard Linear Regression Loss:**
```
L = Σ(yᵢ - ŷᵢ)²
```

**Ridge Regression Loss:**
```
L = Σ(yᵢ - ŷᵢ)² + λ·Σβⱼ²
                   ↑
           L2 Penalty (Ridge)

Where:
- λ (lambda/alpha) = Regularization strength
- Σβⱼ² = Sum of squared coefficients
```

---

#### Effect of Lambda (α)

| α Value | Effect |
|---------|--------|
| α = 0 | Same as Linear Regression |
| α small | Slight regularization |
| α optimal | Best bias-variance balance |
| α large | Heavy regularization → Underfitting |

```
Coefficients
    │
    │ ●──────────────  α = 0 (high coefficients)
    │   ●
    │     ●
    │       ●────────  α optimal
    │         ●
    │           ●────  α high (coefficients → 0)
    └─────────────────── α
```

---

#### Code Example

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Simple Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

# Finding optimal alpha using GridSearchCV
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best alpha:", grid_search.best_params_['alpha'])
print("Best R² score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_
```

---

### Video 64-65: Lasso Regression (L1 Regularization)

#### Lasso Formula

```
L = Σ(yᵢ - ŷᵢ)² + λ·Σ|βⱼ|
                   ↑
           L1 Penalty (Lasso)
```

---

#### Lasso vs Ridge

| Feature | Ridge (L2) | Lasso (L1) |
|---------|------------|------------|
| Penalty | Σβⱼ² | Σ\|βⱼ\| |
| Coefficients | Shrinks towards 0 | Can make exactly 0 |
| Feature Selection | No | Yes (sparse) |
| When to use | All features relevant | Many irrelevant features |

---

#### Why Lasso Creates Sparsity?

**Geometric Intuition:**
```
Ridge (Circle):                Lasso (Diamond):
     β₂                            β₂
      │   ╭───╮                     │   /\
      │  │     │                    │  /  \
      ├──┼─────┼──→ β₁              ├─/────\──→ β₁
      │  │     │                    │ \    /
      │   ╰───╯                     │  \  /
      │                             │   \/

Contours hit CORNERS         Corners are on AXES
of circle → β ≠ 0           → One β becomes 0!
```

---

#### Code Example

```python
from sklearn.linear_model import Lasso

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

# Check which features are selected (non-zero coefficients)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lasso.coef_
})

# Features with coefficient = 0 are eliminated
selected_features = feature_importance[feature_importance['coefficient'] != 0]
print("Selected features:\n", selected_features)
```

---

### Video 66: Elastic Net Regression

#### Combining Ridge + Lasso

```
L = Σ(yᵢ - ŷᵢ)² + λ₁·Σ|βⱼ| + λ₂·Σβⱼ²
                   ↑           ↑
               L1 (Lasso)   L2 (Ridge)
```

**Or in sklearn form:**
```
L = Σ(yᵢ - ŷᵢ)² + α·[ρ·Σ|βⱼ| + (1-ρ)·Σβⱼ²]

Where:
- α = Overall regularization strength
- ρ (l1_ratio) = Mix ratio between L1 and L2
  - ρ = 1 → Pure Lasso
  - ρ = 0 → Pure Ridge
  - ρ = 0.5 → Equal mix
```

---

#### When to Use Elastic Net?

- When you have many correlated features
- When you want feature selection (like Lasso) but more stable (like Ridge)

---

#### Code Example

```python
from sklearn.linear_model import ElasticNet

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50% L1, 50% L2
elastic.fit(X_train, y_train)

# Grid Search for optimal parameters
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

grid_search = GridSearchCV(ElasticNet(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
```

---

### Video 67-68: Linear Regression Assumptions

#### 5 Key Assumptions

**1. Linearity**
- Relationship between X and y should be linear
- Check: Plot residuals vs predicted values

```python
# Check linearity
import matplotlib.pyplot as plt

y_pred = model.predict(X)
residuals = y - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()
```

---

**2. No Multicollinearity**
- Features should not be highly correlated with each other
- Check: VIF (Variance Inflation Factor)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
# VIF > 10 indicates high multicollinearity
```

---

**3. Homoscedasticity**
- Residuals should have constant variance
- Check: Plot residuals vs predicted

```
Good (Homoscedastic):       Bad (Heteroscedastic):
    │  • •  •  •               │         • •
    │  •  •  • •               │       •   •
────┼─────────────         ────┼───•──────────
    │ •  •   •  •              │  •  •
    │  • •  •   •              │ • •
    
Constant spread              Spread increases
```

---

**4. Independence of Errors**
- Residuals should be independent (no autocorrelation)
- Check: Durbin-Watson test

```python
from statsmodels.stats.stattools import durbin_watson

# Durbin-Watson test
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson: {dw_stat}")
# Value close to 2 = no autocorrelation
# < 1.5 or > 2.5 = problem
```

---

**5. Normality of Residuals**
- Residuals should follow normal distribution
- Check: Q-Q plot, Shapiro-Wilk test

```python
from scipy import stats
import statsmodels.api as sm

# Q-Q Plot
sm.qqplot(residuals, line='45')
plt.show()

# Shapiro-Wilk test
stat, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk p-value: {p_value}")
# p > 0.05 = residuals are normal
```

---

#### Summary Table: Regression Algorithms

| Algorithm | Use Case | Key Feature |
|-----------|----------|-------------|
| **Linear Regression** | Basic, interpretable | No regularization |
| **Ridge** | Prevent overfitting | Shrinks coefficients |
| **Lasso** | Feature selection | Sparse coefficients |
| **Elastic Net** | Correlated features | Best of both |
| **Polynomial** | Non-linear relationships | Feature transformation |

---


---

## Section 7: Logistic Regression & Classification (Videos 69-79)

### Video 69-70: Introduction to Classification & Perceptron Trick

#### Classification vs Regression

| Aspect | Regression | Classification |
|--------|------------|----------------|
| Output | Continuous (2.5, 3.7, ...) | Discrete (0, 1, 2, ...) |
| Example | Predict salary | Predict spam/not spam |
| Line | Best fit line | Decision boundary |

---

#### Why Not Linear Regression for Classification?

**Problem:** Linear regression outputs can be < 0 or > 1

```
      Probability
         │       ────────  Linear output > 1 ❌
       1 ├───────●───●──
         │      ╱
    0.5  ├────╱──────────
         │  ╱
       0 ├╱───●───●──────
         │────────── ← Linear output < 0 ❌
         └───────────────── X
```

**Solution:** Use Sigmoid function to squash output between 0 and 1

---

#### Perceptron Trick (Intuition)

**Goal:** Find a line that separates two classes.

```
         │    ○ ○
         │  ○   ○ ○
         │ ─────────  ← Decision Boundary
         │ ● ●
         │   ● ● ●
         └─────────────
         
● = Class 0,  ○ = Class 1
```

**Rule:**
- If point is misclassified → Move line towards it
- Repeat until convergence

---

### Video 71: Sigmoid Function

#### Sigmoid Formula

```
              1
σ(z) = ─────────────
        1 + e⁻ᶻ

Where z = wx + b (linear combination)
```

---

#### Sigmoid Properties

```
σ(z)
  1 ├────────────────●●●●●
    │              ●●
    │            ●
0.5 ├──────────●─────────
    │        ●
    │      ●●
  0 ├●●●●●●──────────────
    └────────────────────── z
        -∞     0     +∞
```

| Property | Value |
|----------|-------|
| Range | (0, 1) |
| σ(0) | 0.5 |
| σ(+∞) | → 1 |
| σ(-∞) | → 0 |
| Derivative | σ(z)·(1 - σ(z)) |

---

#### Code Example

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.axhline(y=0.5, color='r', linestyle='--')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()
```

---

### Video 72: Logistic Regression Model

#### How Logistic Regression Works

**Step 1:** Calculate linear combination
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

**Step 2:** Apply sigmoid
```
P(y=1|X) = σ(z) = 1/(1 + e⁻ᶻ)
```

**Step 3:** Make prediction
```
ŷ = 1  if P(y=1) ≥ 0.5
ŷ = 0  if P(y=1) < 0.5
```

---

#### Decision Boundary

**Key Insight:** When σ(z) = 0.5, z = 0

```
z = 0
w₁x₁ + w₂x₂ + b = 0

This is the equation of the decision boundary!
```

**Visualization:**
```
      x₂ │    ○ ○ ○
         │   ○  ○   ← Class 1 (P > 0.5)
         │ ╲
         │  ╲ Decision Boundary
         │   ╲  (wx + b = 0)
         │    ╲
         │  ●  ╲● ● ← Class 0 (P < 0.5)
         │ ● ●  ╲
         └──────────── x₁
```

---

### Video 73: Loss Function - Binary Cross Entropy

#### Why Not MSE for Classification?

**Problem:** MSE creates non-convex loss surface → Multiple local minima

```
MSE Loss Surface:            BCE Loss Surface:
    │  ╭╮  ╭╮                    │╲
    │ ╯  ╰╯ ╰                    │ ╲
    │╯       ╰                   │  ╲
    Multiple minima!             │   ╰──
                                Single minimum ✓
```

---

#### Binary Cross Entropy (Log Loss)

**For single sample:**
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

Where:
- y = actual label (0 or 1)
- ŷ = predicted probability
```

**Intuition:**
| y | ŷ | Loss |
|---|---|------|
| 1 | 0.9 | -log(0.9) = 0.1 (low) |
| 1 | 0.1 | -log(0.1) = 2.3 (high) |
| 0 | 0.1 | -log(0.9) = 0.1 (low) |
| 0 | 0.9 | -log(0.1) = 2.3 (high) |

**For entire dataset:**
```
L = -1/n · Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
```

---

#### Maximum Likelihood Estimation (MLE)

**BCE is derived from maximizing likelihood:**

```
Likelihood = Π P(yᵢ|xᵢ)

Taking log:
Log-Likelihood = Σ log(P(yᵢ|xᵢ))

Maximizing Log-Likelihood = Minimizing -Log-Likelihood
                          = Minimizing BCE
```

---

### Video 74: Gradient Descent for Logistic Regression

#### Gradient Calculation

**Derivative of BCE w.r.t. weights:**
```
∂L/∂wⱼ = 1/n · Σ (ŷᵢ - yᵢ)·xᵢⱼ

∂L/∂b = 1/n · Σ (ŷᵢ - yᵢ)
```

**Note:** Same form as Linear Regression! (Because sigmoid derivative simplifies nicely)

---

#### Update Rules

```
wⱼ = wⱼ - η · ∂L/∂wⱼ
b = b - η · ∂L/∂b
```

---

#### Code from Scratch

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Usage
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

### Video 75: Softmax & Multinomial Logistic Regression

#### Binary vs Multi-class

| Type | Classes | Output |
|------|---------|--------|
| Binary | 2 | Single probability |
| Multi-class | > 2 | Probability for each class |

---

#### Softmax Function

**Converts scores to probabilities:**
```
              e^zᵢ
P(class=i) = ──────
             Σ e^zⱼ

Where zᵢ = wᵢ·x + bᵢ for each class
```

**Properties:**
- All outputs sum to 1
- All outputs between 0 and 1

---

#### Example

```
Scores: z = [2.0, 1.0, 0.5]

Softmax:
P(class 0) = e²·⁰ / (e²·⁰ + e¹·⁰ + e⁰·⁵) = 7.39/11.52 = 0.64
P(class 1) = e¹·⁰ / (e²·⁰ + e¹·⁰ + e⁰·⁵) = 2.72/11.52 = 0.24
P(class 2) = e⁰·⁵ / (e²·⁰ + e¹·⁰ + e⁰·⁵) = 1.65/11.52 = 0.14
                                            ────────
                                            Sum = 1.0
```

---

#### Code Example

```python
from sklearn.linear_model import LogisticRegression

# Multi-class classification
# sklearn automatically uses softmax when more than 2 classes
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Get probabilities for each class
proba = model.predict_proba(X_test)
print("Probabilities shape:", proba.shape)  # (n_samples, n_classes)

# Predict class
y_pred = model.predict(X_test)
```

---

### Video 76: Polynomial Features in Logistic Regression

#### Non-linear Decision Boundaries

**Problem:** Linear decision boundary can't separate some data.

```
Linear Boundary (Bad):       Circular Boundary (Good):
     │    ○ ○                     │    ○ ○
     │  ○ ● ● ○                   │  ○╭──╮○
     │ ○ ● ● ○                    │ ○│●●│○
     │  ○ ● ● ○                   │  ○╰──╯○
     │    ○ ○                     │    ○ ○
```

**Solution:** Add polynomial features!

```
Original: [x₁, x₂]
With degree 2: [x₁, x₂, x₁², x₂², x₁·x₂]
```

---

#### Code Example

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create pipeline with polynomial features
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Now decision boundary can be curved!
```

---

### Video 77: Classification Metrics - Part 1

#### Accuracy

```
                 Correct Predictions      TP + TN
Accuracy = ─────────────────────── = ───────────────
              Total Predictions       TP + TN + FP + FN
```

**Problem:** Misleading for imbalanced data!

---

#### Confusion Matrix

```
                    Predicted
                    0       1
              ┌─────────┬─────────┐
Actual    0   │   TN    │   FP    │
              ├─────────┼─────────┤
          1   │   FN    │   TP    │
              └─────────┴─────────┘

TN = True Negative  (Correct rejection)
FP = False Positive (Type I Error)
FN = False Negative (Type II Error)
TP = True Positive  (Correct detection)
```

---

#### Type I vs Type II Errors

| Error | Description | Example (Spam Detection) |
|-------|-------------|--------------------------|
| **Type I (FP)** | False alarm | Normal email marked as spam |
| **Type II (FN)** | Miss | Spam email goes to inbox |

**Which is worse?** → Depends on the problem!

---

#### Code Example

```python
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualize
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

---

### Video 78: Classification Metrics - Part 2

#### Precision

**"Of all predicted positive, how many were actually positive?"**

```
              TP
Precision = ──────
            TP + FP
```

**Use when:** False Positives are costly (e.g., spam filter)

---

#### Recall (Sensitivity, TPR)

**"Of all actual positive, how many did we catch?"**

```
           TP
Recall = ──────
         TP + FN
```

**Use when:** False Negatives are costly (e.g., cancer detection)

---

#### F1 Score

**Harmonic mean of Precision and Recall:**

```
         2 × Precision × Recall
F1 = ─────────────────────────────
         Precision + Recall
```

**Use when:** You need balance between Precision and Recall

---

#### Precision-Recall Tradeoff

```
           │    Precision
           │╲     ╱
           │ ╲   ╱
           │  ╲ ╱  Recall
           │   ╳
           │  ╱ ╲
           │ ╱   ╲
           └───────────────
           Low    Threshold    High
           
↑ Threshold → ↑ Precision, ↓ Recall
↓ Threshold → ↓ Precision, ↑ Recall
```

---

#### Code Example

```python
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Complete classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### Video 79: ROC-AUC Curve

#### ROC Curve

**ROC = Receiver Operating Characteristic**

```
TPR (Sensitivity)
    │
  1 ├────────────●
    │          ●╱
    │        ●╱
    │      ●╱
    │    ●╱  ROC Curve
    │  ●╱
    │●╱
  0 ├───────────────
    0              1
         FPR (1 - Specificity)
```

**Definitions:**
```
TPR = TP / (TP + FN)  (Recall)
FPR = FP / (FP + TN)  (False Positive Rate)
```

---

#### AUC (Area Under Curve)

| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect classifier |
| 0.9 - 1.0 | Excellent |
| 0.8 - 0.9 | Good |
| 0.7 - 0.8 | Fair |
| 0.5 | Random (no skill) |
| < 0.5 | Worse than random |

---

#### Why Use ROC-AUC?

1. **Threshold independent:** Shows performance across all thresholds
2. **Works with imbalanced data:** Unlike accuracy
3. **Easy to compare models:** Higher AUC = Better model

---

#### Code Example

```python
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# Get predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)
# Or directly: roc_auc = roc_auc_score(y_test, y_proba)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

print(f"AUC Score: {roc_auc:.4f}")
```

---

#### Finding Optimal Threshold

```python
# Find threshold that maximizes (TPR - FPR)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.4f}")

# Use optimal threshold for predictions
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
```

---

#### Summary: When to Use Which Metric?

| Metric | Use When |
|--------|----------|
| **Accuracy** | Balanced classes |
| **Precision** | FP is costly (spam, recommendation) |
| **Recall** | FN is costly (disease, fraud detection) |
| **F1 Score** | Need balance, imbalanced data |
| **AUC-ROC** | Comparing models, probability ranking |

---

#### Logistic Regression Hyperparameters

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l2',           # Regularization: 'l1', 'l2', 'elasticnet', 'none'
    C=1.0,                  # Inverse of regularization strength (small C = strong reg)
    solver='lbfgs',         # Algorithm: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    max_iter=100,           # Maximum iterations
    class_weight=None,      # 'balanced' for imbalanced data
    random_state=42
)
```

| Hyperparameter | Options |
|----------------|---------|
| **C** | Higher = less regularization |
| **penalty** | 'l1' (Lasso), 'l2' (Ridge), 'elasticnet' |
| **solver** | 'liblinear' for L1, 'lbfgs' for L2 |
| **class_weight** | 'balanced' for imbalanced datasets |

---


---

## Section 8: Decision Trees (Videos 80-84)

### Video 80: Decision Tree - Geometric Intuition

#### What is a Decision Tree?

Decision Tree ek **non-parametric, supervised learning** algorithm hai jo data ko recursively split karta hai.

```
                    ┌─────────────────┐
                    │  CGPA >= 7.5?   │  ← Root Node
                    └────────┬────────┘
                   Yes ╱     │      ╲ No
                      ╱      │       ╲
            ┌────────┴──┐    │    ┌───┴────────┐
            │ IQ >= 120? │    │    │  Package   │
            └─────┬─────┘    │    │  = 5 LPA   │
           ╱      │      ╲   │    └────────────┘
        Yes       │      No  │         ↑
         ╱        │        ╲ │      Leaf Node
┌───────┴───┐     │    ┌────┴───┐
│  Package  │     │    │ Package│
│  = 15 LPA │     │    │ = 8 LPA│
└───────────┘     │    └────────┘
    ↑             │
 Leaf Node        │
```

---

#### Key Terminology

| Term | Description |
|------|-------------|
| **Root Node** | Top node, first decision |
| **Internal Node** | Intermediate decisions |
| **Leaf Node** | Final prediction (no children) |
| **Branch** | Connection between nodes |
| **Depth** | Longest path from root to leaf |

---

#### Geometric View (2D Classification)

```
      IQ
       │    
  120 ─┼────────┬────────
       │ Class 1│ Class 1
       │  ●  ●  │  ●  ●
       │────────┤────────
       │ Class 0│ Class 1
       │  ○  ○  │  ●  ●
       └────────┴────────── CGPA
              7.5
              
Decision Tree creates AXIS-PARALLEL boundaries!
```

---

#### Decision Tree for Classification vs Regression

| Type | Output | Leaf Value |
|------|--------|------------|
| **Classification** | Class labels | Majority class in leaf |
| **Regression** | Continuous | Mean of values in leaf |

---

### Video 81: Splitting Criteria - Entropy & Information Gain

#### Entropy (Measure of Impurity)

**Formula:**
```
H(S) = -Σ pᵢ · log₂(pᵢ)

Where pᵢ = proportion of class i in set S
```

**Examples:**
```
Pure set (all same class):     Impure set (50-50):
H = -1·log₂(1) = 0             H = -0.5·log₂(0.5) - 0.5·log₂(0.5) = 1

  ● ● ● ●                        ● ● ○ ○
  ● ● ● ●                        ○ ● ○ ●
  Entropy = 0                    Entropy = 1 (maximum)
```

---

#### Information Gain

**Goal:** Select split that maximizes information gain (reduces entropy most).

```
Information Gain = H(parent) - [weighted average of H(children)]

IG(S, A) = H(S) - Σ (|Sᵥ|/|S|) · H(Sᵥ)

Where:
- S = parent set
- A = attribute to split on
- Sᵥ = subset for each value v of attribute A
```

---

#### Example Calculation

```
Parent: 8 samples (4 Yes, 4 No)
H(parent) = -0.5·log₂(0.5) - 0.5·log₂(0.5) = 1

Split on "Weather":
├── Sunny: 3 samples (2 Yes, 1 No) → H = 0.918
├── Rainy: 3 samples (1 Yes, 2 No) → H = 0.918
└── Cloudy: 2 samples (1 Yes, 1 No) → H = 1

Weighted Avg Entropy = (3/8)·0.918 + (3/8)·0.918 + (2/8)·1 = 0.94

Information Gain = 1 - 0.94 = 0.06
```

---

#### Gini Impurity (Alternative to Entropy)

```
Gini(S) = 1 - Σ pᵢ²

Where pᵢ = proportion of class i
```

**Comparison:**

| Metric | Formula | Range |
|--------|---------|-------|
| Entropy | -Σ pᵢ·log₂(pᵢ) | [0, log₂(k)] |
| Gini | 1 - Σ pᵢ² | [0, 0.5] for binary |

**In practice:** Both give similar results. Gini is faster (no log).

---

### Video 82: Regression Trees

#### How Regression Trees Work

**Splitting Criterion:** Minimize variance (MSE)

```
MSE = (1/n) · Σ(yᵢ - ȳ)²

Variance Reduction = MSE(parent) - weighted_avg(MSE(children))
```

---

#### Prediction

**Leaf value = Mean of training samples in that leaf**

```
         ┌───────────────┐
         │ CGPA >= 7.5?  │
         └───────┬───────┘
        Yes ╱    │    ╲ No
           ╱     │     ╲
    ┌─────┴───┐  │  ┌───┴─────┐
    │IQ >= 120│  │  │ Mean    │
    └────┬────┘  │  │ = 5 LPA │
    ╱    │    ╲  │  └─────────┘
   ╱     │     ╲ │
┌─┴───┐  │  ┌───┴─┐
│Mean │  │  │Mean │
│=15  │  │  │=8   │
└─────┘  │  └─────┘
```

---

#### Code Example

```python
from sklearn.tree import DecisionTreeRegressor

# Create and train
reg_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
reg_tree.fit(X_train, y_train)

# Predict
y_pred = reg_tree.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

---

### Video 83: Decision Tree Hyperparameters

#### Key Hyperparameters

| Parameter | Description | Effect |
|-----------|-------------|--------|
| **max_depth** | Maximum tree depth | ↑ = More complex, overfitting |
| **min_samples_split** | Min samples to split a node | ↑ = More pruning |
| **min_samples_leaf** | Min samples in leaf | ↑ = More pruning |
| **max_features** | Features to consider for split | ↓ = More randomness |
| **criterion** | Splitting metric | 'gini', 'entropy' |

---

#### Overfitting vs Underfitting

```
Underfitting (max_depth=1):       Overfitting (max_depth=∞):
    │                                 │
    │ ○ ○ ○ ● ● ●                     │ ○│○│○│●│●│●
    │───────────                      │──┼─┼─┼─┼─┼──
    │ ○ ○ ○ ● ● ●                     │ ○│○│○│●│●│●
    │                                 │
    Too simple!                       Memorizes noise!
```

---

#### Code Example with Hyperparameter Tuning

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Grid search
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

---

### Video 84: Tree Visualization & Feature Importance

#### Visualization

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Train a tree
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

# Visualize
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['No', 'Yes'],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()
```

---

#### Feature Importance

**How it's calculated:**
- Based on total reduction in impurity (Gini/Entropy) by each feature
- Normalized to sum to 1

```python
# Get feature importance
importance = dt.feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualize
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

---

#### Advantages & Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| ✅ Easy to understand & interpret | ❌ Prone to overfitting |
| ✅ No feature scaling needed | ❌ High variance (small data change = different tree) |
| ✅ Handles both numerical & categorical | ❌ Axis-parallel boundaries only |
| ✅ Feature selection built-in | ❌ Not stable |

**Solution to disadvantages:** Ensemble Methods (Random Forest, Boosting)

---


---

## Section 9: Ensemble Learning (Videos 85-105)

### Video 85: Introduction to Ensemble Learning

#### What is Ensemble Learning?

**"Wisdom of the Crowd"** - Multiple models milkar better predictions dete hain than a single model.

```
                ┌─────────────────────────────────────┐
                │         ENSEMBLE                    │
                │  ┌────────┬────────┬────────┐       │
    Input   ───►│  │Model 1 │Model 2 │Model 3 │ ──────┼──► Final
    Data        │  │  ↓     │   ↓    │   ↓    │       │    Prediction
                │  │ Pred 1 │ Pred 2 │ Pred 3 │       │
                │  └────────┴────────┴────────┘       │
                │              │                      │
                │         Aggregation                 │
                │     (Voting/Averaging)              │
                └─────────────────────────────────────┘
```

---

#### Why Ensembles Work?

**Key Conditions:**
1. **Base models should be diverse** (different algorithms or different data)
2. **Base models should be better than random** (accuracy > 50%)
3. **Errors should be uncorrelated**

**Example:**
```
3 Classifiers, each 70% accurate
If they make INDEPENDENT errors:

P(all 3 wrong) = 0.3³ = 0.027
P(at least 2 correct) = P(ensemble correct) ≈ 78%

Ensemble is better than any individual!
```

---

#### Types of Ensemble Methods

| Type | Approach | Examples |
|------|----------|----------|
| **Voting** | Different algorithms, same data | VotingClassifier |
| **Bagging** | Same algorithm, different data subsets | Random Forest |
| **Boosting** | Sequential, fix errors | AdaBoost, XGBoost |
| **Stacking** | Meta-learner on predictions | StackingClassifier |

---

### Video 86-87: Voting Ensemble

#### Hard Voting (Majority Voting)

**Each model votes for a class, majority wins.**

```
Sample X:
├── Model 1: Class A
├── Model 2: Class B
├── Model 3: Class A
└── Final Prediction: Class A (2 votes)
```

---

#### Soft Voting (Probability Averaging)

**Average probabilities, predict highest probability class.**

```
Sample X:
├── Model 1: P(A)=0.7, P(B)=0.3
├── Model 2: P(A)=0.4, P(B)=0.6
├── Model 3: P(A)=0.9, P(B)=0.1
└── Average: P(A)=0.67, P(B)=0.33
└── Final Prediction: Class A
```

**Soft voting usually better** (uses more information)

---

#### Code Example

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Create base models
log_clf = LogisticRegression()
tree_clf = DecisionTreeClassifier()
svm_clf = SVC(probability=True)  # probability=True for soft voting

# Hard Voting
hard_voting = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('dt', tree_clf),
        ('svc', svm_clf)
    ],
    voting='hard'
)

# Soft Voting
soft_voting = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('dt', tree_clf),
        ('svc', svm_clf)
    ],
    voting='soft'
)

# Train and evaluate
hard_voting.fit(X_train, y_train)
soft_voting.fit(X_train, y_train)

print(f"Hard Voting Accuracy: {hard_voting.score(X_test, y_test):.4f}")
print(f"Soft Voting Accuracy: {soft_voting.score(X_test, y_test):.4f}")
```

---

#### Voting for Regression

```python
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Create voting regressor
voting_reg = VotingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('dt', DecisionTreeRegressor()),
        ('svr', SVR())
    ]
)

voting_reg.fit(X_train, y_train)
y_pred = voting_reg.predict(X_test)
```

---

### Video 88-90: Bagging (Bootstrap Aggregating)

#### Bagging Concept

**Same algorithm + Different data subsets (with replacement)**

```
Original Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
              ↓ Bootstrap Sampling (with replacement)
              
Bootstrap 1: [2, 3, 3, 5, 7, 7, 8, 9, 10, 10] → Model 1
Bootstrap 2: [1, 1, 2, 4, 5, 6, 8, 9, 9, 10]  → Model 2
Bootstrap 3: [1, 3, 4, 4, 5, 6, 6, 7, 8, 10]  → Model 3
                        ↓
                   Aggregation
              ↓                    ↓
      Classification          Regression
      (Majority Vote)         (Average)
```

---

#### Why Bagging Reduces Variance

```
           Single Tree (High Variance):    Bagged Trees (Lower Variance):
                    
Prediction │    ╭╮╭╮                    │    ___________
           │   ╱  ╲╱ ╲                   │   /           \
           │  ╱      ╲                  │  /             \
           │ ╱        ╲                 │ /               \
           └───────────────             └───────────────────
           
           Wiggly, sensitive            Smooth, stable
           to data changes              
```

**Bagging averages out the variance!**

---

#### Bagging Variants

| Variant | Rows Sampled | Features Sampled |
|---------|--------------|------------------|
| **Bagging** | With replacement | All |
| **Pasting** | Without replacement | All |
| **Random Subspaces** | All | Random subset |
| **Random Patches** | Random subset | Random subset |

---

#### Code Example

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier

# Bagging Classifier
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,         # Number of trees
    max_samples=0.8,          # 80% of data per tree
    bootstrap=True,           # With replacement
    oob_score=True,           # Out-of-bag evaluation
    random_state=42,
    n_jobs=-1                 # Parallel processing
)

bagging_clf.fit(X_train, y_train)
print(f"Accuracy: {bagging_clf.score(X_test, y_test):.4f}")
print(f"OOB Score: {bagging_clf.oob_score_:.4f}")

# Bagging Regressor
bagging_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=100,
    random_state=42
)
```

---

#### Out-of-Bag (OOB) Score

**Free validation without separate test set!**

```
Bootstrap sample uses ~63% of data
Remaining ~37% are "Out-of-Bag" samples
Use OOB samples to evaluate each tree
Average OOB errors = OOB Score (≈ CV score)
```

---

### Video 91-93: Random Forest

#### Random Forest = Bagging + Feature Randomness

```
┌─────────────────────────────────────────────────┐
│                  RANDOM FOREST                   │
│                                                  │
│  Bagging (row sampling)                          │
│      +                                           │
│  Feature sampling at each split                  │
│      =                                           │
│  Even more diverse trees!                        │
└─────────────────────────────────────────────────┘
```

---

#### How Random Forest Works

**At each split:**
1. Randomly select `max_features` features
2. Find best split among only those features
3. This increases tree diversity

```
All Features: [CGPA, IQ, Gender, Age, 12th%]

Split 1: Random select [CGPA, Age, 12th%] → Best: CGPA
Split 2: Random select [IQ, Gender, Age]   → Best: IQ
Split 3: Random select [CGPA, Gender, 12th%] → Best: 12th%
```

---

#### Random Forest vs Bagging

| Aspect | Bagging | Random Forest |
|--------|---------|---------------|
| Row sampling | ✅ | ✅ |
| Feature sampling | ❌ (all features) | ✅ (subset at each split) |
| Tree correlation | Higher | Lower |
| Variance reduction | Good | Better |

---

#### Code Example

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,          # Number of trees
    max_depth=10,              # Max depth of each tree
    max_features='sqrt',       # Features per split: 'sqrt', 'log2', int, float
    min_samples_split=5,       # Min samples to split
    min_samples_leaf=2,        # Min samples in leaf
    bootstrap=True,            # Use bootstrap sampling
    oob_score=True,            # Calculate OOB score
    random_state=42,
    n_jobs=-1                  # Parallel processing
)

rf_clf.fit(X_train, y_train)

print(f"Accuracy: {rf_clf.score(X_test, y_test):.4f}")
print(f"OOB Score: {rf_clf.oob_score_:.4f}")

# Feature Importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)
```

---

#### Random Forest Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_estimators` | Number of trees | 100-500 |
| `max_depth` | Tree depth | 5-20 or None |
| `max_features` | Features per split | 'sqrt' for clf, 'auto' for reg |
| `min_samples_split` | Min samples to split | 2-10 |
| `min_samples_leaf` | Min samples in leaf | 1-4 |
| `bootstrap` | Use bootstrap sampling | True |

---

### Video 94-96: Boosting - AdaBoost

#### Boosting Concept

**Sequential learning: Each model fixes errors of previous models**

```
Data → Model 1 → Errors
              ↓
        (Weight errors higher)
              ↓
       Model 2 → Errors
              ↓
        (Weight errors higher)
              ↓
       Model 3 → ...
              ↓
    Final = Weighted combination
```

---

#### Bagging vs Boosting

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Focuses on | Variance reduction | Bias reduction |
| Data sampling | Random with replacement | Weighted (errors get higher weight) |
| Model type | Complex models (deep trees) | Simple models (stumps) |

---

#### AdaBoost Algorithm

**Step 1:** Initialize equal weights for all samples
```
wᵢ = 1/n for all samples
```

**Step 2:** For each iteration t:
1. Train weak learner (decision stump) on weighted data
2. Calculate error rate:
   ```
   εₜ = Σ wᵢ · I(yᵢ ≠ ŷᵢ)  (sum of weights of misclassified)
   ```
3. Calculate model weight (alpha):
   ```
   αₜ = 0.5 · ln((1 - εₜ) / εₜ)
   ```
4. Update sample weights:
   ```
   wᵢ = wᵢ · exp(-αₜ · yᵢ · ŷᵢ)
   Normalize: wᵢ = wᵢ / Σwⱼ
   ```

**Step 3:** Final prediction:
```
F(x) = sign(Σ αₜ · hₜ(x))
```

---

#### Visual Example

```
Iteration 1:           Iteration 2:           Iteration 3:
   ○ ○ ●                  ○ ○ ●                  ○ ○ ●
 ○ ○ ● ●   ───────►   ○ ○ ● ●   ───────►   ○ ○ ● ●
   ○ ● ●                  ○ ● ●                  ○ ● ●
     │                      │                      │
     │                      │                      ╲
  Stump 1               Stump 2               Stump 3
  (vertical)            (horizontal)          (diagonal)
  
Final = Weighted combination of all stumps
```

---

#### Code Example

```python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

# AdaBoost Classifier
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
    n_estimators=50,       # Number of weak learners
    learning_rate=1.0,     # Contribution of each learner
    algorithm='SAMME.R',   # 'SAMME' or 'SAMME.R' (uses probabilities)
    random_state=42
)

ada_clf.fit(X_train, y_train)
print(f"Accuracy: {ada_clf.score(X_test, y_test):.4f}")

# AdaBoost Regressor
ada_reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4),
    n_estimators=50,
    learning_rate=0.5,
    loss='linear',         # 'linear', 'square', 'exponential'
    random_state=42
)
```

---

### Video 97-99: Gradient Boosting

#### Gradient Boosting Concept

**Fit to residuals (negative gradients) instead of reweighting samples**

```
Step 1: F₀(x) = mean(y)  (initial prediction)

Step 2: For m = 1 to M:
    Calculate residuals: rᵢ = yᵢ - Fₘ₋₁(xᵢ)
    Fit tree hₘ to residuals
    Update: Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)
    
Final: F(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ... + η·hₘ(x)
```

---

#### Visual Explanation

```
Step 1: Predict mean
         y │    ● ●
           │  ────────── Mean = 5
           │● ●
           │
         
Step 2: Calculate residuals
         r │  ● ●        (Residuals = y - mean)
         0 ├────────────
           │● ●
           
Step 3: Fit tree to residuals
         r │  ╱────      (Tree learns to predict residuals)
         0 ├─╱
           │────╲
           
Step 4: Update predictions
         F₁ = F₀ + η·h₁  (New prediction is closer to actual)
         
Repeat...
```

---

#### Gradient Boosting for Classification

**Uses pseudo-residuals based on loss gradient:**

```
For log-loss (binary classification):
Pseudo-residual = yᵢ - P(yᵢ = 1)
```

---

#### Code Example

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Shrinkage (step size)
    max_depth=3,           # Depth of each tree
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,         # Fraction of samples for each tree (Stochastic GB)
    max_features=None,     # Features per split
    random_state=42
)

gb_clf.fit(X_train, y_train)
print(f"Accuracy: {gb_clf.score(X_test, y_test):.4f}")

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    loss='squared_error',  # 'squared_error', 'absolute_error', 'huber'
    random_state=42
)

gb_reg.fit(X_train, y_train)
print(f"R² Score: {gb_reg.score(X_test, y_test):.4f}")
```

---

### Video 100-103: XGBoost (Extreme Gradient Boosting)

#### What is XGBoost?

**XGBoost = Gradient Boosting + Software Optimizations + Regularization**

```
XGBoost Advantages:
├── Performance (regularization, better splits)
├── Speed (parallel processing, cache optimization)
└── Flexibility (custom objectives, missing values)
```

---

#### XGBoost vs Gradient Boosting

| Feature | Gradient Boosting | XGBoost |
|---------|-------------------|---------|
| Regularization | None | L1 + L2 |
| Parallel processing | No | Yes (at split level) |
| Missing values | Manual handling | Built-in handling |
| Tree pruning | Pre-pruning | Post-pruning (max_depth then prune) |
| Second-order derivatives | No | Yes (Newton's method) |
| Speed | Slow | 10x faster |

---

#### XGBoost Objective Function

```
Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)
      ↑              ↑
    Loss         Regularization

Where Ω(f) = γT + (1/2)λ‖w‖²

- T = number of leaves
- w = leaf weights
- γ = penalty for number of leaves (L1)
- λ = penalty for leaf weights (L2)
```

---

#### Key XGBoost Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_estimators` | Number of trees | 100-1000 |
| `max_depth` | Tree depth | 3-10 |
| `learning_rate` (eta) | Step size | 0.01-0.3 |
| `subsample` | Row sampling ratio | 0.5-1.0 |
| `colsample_bytree` | Column sampling | 0.5-1.0 |
| `lambda` (reg_lambda) | L2 regularization | 0-10 |
| `alpha` (reg_alpha) | L1 regularization | 0-10 |
| `gamma` | Min loss reduction for split | 0-5 |

---

#### Code Example - XGBoost for Regression

```python
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Initialize XGBRegressor
xgb_reg = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,          # L2 regularization
    reg_alpha=0,           # L1 regularization
    random_state=42
)

# Train
xgb_reg.fit(X_train, y_train)

# Predict
y_pred = xgb_reg.predict(X_test)

# Evaluate
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
```

---

#### Code Example - XGBoost for Classification

```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize XGBClassifier
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',  # or 'multi:softmax' for multi-class
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Train
xgb_clf.fit(X_train, y_train)

# Predict
y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

#### Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

grid_search = GridSearchCV(
    xgb, param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

---

#### Feature Importance in XGBoost

```python
import matplotlib.pyplot as plt

# Get feature importance
importance = xgb_clf.feature_importances_

# Plot
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importance)
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()

# Alternative: using plot_importance
from xgboost import plot_importance
plot_importance(xgb_clf, max_num_features=10)
plt.show()
```

---

### Video 104-105: Stacking

#### Stacking Concept

**Use predictions of base models as features for meta-learner**

```
        ┌─────────────────────────────────────────────────┐
        │               STACKING                           │
        │                                                  │
        │  ┌────────┬────────┬────────┐                   │
  X  ───┼─►│Model 1 │Model 2 │Model 3 │                   │
        │  │   ↓    │   ↓    │   ↓    │                   │
        │  │ Pred 1 │ Pred 2 │ Pred 3 │                   │
        │  └───┬────┴───┬────┴───┬────┘                   │
        │      │        │        │                        │
        │      └────────┼────────┘                        │
        │               ↓                                 │
        │        [P1, P2, P3]  ← New Features             │
        │               ↓                                 │
        │     ┌─────────────────┐                        │
        │     │  Meta-Learner   │                        │──► Final
        │     │ (e.g., LogReg)  │                        │    Prediction
        │     └─────────────────┘                        │
        └─────────────────────────────────────────────────┘
```

---

#### How to Avoid Overfitting in Stacking

**Use K-Fold predictions for training meta-learner:**

```
Fold 1: Train on [2,3,4,5], Predict on [1]
Fold 2: Train on [1,3,4,5], Predict on [2]
Fold 3: Train on [1,2,4,5], Predict on [3]
Fold 4: Train on [1,2,3,5], Predict on [4]
Fold 5: Train on [1,2,3,4], Predict on [5]

Combine predictions → Use as meta-learner input
```

---

#### Code Example

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define base models
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('svc', SVC(kernel='rbf', probability=True))
]

# Define meta-learner
meta_learner = LogisticRegression()

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,                    # K-fold for base model predictions
    stack_method='auto',     # 'auto', 'predict_proba', 'predict'
    n_jobs=-1
)

# Train
stacking_clf.fit(X_train, y_train)

# Evaluate
print(f"Stacking Accuracy: {stacking_clf.score(X_test, y_test):.4f}")
```

---

#### Stacking for Regression

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

base_models = [
    ('rf', RandomForestRegressor(n_estimators=50)),
    ('gb', GradientBoostingRegressor(n_estimators=50)),
    ('xgb', XGBRegressor(n_estimators=50))
]

stacking_reg = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(),
    cv=5
)

stacking_reg.fit(X_train, y_train)
print(f"R² Score: {stacking_reg.score(X_test, y_test):.4f}")
```

---

#### Blending (Simplified Stacking)

**Difference from Stacking:** Use holdout set instead of K-fold

```python
# Manual Blending
from sklearn.model_selection import train_test_split

# Split training data into two parts
X_train_blend, X_val_blend, y_train_blend, y_val_blend = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train base models on train_blend
model1 = RandomForestClassifier().fit(X_train_blend, y_train_blend)
model2 = GradientBoostingClassifier().fit(X_train_blend, y_train_blend)

# Get predictions on val_blend (for meta-learner training)
pred1 = model1.predict_proba(X_val_blend)[:, 1]
pred2 = model2.predict_proba(X_val_blend)[:, 1]

# Create meta features
meta_features = np.column_stack([pred1, pred2])

# Train meta-learner
meta_learner = LogisticRegression()
meta_learner.fit(meta_features, y_val_blend)

# For final predictions:
# 1. Get base model predictions on test set
# 2. Stack them
# 3. Pass to meta-learner
```

---

#### Summary: Ensemble Methods Comparison

| Method | Training | Reduces | Best For |
|--------|----------|---------|----------|
| **Voting** | Parallel | - | Combining different algorithms |
| **Bagging** | Parallel | Variance | High-variance models (trees) |
| **Random Forest** | Parallel | Variance | General purpose, interpretable |
| **AdaBoost** | Sequential | Bias | Weak learners |
| **Gradient Boosting** | Sequential | Bias | Structured data |
| **XGBoost** | Sequential | Bias | Competitions, production |
| **Stacking** | Multi-level | Both | Maximum performance |

---

#### When to Use Which?

```
Start with:
├── Random Forest (good baseline)
│
├── If RF not enough:
│   ├── XGBoost (usually best)
│   └── LightGBM (faster for large data)
│
├── If need interpretation:
│   └── Single Decision Tree or RF feature importance
│
└── For competitions:
    └── Stacking/Blending of XGBoost, LightGBM, CatBoost
```

---


---

## Section 10: Clustering (Videos 106-114)

### Video 106: Introduction to Clustering

#### What is Clustering?

**Unsupervised Learning:** Find natural groups in data without labels.

```
Before Clustering:              After Clustering:
    •  •      •                   ○  ○      ●
  •    •    •   •               ○    ○    ●   ●
    •  •      •  •                ○  ○      ●  ●
        •   •                         △   △
          •  •                          △  △
```

---

#### Types of Clustering

| Type | Method | Example |
|------|--------|---------|
| **Partitioning** | Divide into k clusters | K-Means |
| **Hierarchical** | Build tree of clusters | Agglomerative |
| **Density-based** | Find dense regions | DBSCAN |

---

#### Applications

- Customer Segmentation
- Document Clustering
- Image Segmentation
- Anomaly Detection
- Recommendation Systems

---

### Video 107-109: K-Means Clustering

#### K-Means Algorithm

**Goal:** Partition n points into k clusters by minimizing within-cluster variance.

```
Step 1: Initialize k centroids randomly
Step 2: Assign each point to nearest centroid
Step 3: Recalculate centroids as mean of assigned points
Step 4: Repeat 2-3 until convergence
```

---

#### Visual Example

```
Iteration 1:                    Iteration 2:                    Converged:
    •  •     +                      ○  ○     +                      ○  ○     ○
  •    •  x   •                   ○    ○  x   ●                   ○    ○       ●
    •  •     •  •                   ○  ○     ●  ●                   ○  ○     ●  ●
                                            
+ = Centroid 1                  Centroids moved              Final clusters
x = Centroid 2                  Points reassigned
```

---

#### Mathematical Formulation

**Objective:** Minimize Within-Cluster Sum of Squares (WCSS)

```
WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²

Where:
- Cₖ = cluster k
- μₖ = centroid of cluster k
- ||xᵢ - μₖ||² = squared Euclidean distance
```

---

#### Code Example

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,           # Number of clusters
    init='k-means++',       # Smart initialization
    n_init=10,              # Number of initializations
    max_iter=300,           # Max iterations
    random_state=42
)

# Fit and predict
y_pred = kmeans.fit_predict(X)

# Get cluster centers
centroids = kmeans.cluster_centers_

# Get inertia (WCSS)
print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")

# Visualize (2D)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
plt.title('K-Means Clustering')
plt.show()
```

---

#### Elbow Method (Finding Optimal K)

```python
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method')
plt.show()
```

```
WCSS │
     │╲
     │ ╲
     │  ╲___  ← Elbow point (optimal k)
     │      ╲____
     │           ╲___
     └────────────────── k
         3 (optimal)
```

---

#### K-Means Limitations

| Limitation | Description |
|------------|-------------|
| Need to specify k | Must know clusters beforehand |
| Sensitive to initialization | Different runs → different results |
| Assumes spherical clusters | Can't handle non-convex shapes |
| Sensitive to outliers | Outliers pull centroids |
| Requires numerical features | Can't handle categorical directly |

---

### Video 110-111: Hierarchical Clustering

#### Agglomerative (Bottom-Up) Approach

```
Step 1: Each point is a cluster
Step 2: Find two closest clusters
Step 3: Merge them
Step 4: Repeat until one cluster remains
```

```
    A   B   C   D   E          Start: 5 clusters
    ●   ●   ●   ●   ●
    └─┬─┘   │   └─┬─┘          Merge closest
      AB    C    DE
      └──┬──┘    │             Merge next
        ABC     DE
         └───┬───┘             Final merge
           ABCDE
```

---

#### Linkage Methods

| Method | Distance Between Clusters |
|--------|---------------------------|
| **Single** | Min distance between any two points |
| **Complete** | Max distance between any two points |
| **Average** | Average distance between all pairs |
| **Ward** | Minimize variance increase |

```
Single (chaining problem):    Complete (compact):
     ●──●──●──●                    ●──●
                                   │  │
                                   ●──●
```

---

#### Dendrogram

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Calculate linkage
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.axhline(y=150, color='r', linestyle='--')  # Cut line for k clusters
plt.show()
```

---

#### Code Example

```python
from sklearn.cluster import AgglomerativeClustering

# Create model
agg_clustering = AgglomerativeClustering(
    n_clusters=3,            # Number of clusters
    linkage='ward',          # 'ward', 'complete', 'average', 'single'
    metric='euclidean'       # Distance metric
)

# Fit and predict
y_pred = agg_clustering.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.show()
```

---

### Video 112-114: DBSCAN

#### DBSCAN Concept

**Density-Based Spatial Clustering of Applications with Noise**

```
Parameters:
- eps (ε): Maximum distance between two points to be neighbors
- min_samples: Minimum points to form a dense region
```

---

#### Point Types

| Type | Definition |
|------|------------|
| **Core Point** | Has ≥ min_samples within eps |
| **Border Point** | Within eps of core point, but not core itself |
| **Noise Point** | Neither core nor border |

```
      ε radius
    ┌─────────┐
    │  • • •  │ ← Core point (5 neighbors)
    │• ●─────●│
    │  • •    │
    └─────────┘
         ↑
      Border point (2 neighbors, within eps of core)
      
    ○  ← Noise point (no neighbors)
```

---

#### DBSCAN Algorithm

```
For each point p:
    1. Find all points within eps distance
    2. If count ≥ min_samples → Core point
       - Create cluster or expand existing
    3. Else if within eps of core → Border point
    4. Else → Noise point
```

---

#### Advantages over K-Means

| Aspect | K-Means | DBSCAN |
|--------|---------|--------|
| Number of clusters | Must specify | Automatic |
| Cluster shapes | Only spherical | Any shape |
| Outlier handling | No | Yes (noise points) |

```
K-Means fails:              DBSCAN works:
    ○○○●●●                     ○○○●●●
    ○  ○●●●                     ○  ○●●●
    ○○○                        ○○○   
       ●●●                        ●●●
```

---

#### Code Example

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Scale data (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create DBSCAN model
dbscan = DBSCAN(
    eps=0.5,               # Maximum distance
    min_samples=5,         # Minimum points for core
    metric='euclidean'     # Distance metric
)

# Fit and predict
y_pred = dbscan.fit_predict(X_scaled)

# Number of clusters (excluding noise)
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
n_noise = list(y_pred).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Visualize
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis')
plt.title(f'DBSCAN: {n_clusters} clusters found')
plt.show()
```

---

#### Finding Optimal eps

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Calculate k-nearest neighbor distances
k = 5  # Same as min_samples
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)

# Sort distances to k-th neighbor
k_distances = np.sort(distances[:, k-1])

# Plot
plt.plot(k_distances)
plt.xlabel('Points')
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.title('K-Distance Plot (Elbow = optimal eps)')
plt.show()
```

---

#### Clustering Evaluation Metrics

**With ground truth labels:**
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
```

**Without ground truth (internal metrics):**
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Silhouette Score (-1 to 1, higher is better)
silhouette = silhouette_score(X, y_pred)

# Calinski-Harabasz Index (higher is better)
ch_score = calinski_harabasz_score(X, y_pred)
```

---

#### Summary: Clustering Algorithms

| Algorithm | Pros | Cons | Use When |
|-----------|------|------|----------|
| **K-Means** | Fast, scalable | Needs k, spherical only | Known k, spherical clusters |
| **Hierarchical** | Dendrogram, any k | Slow for large data | Small data, need hierarchy |
| **DBSCAN** | Auto k, any shape, outliers | Sensitive to params | Unknown k, non-spherical |

---


---

## Section 11: Support Vector Machines (Videos 115-118)

### Video 115: SVM - Geometric Intuition

#### What is SVM?

**Support Vector Machine:** Find the hyperplane that best separates classes with maximum margin.

```
         │    ○ ○ ○
         │   ○  ○
         │ ╱─────────  ← Decision Boundary (Hyperplane)
         │╱  Support Vectors
         │     ●
         │   ● ● ●
         │  ●   ●
         └─────────────
         
Margin = Distance between hyperplane and nearest points
Goal: MAXIMIZE the margin
```

---

#### Key Concepts

| Term | Definition |
|------|------------|
| **Hyperplane** | Decision boundary (line in 2D, plane in 3D) |
| **Margin** | Distance between hyperplane and closest points |
| **Support Vectors** | Points closest to hyperplane (define the margin) |

---

#### Why Maximum Margin?

```
Small Margin:                  Large Margin (Better):
     │    ○ ○                       │    ○ ○
     │  ○○╱                         │  ○  ╱
     │  ╱ ●●                        │   ╱
     │╱  ● ●                        │  ╱  ● ●
     │                              │ ╱    ●
                                    
More likely to               More robust to
misclassify new data         new data points
```

---

#### Hard Margin vs Soft Margin

**Hard Margin SVM:**
- All points must be correctly classified
- No points inside margin
- Only works for linearly separable data

**Soft Margin SVM:**
- Allows some misclassifications
- Uses penalty parameter C
- Works for non-linearly separable data

---

### Video 116: SVM Mathematics

#### Hard Margin Formulation

**Objective:** Maximize margin = Minimize ||w||

```
Minimize: (1/2)||w||²

Subject to: yᵢ(w·xᵢ + b) ≥ 1  for all i

Where:
- w = weight vector (normal to hyperplane)
- b = bias
- yᵢ = class label (+1 or -1)
```

---

#### Soft Margin Formulation

```
Minimize: (1/2)||w||² + C·Σξᵢ

Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
           ξᵢ ≥ 0

Where:
- C = penalty for misclassification
- ξᵢ = slack variable (how much point violates margin)
```

**Effect of C:**

| C Value | Effect |
|---------|--------|
| C → ∞ | Hard margin (no violations allowed) |
| C large | Small margin, few violations |
| C small | Large margin, more violations allowed |

---

#### Code Example (Linear SVM)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Scale features (important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM
svm_linear = SVC(
    kernel='linear',
    C=1.0,              # Regularization parameter
    random_state=42
)

svm_linear.fit(X_train_scaled, y_train)
print(f"Accuracy: {svm_linear.score(X_test_scaled, y_test):.4f}")

# Get support vectors
print(f"Number of support vectors: {len(svm_linear.support_vectors_)}")
```

---

### Video 117: Kernel Trick

#### Non-Linear Classification Problem

```
Cannot separate with line:      Can separate in higher dimension:
      │  ○ ● ○                         z │
      │ ○ ● ● ○                          │    ○   ○
      │  ○ ● ○                           │  ● ● ●
      └─────────                         └───────────
                                              x²
Transform x → (x, x²)
```

---

#### Kernel Functions

**Kernel = Dot product in higher-dimensional space without explicit transformation**

| Kernel | Formula | Use Case |
|--------|---------|----------|
| **Linear** | K(x,y) = x·y | Linearly separable |
| **Polynomial** | K(x,y) = (γx·y + r)^d | Polynomial boundaries |
| **RBF (Gaussian)** | K(x,y) = exp(-γ\|\|x-y\|\|²) | Most common, flexible |
| **Sigmoid** | K(x,y) = tanh(γx·y + r) | Neural network-like |

---

#### RBF Kernel (Radial Basis Function)

```
K(x, y) = exp(-γ||x - y||²)

Where γ = 1/(2σ²)
```

**Effect of γ (gamma):**

| γ Value | Effect |
|---------|--------|
| γ small | Smooth decision boundary (low variance) |
| γ large | Complex boundary (high variance, overfitting) |

```
Small γ:                    Large γ:
    │ ○○   ╱○○                 │ ○○  ╭╮ ○○
    │ ○○  ╱ ○                  │ ○○ ╯ ╰─╮○
    │────╱───                  │ ○ ╭──╯ ╰─
    │ ●●╱● ●                   │ ●╯● ●╭─╯
    
Underfitting                Overfitting
```

---

#### Code Example (Non-Linear SVM)

```python
from sklearn.svm import SVC

# RBF Kernel (most common)
svm_rbf = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',        # 'scale', 'auto', or float
    random_state=42
)

svm_rbf.fit(X_train_scaled, y_train)
print(f"RBF SVM Accuracy: {svm_rbf.score(X_test_scaled, y_test):.4f}")

# Polynomial Kernel
svm_poly = SVC(
    kernel='poly',
    degree=3,             # Polynomial degree
    C=1.0,
    gamma='scale',
    coef0=1,              # Independent term (r)
    random_state=42
)

svm_poly.fit(X_train_scaled, y_train)
print(f"Polynomial SVM Accuracy: {svm_poly.score(X_test_scaled, y_test):.4f}")
```

---

### Video 118: SVM Hyperparameter Tuning

#### Key Hyperparameters

| Parameter | Description | Effect |
|-----------|-------------|--------|
| **C** | Regularization | High C → Less regularization |
| **kernel** | Kernel function | 'linear', 'rbf', 'poly', 'sigmoid' |
| **gamma** | Kernel coefficient | High γ → Complex boundary |
| **degree** | Polynomial degree | Only for poly kernel |

---

#### GridSearchCV for SVM

```python
from sklearn.model_selection import GridSearchCV

# Parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'linear']
}

# Grid search
svm = SVC(random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Use best model
best_svm = grid_search.best_estimator_
print(f"Test Accuracy: {best_svm.score(X_test_scaled, y_test):.4f}")
```

---

#### SVM for Regression (SVR)

```python
from sklearn.svm import SVR

svr = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1,          # Tube width (no penalty within tube)
    gamma='scale'
)

svr.fit(X_train_scaled, y_train)
y_pred = svr.predict(X_test_scaled)

from sklearn.metrics import r2_score
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
```

---

#### SVM Advantages & Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| ✅ Effective in high dimensions | ❌ Slow for large datasets |
| ✅ Memory efficient (support vectors) | ❌ Requires feature scaling |
| ✅ Versatile (kernel trick) | ❌ No probability estimates (by default) |
| ✅ Works well with clear margin | ❌ Not good for noisy data |

---


---

## Section 12: Naive Bayes (Videos 119-125)

### Video 119-121: Probability Basics

#### Conditional Probability

```
P(A|B) = Probability of A given B has occurred

            P(A ∩ B)
P(A|B) = ───────────
            P(B)
```

**Example:**
```
Email Dataset:
- 60% are spam
- 80% of spam contains "free"
- 10% of non-spam contains "free"

P(Spam | contains "free") = ?
```

---

#### Bayes' Theorem

```
              P(B|A) · P(A)
P(A|B) = ─────────────────────
                P(B)

Or with evidence expansion:

              P(B|A) · P(A)
P(A|B) = ────────────────────────────────────
         P(B|A)·P(A) + P(B|¬A)·P(¬A)
```

---

#### Example Calculation

```
Given:
- P(Spam) = 0.60
- P(Not Spam) = 0.40
- P("free" | Spam) = 0.80
- P("free" | Not Spam) = 0.10

P(Spam | "free") = P("free"|Spam)·P(Spam) / P("free")

P("free") = P("free"|Spam)·P(Spam) + P("free"|Not Spam)·P(Not Spam)
          = 0.80 × 0.60 + 0.10 × 0.40
          = 0.48 + 0.04 = 0.52

P(Spam | "free") = (0.80 × 0.60) / 0.52 = 0.923
```

---

### Video 122-123: Naive Bayes Classifier

#### Why "Naive"?

**Assumption:** Features are conditionally independent given the class.

```
P(x₁, x₂, x₃ | Class) = P(x₁|Class) · P(x₂|Class) · P(x₃|Class)

This is "naive" because features are rarely truly independent!
But it works surprisingly well in practice.
```

---

#### Naive Bayes Formula

```
P(Class | X) ∝ P(Class) · ∏ P(xᵢ | Class)
                           i

Choose class with highest posterior probability:
ŷ = argmax P(Class) · ∏ P(xᵢ | Class)
     Class             i
```

---

#### Example: Spam Classification

```
Email: "Get free money now"
Features: ["get", "free", "money", "now"]

P(Spam | email) ∝ P(Spam) · P("get"|Spam) · P("free"|Spam) · P("money"|Spam) · P("now"|Spam)

P(Not Spam | email) ∝ P(Not Spam) · P("get"|Not Spam) · P("free"|Not Spam) · ...

Compare probabilities → Higher one wins
```

---

#### Types of Naive Bayes

| Type | Feature Type | Distribution |
|------|--------------|--------------|
| **Gaussian NB** | Continuous | Normal distribution |
| **Multinomial NB** | Discrete (counts) | Multinomial distribution |
| **Bernoulli NB** | Binary | Bernoulli distribution |
| **Categorical NB** | Categorical | Categorical distribution |

---

### Video 124: Gaussian Naive Bayes

#### For Continuous Features

**Assumption:** Features follow Gaussian (normal) distribution within each class.

```
                    (xᵢ - μc)²
                   ──────────
                      2σc²
           1
P(xᵢ|c) = ────── · e
          √(2πσc²)

Where:
- μc = mean of feature i for class c
- σc = std dev of feature i for class c
```

---

#### Code Example

```python
from sklearn.naive_bayes import GaussianNB

# Gaussian Naive Bayes (for continuous features)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### Video 125: Multinomial & Bernoulli Naive Bayes

#### Multinomial Naive Bayes

**For count data (e.g., word frequencies in text)**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Text classification example
texts = ["free money now", "meeting tomorrow", "get free offer", ...]
labels = [1, 0, 1, ...]  # 1 = spam, 0 = not spam

# Convert text to count vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train Multinomial NB
mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
mnb.fit(X_train, y_train)

# Predict
y_pred = mnb.predict(X_test)
```

---

#### Bernoulli Naive Bayes

**For binary features (presence/absence)**

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

# Convert to binary (word present or not)
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

# Train Bernoulli NB
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train, y_train)
```

---

#### Laplace Smoothing (alpha)

**Problem:** Zero probability for unseen words

```
P("unseen_word" | Spam) = 0

Then: P(Spam | email with "unseen_word") = 0  (regardless of other words!)
```

**Solution:** Add smoothing parameter α

```
              count(word, class) + α
P(word|class) = ─────────────────────────────
               total_words_in_class + α·|V|

Where |V| = vocabulary size
α = 1 is Laplace smoothing
```

---

#### Complete Example: Spam Classification

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample data
emails = [
    "Get free money now",
    "Meeting at 3pm tomorrow",
    "Win a free iPhone",
    "Project deadline extended",
    "Click here for free gift",
    "Lunch meeting confirmed"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = spam

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(emails)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3)

# Train
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train, y_train)

# Evaluate
y_pred = nb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Predict new email
new_email = ["Get your free prize now"]
new_X = tfidf.transform(new_email)
prediction = nb.predict(new_X)
print(f"New email is: {'Spam' if prediction[0] == 1 else 'Not Spam'}")
```

---

#### Naive Bayes Advantages & Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| ✅ Fast training and prediction | ❌ Assumes feature independence |
| ✅ Works well with high dimensions | ❌ Not good for continuous features |
| ✅ Works well with small data | ❌ Probability estimates not reliable |
| ✅ Good for text classification | ❌ Can't learn feature interactions |

---

#### When to Use Naive Bayes?

```
Good For:
├── Text classification (spam, sentiment)
├── Small datasets
├── Real-time prediction (fast)
├── Multi-class problems
└── When features are reasonably independent

Not Good For:
├── Features with strong correlations
├── Complex relationships
└── Numerical feature heavy problems
```

---

## Quick Reference: Algorithm Selection Guide

### Classification Algorithms

| Algorithm | Best For | Complexity |
|-----------|----------|------------|
| **Logistic Regression** | Baseline, interpretable | Low |
| **Naive Bayes** | Text, small data | Very Low |
| **Decision Tree** | Interpretable rules | Medium |
| **Random Forest** | General purpose | Medium-High |
| **XGBoost** | Best performance | High |
| **SVM** | High-dimensional, clear margin | Medium |
| **KNN** | Simple, non-linear | Low |

### Regression Algorithms

| Algorithm | Best For | Complexity |
|-----------|----------|------------|
| **Linear Regression** | Baseline, interpretable | Very Low |
| **Ridge/Lasso** | Regularization needed | Low |
| **Decision Tree** | Non-linear, interpretable | Medium |
| **Random Forest** | General purpose | Medium-High |
| **XGBoost** | Best performance | High |
| **SVR** | Small data, non-linear | Medium |

### Clustering Algorithms

| Algorithm | Best For |
|-----------|----------|
| **K-Means** | Spherical clusters, known k |
| **Hierarchical** | Dendrogram, small data |
| **DBSCAN** | Arbitrary shapes, unknown k |

---

## End of Notes

These comprehensive notes cover the "100 Days of Machine Learning" course from CampusX. Key topics include:

1. ✅ ML Fundamentals & Types
2. ✅ Data Handling & EDA
3. ✅ Feature Engineering
4. ✅ Missing Data & Outliers
5. ✅ Dimensionality Reduction (PCA)
6. ✅ Linear Models & Regression
7. ✅ Logistic Regression & Classification
8. ✅ Decision Trees
9. ✅ Ensemble Methods (Bagging, Boosting, XGBoost, Stacking)
10. ✅ Clustering (K-Means, Hierarchical, DBSCAN)
11. ✅ Support Vector Machines
12. ✅ Naive Bayes

**Pro Tips:**
- Always start with simple models (baseline)
- Feature engineering > Complex models
- Cross-validation is essential
- Ensemble methods usually win competitions
- XGBoost/LightGBM for structured data
- Deep Learning for unstructured data (images, text)

---
