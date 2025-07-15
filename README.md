# 🆘 Disaster Tweet Classification

This NLP project focuses on classifying whether a tweet is about a real disaster or not. It was based on a Kaggle competition with over 7,500 labeled tweets and involved building a full text classification pipeline — from preprocessing raw text and engineering features to model training and evaluation.

---

## 📌 Problem Statement

Given a tweet, predict whether it refers to a real disaster (`1`) or not (`0`).  
The dataset consists of approximately 7,500 tweets:
- ~4,300 non-disaster (class 0)
- ~3,200 disaster (class 1)

Classes are relatively balanced, allowing for robust model training.

---

## 🔍 Project Pipeline

### 1. 🧠 Exploratory Data Analysis (EDA)
- Inspected data shape, nulls, and types using `pandas`
- Visualized target distribution using bar plots
- Confirmed dataset was not heavily imbalanced
- Generated a correlation heatmap for numeric/engineered features

### 2. 🧼 Text Preprocessing & Feature Engineering
- Removed URLs, mentions, hashtags, punctuation, and special characters
- Converted to lowercase, removed stopwords (using `nltk`)
- Lemmatized text using `spaCy`
- Engineered additional features:
  - Word count
  - Average word length
  - Number of stop words
  - Count of special characters
- Explored sentiment analysis using VADER, TextBlob, and spaCy
  - Chose **spaCy sentiment** for its better class separation

### 3. 🔁 Feature Vectorization & Selection
- Vectorized cleaned text using **TF-IDF**
- Scaled numerical features using `StandardScaler`
- Combined text and numeric features using sparse matrix stacking
- Applied **chi-squared test** using `SelectKBest` to retain top 2,000 TF-IDF features

### 4. 🤖 Model Building & Evaluation
Trained and evaluated 3 classification models:

| Model              | Train F1 | Test F1 | Notes                   |
|-------------------|----------|---------|-------------------------|
| Logistic Regression | 0.81     | 0.76    | ✅ Best generalization   |
| Random Forest       | 0.90     | 0.72    | Overfit                 |
| MLPClassifier       | 0.96     | 0.72    | High overfitting        |

- Evaluation metric: **F1-score**
- Logistic Regression chosen as final model for its balance of performance and simplicity

---

## 🛠️ Tools & Libraries

- **Python**: Core language
- **Pandas**, **NumPy**: Data handling
- **Matplotlib**, **Seaborn**: Visualization
- **NLTK**, **spaCy**: Text preprocessing (stopword removal, lemmatization)
- **TextBlob**, **VADER**: Sentiment scoring (tested)
- **Scikit-learn**: TF-IDF, feature selection, models, evaluation

---

## ⚠️ Challenges Faced

- Handling noisy, unstructured text
- Overfitting with complex models (Random Forest, MLP)
- Deciding between multiple sentiment extraction tools

### ✅ Solutions
- Used custom text cleaner functions
- Engineered new numerical features to capture tweet characteristics
- Applied chi-squared feature selection to reduce dimensionality and overfitting

---

## ✅ Final Outcome

- Built a strong pipeline combining NLP + numeric feature engineering
- Achieved **F1-score of 0.76** on unseen data using Logistic Regression
- Gained deep understanding of end-to-end text classification workflows
- Strengthened confidence in model evaluation and generalization techniques

---

## 📎 Dataset

- **Source**: [Kaggle – NLP Getting Started Competition](https://www.kaggle.com/competitions/nlp-getting-started)
- 📌 Dataset **not included** in this repository due to Kaggle’s competition rules  
- Please download from Kaggle directly after logging in

---

## 👤 Author

**Pawan Solanki**  
3rd Year BS in Data Science & Applications  
Indian Institute of Technology, Madras

---

## 🙌 Acknowledgements

- Dataset & competition hosted on Kaggle
- Thanks to the Kaggle community for discussions and resources