# SMS Spam Detection

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using natural language processing (NLP) and supervised machine learning.

## 📌 Project Overview

Spam messages are unsolicited messages that may contain advertisements, scams, phishing attempts, fraudulent offers, or other unwanted content. Automatically identifying these messages can help improve messaging security and user experience.

This project builds an end-to-end SMS spam detection pipeline that:

* Loads and explores an SMS message dataset
* Cleans and preprocesses text data
* Converts text into numerical features using TF-IDF
* Trains machine learning classification models
* Evaluates model performance
* Predicts whether new SMS messages are spam or legitimate

## 🎯 Objectives

The main objectives of this project are to:

1. Understand the characteristics of spam and legitimate SMS messages.
2. Perform text preprocessing and normalization.
3. Convert textual data into machine-readable numerical features.
4. Train classification models for spam detection.
5. Evaluate models using appropriate classification metrics.
6. Build a reusable prediction pipeline for new SMS messages.

## 🗂️ Dataset

This project can be implemented using the **SMS Spam Collection Dataset**, a commonly used dataset for SMS spam classification.

The dataset contains SMS messages labeled as either:

* `ham` — legitimate/non-spam message
* `spam` — unwanted or spam message

A typical dataset contains two columns:

| Column    | Description                        |
| --------- | ---------------------------------- |
| `label`   | Class of the SMS (`ham` or `spam`) |
| `message` | Text content of the SMS            |

Example:

```text
label,message
ham,"Hey, are we still meeting today?"
spam,"Congratulations! You have won a free prize. Call now!"
```

## 🧰 Technologies Used

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical operations
* **Matplotlib / Seaborn** — data visualization
* **NLTK** — natural language processing
* **Scikit-learn** — machine learning
* **Jupyter Notebook** — experimentation and analysis

## 📁 Project Structure

```text
sms-spam-detection/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   └── sms_spam_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── spam_classifier.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The exact structure may vary depending on the implementation.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ishanegi7/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the main dependencies with:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn jupyter
```

## 📊 Exploratory Data Analysis

Before training the model, the dataset is analyzed to understand:

* Number of spam and ham messages
* Distribution of message lengths
* Most frequently occurring words
* Class imbalance
* Common patterns in spam messages

Example:

```python
import pandas as pd

df = pd.read_csv("data/spam.csv")

print(df.head())
print(df["label"].value_counts())
print(df.info())
```

A class distribution visualization can also be created:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x="label")
plt.title("Spam vs Ham Messages")
plt.show()
```

## 🧹 Text Preprocessing

Raw SMS messages contain punctuation, numbers, URLs, stop words, capitalization, and other information that may not be useful in their original form.

The preprocessing pipeline may include:

1. Converting text to lowercase
2. Removing punctuation
3. Removing unnecessary numbers or special characters
4. Tokenization
5. Removing stop words
6. Stemming or lemmatization
7. Converting the cleaned text into numerical features

Example:

```python
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

## 🔢 Feature Extraction

Machine learning algorithms cannot directly work with raw text. Therefore, SMS messages must be converted into numerical representations.

This project uses **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF assigns higher importance to words that are frequent in a particular message but relatively uncommon across the entire dataset.

Example:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(df["message"])
```

## 🤖 Machine Learning Models

Several classification algorithms can be evaluated for SMS spam detection.

### Naive Bayes

Multinomial Naive Bayes is particularly suitable for text classification because it works well with sparse word-frequency or TF-IDF representations.

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### Support Vector Machine

```python
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train, y_train)
```

The final model should be selected based on validation/test performance rather than assuming one algorithm will always perform best.

## 🧪 Train/Test Split

The dataset is divided into training and testing subsets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
```

Using `stratify` helps preserve the proportion of spam and ham messages in both subsets.

## 📈 Model Evaluation

Because spam detection is a classification problem, several metrics should be considered.

### Accuracy

Measures the proportion of correctly classified messages.

### Precision

Measures how many messages predicted as spam were actually spam.

High precision helps reduce legitimate messages being incorrectly classified as spam.

### Recall

Measures how many actual spam messages were successfully detected.

High recall helps reduce spam messages being missed.

### F1-Score

The F1-score is the harmonic mean of precision and recall.

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, pos_label="spam"))
print("Recall:", recall_score(y_test, predictions, pos_label="spam"))
print("F1 Score:", f1_score(y_test, predictions, pos_label="spam"))

print(classification_report(y_test, predictions))
```

## 📋 Confusion Matrix

A confusion matrix provides a detailed view of classification results.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()
```

The four possible outcomes are:

| Actual | Predicted | Result         |
| ------ | --------- | -------------- |
| Ham    | Ham       | True Negative  |
| Ham    | Spam      | False Positive |
| Spam   | Ham       | False Negative |
| Spam   | Spam      | True Positive  |

False positives are particularly important because a legitimate message could incorrectly be sent to a spam folder.

## 🔮 Predicting New Messages

After training the model, it can be used to classify new SMS messages.

```python
message = ["Congratulations! You have won a free vacation."]

message_vector = vectorizer.transform(message)
prediction = model.predict(message_vector)

print(prediction[0])
```

Example output:

```text
spam
```

Another example:

```python
message = ["Hey, I'll call you when I reach home."]

message_vector = vectorizer.transform(message)
prediction = model.predict(message_vector)

print(prediction[0])
```

Example output:

```text
ham
```

## 💾 Saving the Model

The trained model and vectorizer can be saved using `joblib`.

```python
import joblib

joblib.dump(model, "models/spam_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
```

They can later be loaded without retraining:

```python
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
```

## 🖥️ Optional Streamlit Application

The trained model can be deployed as a simple web application using Streamlit.

Install Streamlit:

```bash
pip install streamlit
```

Example `app.py`:

```python
import streamlit as st
import joblib

model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("📱 SMS Spam Detector")

message = st.text_area("Enter an SMS message:")

if st.button("Predict"):
    if message.strip():
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]

        if prediction == "spam":
            st.error("🚨 This message is Spam")
        else:
            st.success("✅ This message is Ham (Not Spam)")
    else:
        st.warning("Please enter a message.")
```

Run the application:

```bash
streamlit run app.py
```

## 🧪 Example Predictions

| SMS                                      | Prediction |
| ---------------------------------------- | ---------- |
| `Hey, are you coming to class today?`    | Ham        |
| `Congratulations! You won a free prize!` | Spam       |
| `Can you call me when you get home?`     | Ham        |
| `URGENT! Claim your cash reward now!`    | Spam       |

## 📦 Requirements

A basic `requirements.txt` can contain:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
joblib
jupyter
streamlit
```

## 🔐 Limitations

Although machine learning can identify many spam messages accurately, the system has some limitations:

* New spam patterns may not appear in the training data.
* Short messages can be difficult to classify.
* Spam and legitimate messages may use similar vocabulary.
* Model performance depends heavily on dataset quality.
* Class imbalance can affect evaluation if it is not handled properly.
* TF-IDF does not fully understand the semantic meaning of a message.
* Obfuscated or deliberately misspelled spam may be harder to detect.

## 🚀 Possible Improvements

Future versions of the project could include:

* Hyperparameter tuning
* Cross-validation
* Word and character n-grams
* Handling class imbalance
* More advanced text normalization
* Word embeddings
* Transformer-based models such as BERT
* Real-time SMS classification
* Multilingual spam detection
* REST API deployment
* Docker containerization
* Cloud deployment
* Continuous retraining with newly labeled messages

## 📊 Recommended Experimentation

For a stronger machine learning project, compare multiple approaches:

```text
Raw Text
   │
   ▼
Text Preprocessing
   │
   ▼
TF-IDF / N-grams
   │
   ├── Multinomial Naive Bayes
   ├── Logistic Regression
   └── Linear SVM
   │
   ▼
Evaluation
   │
   ├── Accuracy
   ├── Precision
   ├── Recall
   └── F1-score
   │
   ▼
Best Model
   │
   ▼
Deployment
```

When comparing models, pay particular attention to **precision, recall, and F1-score**, rather than relying exclusively on accuracy.

## 🏆 Expected Outcome

The completed project should provide a trained classification pipeline capable of taking an SMS message as input and returning a classification such as:

```text
Input:
"Congratulations! You have won a free ticket. Claim now!"

Output:
Spam
```

or:

```text
Input:
"Hey, I'll see you at 7 PM."

Output:
Ham
```

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch:

```bash
git checkout -b feature/improvement
```

3. Make your changes.
4. Commit your changes:

```bash
git commit -m "Add model improvement"
```

5. Push the branch:

```bash
git push origin feature/improvement
```

6. Open a Pull Request.

## 📄 License

This project is intended for educational and research purposes. Add the appropriate license for your repository, such as the MIT License, if you intend to distribute the project publicly.

## 👨‍💻 Author

**Your Name**

* GitHub: `https://github.com/your-ishanegi7`
* LinkedIn: `https://www.linkedin.com/in/ishanegi24k7/`

Replace the placeholder information with your actual project and profile details.

## ⭐ Acknowledgments

* SMS Spam Collection dataset
* Scikit-learn documentation and community
* NLTK documentation and community
* Python open-source community

---

**If you found this project useful, consider giving the repository a ⭐ star!**
