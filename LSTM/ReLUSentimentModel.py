import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import layers, models, callbacks
import argparse
import os
import joblib
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# ----------------------------------------------------------------
# 1. ENHANCED TEXT PREPROCESSING
# ----------------------------------------------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to string if not already
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def train_model(df, text_column='headline', label_column='sentiment'):
    """
    Train the sentiment model on the provided DataFrame
    """
    print("Training data shape:", df.shape)
    print("\nSample of training data:")
    print(df.head())
    
    # Clean sentiment labels by stripping whitespace
    df[label_column] = df[label_column].str.strip()
    
    # Preprocess all texts
    print("\nPreprocessing texts...")
    df['processed_text'] = df[text_column].apply(preprocess_text)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_column])
    X_text = df['processed_text'].values

    print("\nUnique labels:", label_encoder.classes_)
    print("Number of samples per class:", np.bincount(y))

    # -------------------------------------------------
    # K-FOLD CROSS-VALIDATION SETUP
    # -------------------------------------------------
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_accuracies = []
    all_reports = []
    best_model = None
    best_accuracy = 0
    best_vectorizer = None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_text, y), 1):
        print(f"\n=== Fold {fold_idx} / {n_splits} ===")

        # Split data
        X_train_text, X_test_text = X_text[train_idx], X_text[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # -------------------------------------------------
        # ADVANCED FEATURE ENGINEERING
        # -------------------------------------------------
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2,
            max_df=0.95
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train_text)
        X_test_tfidf = vectorizer.transform(X_test_text)

        # Convert sparse to dense for Keras
        X_train_dense = X_train_tfidf.toarray()
        X_test_dense = X_test_tfidf.toarray()

        # -------------------------------------------------
        # IMPROVED MODEL ARCHITECTURE
        # -------------------------------------------------
        input_dim = X_train_dense.shape[1]
        num_classes = len(label_encoder.classes_)
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        # -------------------------------------------------
        # OPTIMIZED TRAINING CONFIGURATION
        # -------------------------------------------------
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )

        # -------------------------------------------------
        # TRAIN WITH VALIDATION SPLIT
        # -------------------------------------------------
        history = model.fit(
            X_train_dense, y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # -------------------------------------------------
        # EVALUATE ON THIS FOLD
        # -------------------------------------------------
        y_pred_probs = model.predict(X_test_dense)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Compute accuracy
        fold_accuracy = accuracy_score(y_test, y_pred)
        all_accuracies.append(fold_accuracy)

        # Save best model
        if fold_accuracy > best_accuracy:
            best_accuracy = fold_accuracy
            best_model = model
            best_vectorizer = vectorizer

        # Classification report
        report_str = classification_report(
            y_test, 
            y_pred, 
            target_names=list(label_encoder.classes_),
            zero_division=0
        )

        print(f"Fold {fold_idx} Accuracy: {fold_accuracy:.2f}")
        print(report_str)
        all_reports.append(report_str)

    # -------------------------------------------------
    # SUMMARY OF RESULTS
    # -------------------------------------------------
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    print("\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Mean Accuracy over {n_splits} folds: {mean_accuracy:.2f} (±{std_accuracy:.2f})")

    return best_model, best_vectorizer, label_encoder

def predict_sentiment(model, vectorizer, label_encoder, text):
    """
    Predict sentiment for a single text
    """
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Vectorize
    X = vectorizer.transform([processed_text])
    X_dense = X.toarray()
    
    # Predict
    y_pred_probs = model.predict(X_dense)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    return label_encoder.classes_[y_pred[0]]

def predict_sentiments_batch(model, vectorizer, label_encoder, texts, batch_size=128):
    """
    Predict sentiments for a batch of texts
    """
    # Preprocess all texts at once
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Vectorize all texts at once
    X = vectorizer.transform(processed_texts)
    X_dense = X.toarray()
    
    # Predict in batches
    all_predictions = []
    for i in range(0, len(X_dense), batch_size):
        batch = X_dense[i:i + batch_size]
        y_pred_probs = model.predict(batch, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        all_predictions.extend(y_pred)
    
    return [label_encoder.classes_[pred] for pred in all_predictions]

def main():
    parser = argparse.ArgumentParser(description='Train and use a sentiment analysis model')
    parser.add_argument('--train', type=str, help='Path to training CSV file')
    parser.add_argument('--predict', type=str, help='Path to CSV file for prediction')
    parser.add_argument('--text-column', type=str, default='headline', help='Name of the text column in CSV')
    parser.add_argument('--label-column', type=str, default='sentiment', help='Name of the label column in CSV')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for predictions')
    
    args = parser.parse_args()

    if args.train:
        # Load and train on the training data
        print(f"Loading training data from {args.train}...")
        df = pd.read_csv(args.train)
        print("\nTraining data loaded successfully!")
        model, vectorizer, label_encoder = train_model(df, args.text_column, args.label_column)
        
        # Save the model and vectorizer
        print("\nSaving model files...")
        model.save('sentiment_model.h5')
        joblib.dump(vectorizer, 'vectorizer.joblib')
        joblib.dump(label_encoder, 'label_encoder.joblib')
        print("Model saved to sentiment_model.h5")
        print("Vectorizer saved to vectorizer.joblib")
        print("Label encoder saved to label_encoder.joblib")

    if args.predict:
        # Load the model and vectorizer
        print("\nLoading model files...")
        try:
            model = tf.keras.models.load_model('sentiment_model.h5')
            vectorizer = joblib.load('vectorizer.joblib')
            label_encoder = joblib.load('label_encoder.joblib')
            print("Model files loaded successfully!")
        except Exception as e:
            print(f"Error loading model files: {str(e)}")
            print("Please train the model first using --train")
            return

        # Load and predict on the new data
        print(f"\nLoading data for prediction from {args.predict}...")
        df = pd.read_csv(args.predict)
        print(f"Data loaded successfully! Processing {len(df)} headlines...")
        print("\nSample of data to predict:")
        print(df.head())
        
        # Predict sentiments in batches
        print("\nPredicting sentiments...")
        start_time = time.time()
        df['predicted_sentiment'] = predict_sentiments_batch(
            model, 
            vectorizer, 
            label_encoder, 
            df[args.text_column].values,
            batch_size=args.batch_size
        )
        end_time = time.time()
        print(f"\nPrediction completed in {end_time - start_time:.2f} seconds")
        print(f"Average time per headline: {(end_time - start_time) * 1000 / len(df):.2f} ms")
        
        # Save predictions
        if args.output:
            output_path = args.output
        else:
            output_path = 'predictions.csv'
        
        df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        print("\nSample of predictions:")
        print(df[['Headline', 'predicted_sentiment']].head())
        
        # Print sentiment distribution
        sentiment_counts = df['predicted_sentiment'].value_counts()
        print("\nSentiment distribution:")
        print(sentiment_counts)
        print("\nPercentages:")
        print(sentiment_counts / len(df) * 100)

if __name__ == "__main__":
    main()
