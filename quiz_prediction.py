import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack

file_path = "D:/ML/quiz_data.csv"

TOPIC_COLUMNS = [
    "Digestive System", "Respiratory System", "Circulatory System", "Excretory System",
    "Nervous System", "Human Reproduction", "Endocrine System", "Genetics and Heredity",
    "Immune System and Diseases", "Biotechnology in Human Welfare"
]

MODEL_FILES = {
    "difficulty": "model_difficulty.pkl",
    "topic": "model_topic.pkl",
    "scaler": "scaler.pkl",
    "encoder": "encoder.pkl",
    "labels_difficulty": "labels_difficulty.pkl",
    "labels_topic": "labels_topic.pkl"
}

def train_models():
    st.info("Training models... Please wait.")
    quiz_data = pd.read_csv(file_path)
    quiz_data.fillna(0, inplace=True)
    topic_df = pd.get_dummies(quiz_data["Topic"], prefix="", prefix_sep="")
    topic_df = topic_df.reindex(columns=TOPIC_COLUMNS, fill_value=0)
    quiz_data = pd.concat([quiz_data, topic_df], axis=1)

    num_features = ['QuizID', 'QuestionID', 'PreviousCorrect']
    cat_features = ['PreviousDifficulty']
    
    X = quiz_data[num_features + cat_features]
    Y_difficulty = quiz_data['Difficulty']
    Y_weak_topic = quiz_data[TOPIC_COLUMNS].idxmax(axis=1)
    
    x_train, x_test, y_train_difficulty, y_test_difficulty, y_train_topic, y_test_topic = train_test_split(X, Y_difficulty, Y_weak_topic, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    x_train_numeric = scaler.fit_transform(x_train[num_features])
    x_test_numeric = scaler.transform(x_test[num_features])
    
    encoder = OneHotEncoder(handle_unknown='ignore')
    x_train_cat = encoder.fit_transform(x_train[cat_features])
    x_test_cat = encoder.transform(x_test[cat_features])
    
    x_train_features = hstack([x_train_numeric, x_train_cat]).toarray()
    x_test_features = hstack([x_test_numeric, x_test_cat]).toarray()
    
    y_train_difficulty_encoded, label_classes_difficulty = pd.factorize(y_train_difficulty)
    y_train_topic_encoded, label_classes_topic = pd.factorize(y_train_topic)
    
    model_difficulty = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model_difficulty.fit(x_train_features, y_train_difficulty_encoded)
    
    model_topic = DecisionTreeClassifier(max_depth=20, random_state=42)
    model_topic.fit(x_train_features, y_train_topic_encoded)
    
    joblib.dump(model_difficulty, MODEL_FILES["difficulty"])
    joblib.dump(model_topic, MODEL_FILES["topic"])
    joblib.dump(scaler, MODEL_FILES["scaler"])
    joblib.dump(encoder, MODEL_FILES["encoder"])
    joblib.dump(label_classes_difficulty, MODEL_FILES["labels_difficulty"])
    joblib.dump(label_classes_topic, MODEL_FILES["labels_topic"])
    
    st.success("✅ Models trained and saved successfully!")

if not all(os.path.exists(f) for f in MODEL_FILES.values()):
    train_models()

@st.cache_resource
def load_models():
    return {
        "model_difficulty": joblib.load(MODEL_FILES["difficulty"]),
        "model_topic": joblib.load(MODEL_FILES["topic"]),
        "scaler": joblib.load(MODEL_FILES["scaler"]),
        "encoder": joblib.load(MODEL_FILES["encoder"]),
        "labels_difficulty": joblib.load(MODEL_FILES["labels_difficulty"]),
        "labels_topic": joblib.load(MODEL_FILES["labels_topic"])
    }

resources = load_models()

st.title("⚡ THUNDERS QUIZ PREDICTION")
st.write("Predict quiz difficulty & detect weak topics!")

quiz_id = st.number_input("Quiz ID", min_value=1, step=1)
question_id = st.number_input("Question ID", min_value=1, step=1)
previous_difficulty = st.selectbox("Previous Difficulty", ["Easy", "Medium", "Hard"])
previous_correct = st.radio("Was the previous answer correct?", [0, 1])

st.subheader("Select the Topic for this Question:")
selected_topic = st.selectbox("Topic", TOPIC_COLUMNS)
topic_values = {topic: 1 if topic == selected_topic else 0 for topic in TOPIC_COLUMNS}

if st.button("Predict Performance"):
    try:
        input_numeric_raw = [[quiz_id, question_id, previous_correct] + list(topic_values.values())]
        input_numeric = resources["scaler"].transform(input_numeric_raw)
        input_cat = resources["encoder"].transform([[previous_difficulty]])
        input_features = hstack([input_numeric, input_cat]).toarray()
        
        pred_difficulty = resources["model_difficulty"].predict(input_features)
        pred_topic = resources["model_topic"].predict(input_features)
        
        predicted_difficulty = resources["labels_difficulty"][pred_difficulty[0]]
        predicted_weak_topic = resources["labels_topic"][pred_topic[0]]
        
        st.success(f"🔮 Predicted Difficulty: *{predicted_difficulty}*")
        st.warning(f"⚠ Weak Topic Identified: *{predicted_weak_topic}*")
    except Exception as e:
        st.error(f"Error: {e}")