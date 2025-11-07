import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

def set_custom_style():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f7f7f7;
            color: #000000;
        }
        input, textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
        }454  
        .stTextInput > div > div > input {
            color: #000000 !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

set_custom_style()



@st.cache_data
def load_data():
    raw_data = pd.read_csv("D:\\ML\\random_forest_dataset.csv")
    quiz_data = pd.read_csv("D:\\ML\\quiz_performance_dataset.csv")
    return raw_data, quiz_data

raw_data, quiz_data = load_data()

st.title("EDU-GAMIX ASSISTIVE LEARNING")
st.header(" Predict Quiz Difficulty Level")

X = raw_data.drop(columns='Difficulty Prediction')
Y = raw_data['Difficulty Prediction']

std = StandardScaler()
X_scaled = pd.DataFrame(std.fit_transform(X), columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=45)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

user_input = st.text_input(" Enter quiz parameters (comma-separated values):")

difficulty_result = None
if st.button(" Predict Difficulty Level"):
    try:
        input_values = np.array([float(i.strip()) for i in user_input.split(',')]).reshape(1, -1)
        input_df = pd.DataFrame(input_values, columns=X.columns)
        std_input_values = std.transform(input_df)
        prediction = rf_model.predict(std_input_values)

        difficulty_result = "✅ Move to next difficulty level" if prediction[0] == 1 else "⚠️ Stay at the same level"
        st.markdown(
            f"""
            <div style="background-color: #d0f0c0; padding: 16px; border-radius: 8px; border-left: 5px solid green;">
                <span style="color: #1a3c1a; font-weight: bold; font-size: 18px;">Prediction: {difficulty_result}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error("❌ Invalid input format. Please enter numeric values.")


st.header("🔍 Weak Topic Detection")

X1 = quiz_data[['Quiz_ID', 'Total_Marks', 'Avg_Response_Time', 'Total_Time_Spent'] +
               [f'Topic_{i}_Marks' for i in range(1, 11)]]

Y1 = quiz_data[[f'Topic_{i}_Strength' for i in range(1, 11)]]

std1 = StandardScaler()
X1_scaled = pd.DataFrame(std1.fit_transform(X1), columns=X1.columns)

x1_train, x1_test, y1_train, y1_test = train_test_split(X1_scaled, Y1, test_size=0.2, random_state=3)
y1_train = y1_train.astype(int)
y1_test = y1_test.astype(int)

dt_model = DecisionTreeClassifier()
dt_model.fit(x1_train, y1_train)

user_input_1 = st.text_input(" Enter quiz performance parameters (comma-separated):")

if st.button("🔍 Predict Weak Topics"):
    try:
        input_values_1 = np.array([float(i.strip()) for i in user_input_1.split(',')]).reshape(1, -1)
        input_df_1 = pd.DataFrame(input_values_1, columns=X1.columns)
        std1_input_values = std1.transform(input_df_1)
        prediction1 = dt_model.predict(std1_input_values)

        topic_names = ['Digestive System', 'Respiratory System', 'Circulatory System', 'Excretory System',
                       'Nervous System', 'Human Reproduction', 'Endocrine System', 'Genetics and Heredity',
                       'Immune System and Diseases', 'Biotechnology in Human Welfare']

        st.subheader(" Predicted Topic Strengths")
        for i, topic in enumerate(topic_names):
            strength = prediction1[0][i]
            level = "✅ Strong" if strength == 1 else "❌ Weak"
            st.markdown(
                f"<span style='color:#000000;font-size:16px;'>{topic}: <strong>{level}</strong></span>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error("❌ Invalid input format. Please enter numeric values.")
