import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pennylane as qml
from pennylane.qnn import TorchLayer
import torch
from torch import nn

# --- Constants ---
DATASET_PATH = 'patient_vital_signs_dataset.csv'
DB_PATH = 'predictions.db'
TABLE_NAME = 'vital_predictions'

# --- DB Functions ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            heart_rate REAL,
            temperature REAL,
            spo2 REAL,
            respiratory_rate REAL,
            bp_systolic REAL,
            bp_diastolic REAL,
            predicted_status TEXT,
            model_used TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(data, predicted_status, model_used):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
        INSERT INTO {TABLE_NAME} (
            heart_rate, temperature, spo2, respiratory_rate,
            bp_systolic, bp_diastolic, predicted_status, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['heart_rate'][0], data['temperature'][0], data['spo2'][0],
        data['respiratory_rate'][0], data['bp_systolic'][0], data['bp_diastolic'][0],
        predicted_status, model_used
    ))
    conn.commit()
    conn.close()

def load_predictions():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=[
            'timestamp', 'heart_rate', 'temperature', 'spo2', 'respiratory_rate',
            'bp_systolic', 'bp_diastolic', 'predicted_status', 'model_used'
        ])
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY timestamp DESC", conn)
        return df
    except Exception as e:
        st.error(f"Error loading predictions from database: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# --- Data and Model Training ---
@st.cache_data
def load_data(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        st.error(f"Dataset not found at {filepath}")
        return None

@st.cache_resource
def train_all_models(df):
    features = ['heart_rate', 'temperature', 'spo2', 'respiratory_rate', 'bp_systolic', 'bp_diastolic']
    target = 'alert_status'
    X = df[features]
    y = LabelEncoder().fit_transform(df[target])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train ML model (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # Train DL model (Keras)
    dl_model = Sequential([
        Dense(32, input_shape=(6,), activation='relu'),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dl_model.fit(X_scaled, y, epochs=50, verbose=0)

    # Train QML model
    dev = qml.device('default.qubit', wires=6)

    def circuit(inputs, weights):
        for i in range(6):
            qml.RY(inputs[i], wires=i)
        qml.templates.BasicEntanglerLayers(weights, wires=range(6))
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    weight_shapes = {"weights": (3, 6)}
    qnode = qml.QNode(circuit, dev, interface='torch')
    qlayer = TorchLayer(qnode, weight_shapes, output_dim=3)

    class QuantumNN(nn.Module):
        def _init_(self):  # Corrected
            super()._init_()  # Corrected
            self.layer = qlayer

        def forward(self, x):
            return self.layer(x)

    qml_model = QuantumNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(qml_model.parameters(), lr=0.01)

    X_torch = torch.tensor(X_scaled, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.long)

    for epoch in range(20):  # Reduce for performance
        optimizer.zero_grad()
        output = qml_model(X_torch)
        loss = loss_fn(output, y_torch)
        loss.backward()
        optimizer.step()

    return rf_model, dl_model, qml_model, scaler, LabelEncoder().fit(df[target])

# --- Streamlit App ---
st.set_page_config(page_title="Intelligent Vital Monitoring", layout="wide")
st.title("🩺 Intelligent Patient Vital Monitoring with ML, DL & QML")
st.markdown("This system uses Random Forest (ML), Keras Neural Network (DL), and PennyLane Quantum Classifier (QML) to predict patient alert status.")

init_db()
data = load_data(DATASET_PATH)

if data is not None:
    st.subheader("Raw Dataset")
    st.dataframe(data.head())

    rf_model, dl_model, qml_model, scaler, encoder = train_all_models(data.copy())
    st.success("All models trained successfully!")

    st.sidebar.header("Enter Patient Vitals")
    hr = st.sidebar.number_input("Heart Rate (bpm)", 30, 250, 90)
    temp = st.sidebar.number_input("Temperature (°C)", 34.0, 43.0, 37.5)
    sp = st.sidebar.number_input("SpO2 (%)", 70, 100, 95)
    rr = st.sidebar.number_input("Respiratory Rate", 5, 50, 18)
    bps = st.sidebar.number_input("Systolic BP", 70, 250, 120)
    bpd = st.sidebar.number_input("Diastolic BP", 40, 150, 80)

    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest (ML)", "Neural Net (DL)", "Quantum Model (QML)"])
    predict_btn = st.sidebar.button("Predict Status")

    if predict_btn:
        input_df = pd.DataFrame({
            'heart_rate': [hr],
            'temperature': [temp],
            'spo2': [sp],
            'respiratory_rate': [rr],
            'bp_systolic': [bps],
            'bp_diastolic': [bpd]
        })

        st.subheader("Input Vitals")
        st.dataframe(input_df)

        X_input_scaled = scaler.transform(input_df)

        if model_choice == "Random Forest (ML)":
            pred = rf_model.predict(input_df)[0]
        elif model_choice == "Neural Net (DL)":
            pred_probs = dl_model.predict(X_input_scaled)
            pred = np.argmax(pred_probs[0])
        else:
            with torch.no_grad():
                q_input = torch.tensor(X_input_scaled, dtype=torch.float32)
                pred_probs = qml_model(q_input)
                pred = torch.argmax(pred_probs).item()

        label = encoder.inverse_transform([pred])[0]

        if label == 'Critical':
            st.error(f"Predicted Status: {label}")
        elif label == 'Warning':
            st.warning(f"Predicted Status: {label}")
        else:
            st.success(f"Predicted Status: {label}")

        save_prediction(input_df, label, model_choice)
        st.info("Prediction saved to database.")

    st.subheader("Prediction History")
    if st.button("Refresh History"):
        st.rerun()

    pred_df = load_predictions()
    if not pred_df.empty:
        st.dataframe(pred_df)
    else:
        st.info("No saved predictions yet.")
else:
    st.error(f"Failed to load dataset from: {DATASET_PATH}")