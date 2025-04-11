import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

user_rfm = np.array([[187, 9, 78]])
cluster = kmeans_model.predict(user_rfm)

with open('manual_rule_based.pkl', 'rb') as f:
    manual_rule_based = pickle.load(f)

df_RFM = manual_rule_based['df_RFM']
r_bins = manual_rule_based['r_bins']
f_bins = manual_rule_based['f_bins']
m_bins = manual_rule_based['m_bins']
r_labels = manual_rule_based['r_labels']
f_labels = manual_rule_based['f_labels']
m_labels = manual_rule_based['m_labels']

# --- Dữ liệu khách hàng mới ---
new_customer = {
    'Recency': 91,
    'Frequency': 27,
    'Monetary': 361.45
}

# --- Tính rank (so sánh với dữ liệu gốc) ---
r_rank = (df_RFM['Recency'] < new_customer['Recency']).sum() + 1
f_rank = (df_RFM['Frequency'] < new_customer['Frequency']).sum() + 1
m_rank = (df_RFM['Monetary'] < new_customer['Monetary']).sum() + 1

# --- Dự đoán R, F, M dựa vào phân vị đã lưu ---
R = pd.cut([r_rank], bins=r_bins, labels=r_labels, include_lowest=True)[0]
F = pd.cut([f_rank], bins=f_bins, labels=f_labels, include_lowest=True)[0]
M = pd.cut([m_rank], bins=m_bins, labels=m_labels, include_lowest=True)[0]

print(R)
print(F)
print(M)