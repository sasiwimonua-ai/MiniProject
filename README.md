# MiniProject
Mini Project รายวิชา Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# df = pd.read_csv('creditcard_2023.csv')
# ตัวอย่างการสร้างข้อมูลจำลองเพื่อรันโค้ดทดสอบ
np.random.seed(42)
df = pd.DataFrame(np.random.randn(500, 5), columns=['V1', 'V2', 'V3', 'V4', 'Amount'])
df['Class'] = np.random.randint(0, 2, 500)
# 2. สถิติเบื้องต้น (Descriptive Statistics)
print("--- Basic Statistics ---")
# แสดง mean, std, min, max และ quartiles
stats = df.describe()
print(stats)
# ดูการกระจายตัวของ Target Variable (Class)
print("\n--- Target Distribution ---")
print(df['Class'].value_counts(normalize=True) * 100)
# 3. การสร้างกราฟ (Visualization)
plt.figure(figsize=(16, 12))
# --- 3.1 Histogram (ดูการกระจายตัวของข้อมูล) ---
plt.subplot(2, 2, 1)
sns.histplot(df['V1'], kde=True, color='skyblue')
plt.title('Histogram of V1 (Distribution)')
# --- 3.2 Boxplot (ดู Outliers และช่วงของข้อมูล) ---
plt.subplot(2, 2, 2)
sns.boxplot(x='Class', y='Amount', data=df, palette='Set2')
plt.title('Boxplot of Amount by Class')
# --- 3.3 Correlation Heatmap (ดูความสัมพันธ์ระหว่างตัวแปร) ---
plt.subplot(2, 2, 3)
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
# --- 3.4 Violin Plot (เพิ่มเติม: ดูความหนาแน่นและช่วงข้อมูลพร้อมกัน) ---
plt.subplot(2, 2, 4)
sns.violinplot(x='Class', y='V2', data=df, inner="quartile")
plt.title('Violin Plot of V2 by Class')
plt.tight_layout()
plt.show()
