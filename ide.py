import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("train_features.csv")

print("Shape of data:", train.shape)

pd.set_option('display.max_columns', None)
print(train.head())
numeric_data = train.select_dtypes(include=['number'])

print("\nNumeric data shape:", numeric_data.shape)


plt.figure(figsize=(10,5))

plt.plot(numeric_data.iloc[:200, 0], label='Feature 1')
plt.plot(numeric_data.iloc[:200, 1], label='Feature 2')
plt.plot(numeric_data.iloc[:200, 2], label='Feature 3')

plt.title("Sensor Signals (Sample)")
plt.legend()
plt.show()

corr = numeric_data.iloc[:, :20].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(numeric_data.iloc[:, 0], kde=True)
plt.title("Distribution of Feature 1")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x=numeric_data.iloc[:, 1])
plt.title("Boxplot of Feature 2")
plt.show()