import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# 데이터 생성
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 다층 퍼셉트론 모델 정의
model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', random_state=42)


# 모델 학습
model.fit(X_train, y_train)


# 예측
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# 정확도 계산
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# 시각화
plt.figure(figsize=(10, 5))

# 훈련 데이터 시각화
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
plt.title("Train Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 테스트 데이터 시각화
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr')
plt.title("Test Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
