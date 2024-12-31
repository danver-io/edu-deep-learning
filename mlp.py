import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 데이터 생성
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# 데이터 시각화
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='k')
plt.title("Make Moons Example")
plt.show()

# 1. 데이터 생성 (비선형적으로 구분되는 데이터셋: XOR 문제와 비슷)
# X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
# print(X)
# print(y)
#
# # 2. 다층 퍼셉트론 모델 정의
# mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=1000, random_state=42)
#
# # 3. 모델 학습
# mlp.fit(X, y)
#
# # 4. 시각화를 위한 격자(grid) 데이터 생성
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
#                      np.linspace(y_min, y_max, 100))
# grid = np.c_[xx.ravel(), yy.ravel()]
#
# # 5. 격자 데이터에 대한 MLP 예측값 계산
# Z = mlp.predict(grid)
# Z = Z.reshape(xx.shape)
#
# # 6. 결과 시각화
# plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')  # 결정 경계
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')  # 원래 데이터
# plt.title("MLP Decision Boundary")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()