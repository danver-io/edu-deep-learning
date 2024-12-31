import numpy as np

# 입력 데이터: [x1, x2], x1은 (age>=18?), x2는 (is_member?)
X = np.array([
    [0, 0], # age<18, not member
    [0, 1], # age<18, member
    [1, 0], # age>=18, not member
    [1, 1]  # age>=18, member
])
y = np.array([0, 0, 0, 1]) # AND의 진리표

w = np.random.randn(2)*0.01
b = 0.0
learningRate = 0.1
epochs = 10

def step(x):
    return 1 if x >= 0 else 0

print(f"처음 시작 w : {w} b : {b}")
for epoch in range(epochs):
    for i in range(len(X)):
        y_pred = step(np.dot(X[i], w) + b)
        error = y[i] - y_pred
        if error != 0:
            w += learningRate * error * X[i]
            b += learningRate * error
        print(f"{epoch + 1} 에포크 sample[{i + 1}] data : {X[i]} error : {error} w : {w} b : {b} {w[0]}x+{w[1]}y{b}=0")
    print(f"============================================================================")

# 학습 후 확인
test_data = [
    (20, True),  # 나이 20 이상, 회원
    (17, True),  # 나이 17 미만, 회원
    (25, False), # 나이 25 이상, 비회원
    (15, False)  # 나이 15 미만, 비회원
]

for age, member in test_data:
    x1 = 1 if age >= 18 else 0
    x2 = 1 if member else 0
    prediction = step(np.dot([x1, x2], w) + b)
    print(f"Age={age}, Member={member}, Discount={prediction==1}")