import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'  # NanumGothic 폰트 사용
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

st.title("퍼셉트론(Perceptron) 예제")

st.sidebar.header("학습 파라미터 설정")
learning_rate = st.sidebar.slider("학습률 (learning rate)", 0.01, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("반복 횟수 (epochs)", 10, 500, 100, 10)
num_points = st.sidebar.slider("데이터 포인트 개수", 10, 200, 50, 10)

st.write("""
퍼셉트론 알고리즘은 다음과 같습니다.

1. 가중치 초기화 (0 또는 작은 랜덤 값)
2. 모든 샘플에 대해 예측 수행
3. 예측이 실제 레이블과 다를 경우 가중치 업데이트
4. 일정 횟수 반복 또는 모든 샘플이 정분류될 때까지 수행
""")

# 데이터 생성
X = None
Y = None

st.subheader("데이터 생성")
data_option = st.radio("데이터 생성 방식", ("무작위 생성", "수동 입력"))
if data_option == "무작위 생성":
    class_ratio = st.slider("클래스 비율 (0~1)", 0.0, 1.0, 0.5, 0.05)
    if st.button("데이터 무작위 생성"):
        # 랜덤 데이터 생성
        # 두 개의 클래스 (-1, 1)를 갖는 2D 데이터 생성
        X = np.random.randn(num_points, 2)
        y = np.where(np.random.rand(num_points) > class_ratio, 1, -1)
else:
    st.write("아래 텍스트 입력창에 x1, x2, label 형태로 여러 줄 입력해주세요. 예:\n0.1 0.5 1\n-0.3 -0.8 -1")
    user_data = st.text_area("데이터 입력 (x1 x2 label)")
    if st.button("입력 데이터로 반영"):
        lines = user_data.strip().split("\n")
        data_list = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                x1, x2, label = parts
                data_list.append((float(x1), float(x2), int(label)))
        if len(data_list) > 0:
            X = np.array([[d[0], d[1]] for d in data_list])
            y = np.array([d[2] for d in data_list])
        else:
            st.error("유효한 데이터가 없습니다.")
            X, y = None, None
    else:
        X, y = None, None

if X is not None and y is not None:
    # 퍼셉트론 알고리즘 구현
    # 가중치 초기화 (bias 포함하기 위해 w0, w1, w2 형태로 사용)
    w = np.zeros(X.shape[1] + 1)  # [bias, w1, w2]

    # 예측 함수
    def predict(x):
        return np.sign(np.dot(x, w[1:]) + w[0])

    # 학습 과정 기록용
    weight_history = []
    accuracy_history = []

    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            # 예측
            y_pred = predict(X[i])
            # 업데이트 조건
            if y_pred != y[i]:
                # 가중치 업데이트
                w[1:] += learning_rate * y[i] * X[i]
                w[0] += learning_rate * y[i]
                errors += 1

        # 정확도 계산
        predictions = [predict(x) for x in X]
        accuracy = np.mean(predictions == y)
        accuracy_history.append(accuracy)
        weight_history.append(w.copy())

        # 모든 샘플이 정분류되면 중단
        if accuracy == 1.0:
            break

    st.write("최종 정확도:", accuracy_history[-1])
    st.write("최종 가중치 (bias, w1, w2):", w)

    # 결정 경계 그리기
    # w0 + w1*x1 + w2*x2 = 0 => x2 = (-w0 - w1*x1) / w2
    fig, ax = plt.subplots(figsize=(5,5))
    # 데이터 점 플롯
    for label in [-1, 1]:
        mask = (y == label)
        ax.scatter(X[mask, 0], X[mask, 1], label=f'Class {label}', alpha=0.7)

    # 결정 경계 선
    if w[2] != 0:
        x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
        x_plot = np.linspace(x_min, x_max, 100)
        y_plot = -(w[0] + w[1]*x_plot)/w[2]
        ax.plot(x_plot, y_plot, 'k--', label='Decision Boundary')
    ax.legend()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("퍼셉트론 결정 경계")
    st.pyplot(fig)

    # 정확도 변화 그래프
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, len(accuracy_history)+1), accuracy_history)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("학습 과정에서의 정확도 변화")
    st.pyplot(fig2)