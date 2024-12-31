import streamlit as st
import joblib

# 모델 및 벡터라이저 로드
vectorizer = joblib.load('data/model/cv/tfidf_vectorizer.joblib')
classifier = joblib.load('data/model/cv/tfidf_classifier.joblib')

# 페이지 제목 설정
st.title("TF-IDF 키워드 점수 확인")

# 사용자 입력 받기
user_input = st.text_area("문서를 입력해 주세요:", height=200)

# 버튼 클릭 시 예측 수행
if st.button("예측하기"):
    if user_input.strip() == "":
        st.warning("텍스트를 입력해 주세요.")
    else:
        # 사용자가 입력한 텍스트를 TF-IDF 벡터로 변환
        X = vectorizer.transform([user_input])

        # 분류 모델로 클래스별 확률(점수) 예측
        probabilities = classifier.predict_proba(X)[0]
        # 분류 모델의 클래스 이름(레이블)
        classes = classifier.classes_

        st.subheader("클래스별 예측 점수")
        # 각 클래스와 해당 확률을 함께 표시
        for cls, prob in zip(classes, probabilities):
            st.write(f"{cls}: {prob:.4f}")