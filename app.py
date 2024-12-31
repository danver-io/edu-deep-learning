import streamlit as st
import joblib
import pandas as pd
import altair as alt

# 클래스 이름 정의
class_names = {
    0: "계좌번호",
    1: "임신의료비신청서",
    2: "자동차 계기판",
    3: "입퇴원 확인서",
    4: "진단서",
    5: "운전면허증",
    6: "의료비 영수증",
    7: "외래 진료 확인서",
    8: "주민등록증",
    9: "여권",
    10: "결제 확인서",
    11: "약국 영수증",
    12: "처방전",
    13: "이력서",
    14: "의견서",
    15: "자동차 등록증",
    16: "차량 번호판"
}

# 모델 및 벡터라이저 로드
vectorizer = joblib.load('data/model/cv/tfidf_vectorizer.joblib')
classifier = joblib.load('data/model/cv/tfidf_classifier.joblib')

# 페이지 제목 설정
st.title("TF-IDF 키워드 점수 확인")

# 사용자 입력 받기 (한 줄 텍스트 입력)
user_input = st.text_input("문서를 입력해 주세요:")

# 엔터 입력 시 바로 예측 실행
if user_input:
    # 사용자가 입력한 텍스트를 TF-IDF 벡터로 변환
    X = vectorizer.transform([user_input])

    # 분류 모델로 클래스별 확률(점수) 예측
    probabilities = classifier.predict_proba(X)[0]
    # 분류 모델의 클래스 이름(레이블)
    class_ids = classifier.classes_

    # 데이터프레임으로 클래스, 점수, 이름 정리
    result_df = pd.DataFrame({
        'Class ID': class_ids,
        'Class Name': [f"{class_names[class_id]} ({class_id})" for class_id in class_ids],  # 이름과 숫자 결합
        'Score': probabilities
    }).sort_values(by='Class ID', ascending=True)  # 점수 기준 정렬

    st.subheader("클래스별 예측 점수")

    # 가장 높은 점수와 해당 클래스 가져오기
    top_class_row = result_df.sort_values(by='Score', ascending=False).iloc[0]  # 점수 기준 내림차순 정렬 후 첫 번째 행 선택
    top_class_id = top_class_row['Class ID']
    top_class_name = top_class_row['Class Name']
    top_score = top_class_row['Score']

    st.write(f"클래스: {top_class_name}, 점수: {top_score:.4f}")

    # Altair 차트 생성
    chart = alt.Chart(result_df).mark_bar().encode(
        x=alt.X('Class Name', sort=None, title='Class Name (ID)'),
        y=alt.Y('Score', title='Score', scale=alt.Scale(domain=[0, max(result_df['Score']) * 1.2])),  # y축 스케일 조정
        tooltip=['Class Name', 'Score']
    ).properties(
        width=700,
        height=400
    )

    # 막대 위에 가중치 점수 라벨 추가
    text = chart.mark_text(
        align='center',
        baseline='middle',
        dy=-10,  # 막대 위에 표시
        color='black'
    ).encode(
        text=alt.Text('Score:Q', format=".2f")  # 점수를 소수점 2자리로 표시
    )

    # 차트와 텍스트 라벨 결합
    st.altair_chart(chart + text)