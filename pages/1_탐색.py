
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------
# 페이지 설정
# ---------------------------------
st.set_page_config(
    page_title="데이터 탐색",
    page_icon="📊",
    layout="wide"
)

st.title("📊 뇌졸중 데이터 탐색")
st.caption("데이터의 분포와 뇌졸중 발생 비율을 다양한 그래프로 살펴봅니다.")

# ---------------------------------
# 데이터 불러오기
# ---------------------------------
DATA_URL = "https://raw.githubusercontent.com/k-suhyeon03/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# ---------------------------------
# 데이터 확인
# ---------------------------------
required_columns = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
    "stroke"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

# 숫자형 데이터 변환
numeric_columns = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "bmi",
    "stroke"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------
# 1. 나이와 평균 혈당 수치 분포
# ---------------------------------
st.header("1. 나이와 평균 혈당 수치 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "인원수"}
    )
    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="인원수"
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 수치 분포",
        labels={
            "avg_glucose_level": "평균 혈당 수치",
            "count": "인원수"
        }
    )
    fig_glucose.update_layout(
        xaxis_title="평균 혈당 수치",
        yaxis_title="인원수"
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

# ---------------------------------
# 2. 뇌졸중 여부에 따른 나이와 혈당 비교
# ---------------------------------
st.header("2. 뇌졸중 여부에 따른 나이와 혈당 비교")

stroke_df = df[df["stroke"].isin([0, 1])].copy()

stroke_df["뇌졸중 여부"] = stroke_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        stroke_df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={
            "age": "나이",
            "뇌졸중 여부": "구분"
        }
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    fig_glucose_box = px.box(
        stroke_df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={
            "avg_glucose_level": "평균 혈당 수치",
            "뇌졸중 여부": "구분"
        }
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 평균값 비교
mean_table = (
    stroke_df.groupby("뇌졸중 여부")
    .agg(
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean")
    )
    .reset_index()
)

mean_table = mean_table.rename(columns={
    "평균_나이": "평균 나이",
    "평균_혈당": "평균 혈당 수치"
})

mean_table[["평균 나이", "평균 혈당 수치"]] = (
    mean_table[["평균 나이", "평균 혈당 수치"]].round(2)
)

st.subheader("뇌졸중 여부에 따른 평균값")
st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------
# 3. 고혈압과 심장질환에 따른 뇌졸중 비율
# ---------------------------------
st.header("3. 고혈압과 심장질환에 따른 뇌졸중 비율")

def calculate_stroke_rate(data, column):
    result = (
        data[data[column].isin([0, 1])]
        .groupby(column)
        .agg(
            전체_인원=("stroke", "count"),
            뇌졸중_인원=("stroke", "sum")
        )
        .reset_index()
    )

    result["뇌졸중 비율(%)"] = (
        result["뇌졸중_인원"] / result["전체_인원"] * 100
    ).round(2)

    return result

# 고혈압
hypertension_rate = calculate_stroke_rate(
    stroke_df,
    "hypertension"
)

hypertension_display = hypertension_rate.rename(columns={
    "hypertension": "고혈압 여부",
    "전체_인원": "전체 인원",
    "뇌졸중_인원": "뇌졸중 인원"
})

hypertension_display["고혈압 여부"] = (
    hypertension_display["고혈압 여부"].map({
        0: "없음",
        1: "있음"
    })
)

# 심장질환
heart_rate = calculate_stroke_rate(
    stroke_df,
    "heart_disease"
)

heart_display = heart_rate.rename(columns={
    "heart_disease": "심장질환 여부",
    "전체_인원": "전체 인원",
    "뇌졸중_인원": "뇌졸중 인원"
})

heart_display["심장질환 여부"] = (
    heart_display["심장질환 여부"].map({
        0: "없음",
        1: "있음"
    })
)

col1, col2 = st.columns(2)

with col1:
    fig_hypertension = px.bar(
        hypertension_display,
        x="고혈압 여부",
        y="뇌졸중 비율(%)",
        title="고혈압 여부에 따른 뇌졸중 비율",
        text="뇌졸중 비율(%)",
        labels={
            "고혈압 여부": "고혈압",
            "뇌졸중 비율(%)": "뇌졸중 비율(%)"
        }
    )
    fig_hypertension.update_traces(texttemplate="%{text:.2f}%")
    st.plotly_chart(fig_hypertension, use_container_width=True)

with col2:
    fig_heart = px.bar(
        heart_display,
        x="심장질환 여부",
        y="뇌졸중 비율(%)",
        title="심장질환 여부에 따른 뇌졸중 비율",
        text="뇌졸중 비율(%)",
        labels={
            "심장질환 여부": "심장질환",
            "뇌졸중 비율(%)": "뇌졸중 비율(%)"
        }
    )
    fig_heart.update_traces(texttemplate="%{text:.2f}%")
    st.plotly_chart(fig_heart, use_container_width=True)

# 고혈압 표
st.subheader("고혈압 여부에 따른 상세 데이터")
st.dataframe(
    hypertension_display[
        [
            "고혈압 여부",
            "전체 인원",
            "뇌졸중 인원",
            "뇌졸중 비율(%)"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# 심장질환 표
st.subheader("심장질환 여부에 따른 상세 데이터")
st.dataframe(
    heart_display[
        [
            "심장질환 여부",
            "전체 인원",
            "뇌졸중 인원",
            "뇌졸중 비율(%)"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# ---------------------------------
# 4. BMI 결측치와 뇌졸중 비율 비교
# ---------------------------------
st.header("4. BMI 결측치와 뇌졸중 비율 비교")

missing_bmi = stroke_df[stroke_df["bmi"].isna()]
valid_bmi = stroke_df[stroke_df["bmi"].notna()]

overall_rate = stroke_df["stroke"].mean() * 100

missing_bmi_rate = (
    missing_bmi["stroke"].mean() * 100
    if len(missing_bmi) > 0 else 0
)

bmi_comparison = pd.DataFrame({
    "구분": ["전체 데이터", "BMI 결측 데이터"],
    "인원수": [len(stroke_df), len(missing_bmi)],
    "뇌졸중 인원": [
        int(stroke_df["stroke"].sum()),
        int(missing_bmi["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        round(overall_rate, 2),
        round(missing_bmi_rate, 2)
    ]
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)

fig_bmi = px.bar(
    bmi_comparison,
    x="구분",
    y="뇌졸중 비율(%)",
    text="뇌졸중 비율(%)",
    title="전체 데이터와 BMI 결측 데이터의 뇌졸중 비율 비교",
    labels={
        "구분": "데이터 구분",
        "뇌졸중 비율(%)": "뇌졸중 비율(%)"
    }
)

fig_bmi.update_traces(texttemplate="%{text:.2f}%")
st.plotly_chart(fig_bmi, use_container_width=True)

# ---------------------------------
# 5. 흡연 상태별 인원수
# ---------------------------------
st.header("5. 흡연 상태별 인원수")

smoking_table = (
    df["smoking_status"]
    .fillna("정보 없음")
    .value_counts()
    .reset_index()
)

smoking_table.columns = ["흡연 상태", "인원수"]

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)

fig_smoking = px.bar(
    smoking_table,
    x="흡연 상태",
    y="인원수",
    text="인원수",
    title="흡연 상태별 인원수",
    labels={
        "흡연 상태": "흡연 상태",
        "인원수": "인원수"
    }
)

st.plotly_chart(fig_smoking, use_container_width=True)

# ---------------------------------
# 마무리
# ---------------------------------
st.divider()
st.caption(
    "※ 이 페이지는 데이터의 통계적 특징을 탐색하기 위한 실습용입니다. "
    "그래프에서 나타나는 차이가 곧 원인과 결과를 의미하는 것은 아닙니다."
)
