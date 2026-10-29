
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "k-suhyeon03/data/refs/heads/main/stroke.csv"
)


# -----------------------------
# 제목
# -----------------------------
st.title("🔎 데이터 탐색")
st.caption(
    "나이, 평균 혈당, 건강 상태, 흡연 상태와 "
    "뇌졸중 발생의 관계를 데이터로 살펴봅니다."
)


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data(url):
    return pd.read_csv(url, encoding="utf-8")


try:
    df = load_data(DATA_URL)

except Exception as error:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "인터넷 연결과 CSV 주소를 확인해 주세요."
    )
    st.code(str(error))
    st.stop()


required_columns = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
    "stroke",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# 분석에 사용할 열을 숫자형으로 변환
for column in [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "stroke",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# 뇌졸중 발생 여부가 확인된 행만 그룹 비교에 사용
stroke_df = df[df["stroke"].isin([0, 1])].copy()

stroke_df["뇌졸중 여부"] = stroke_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음",
})


# -----------------------------
# 1. 나이와 평균 혈당 분포
# -----------------------------
st.header("1. 나이와 평균 혈당의 분포")

st.write(
    "히스토그램은 값이 특정 구간에 얼마나 많이 "
    "분포하는지 보여 줍니다."
)

col1, col2 = st.columns(2)

with col1:
    age_data = df[["age"]].dropna()

    fig_age = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수",
        },
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.05,
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True,
    )

with col2:
    glucose_data = df[["avg_glucose_level"]].dropna()

    fig_glucose = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수",
        },
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.05,
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True,
    )


# -----------------------------
# 2. 뇌졸중 여부에 따른 상자그림
# -----------------------------
st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

st.write(
    "상자그림은 중앙값, 사분위수, 자료의 퍼짐 정도를 "
    "비교하는 그래프입니다. 두 그룹의 평균값도 표로 "
    "확인할 수 있습니다."
)

box_col1, box_col2 = st.columns(2)

with box_col1:
    age_box_data = stroke_df[
        ["뇌졸중 여부", "age"]
    ].dropna()

    fig_age_box = px.box(
        age_box_data,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders={
            "뇌졸중 여부": [
                "뇌졸중 없음",
                "뇌졸중 있음",
            ]
        },
        title="뇌졸중 여부에 따른 나이",
        labels={
            "age": "나이",
            "뇌졸중 여부": "그룹",
        },
        points="outliers",
    )

    fig_age_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이",
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True,
    )

with box_col2:
    glucose_box_data = stroke_df[
        ["뇌졸중 여부", "avg_glucose_level"]
    ].dropna()

    fig_glucose_box = px.box(
        glucose_box_data,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders={
            "뇌졸중 여부": [
                "뇌졸중 없음",
                "뇌졸중 있음",
            ]
        },
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "그룹",
        },
        points="outliers",
    )

    fig_glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당",
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True,
    )


# 두 그룹의 평균값 표
age_means = stroke_df.groupby(
    "뇌졸중 여부"
)["age"].mean()

glucose_means = stroke_df.groupby(
    "뇌졸중 여부"
)["avg_glucose_level"].mean()

mean_table = pd.DataFrame({
    "뇌졸중 여부": [
        "뇌졸중 없음",
        "뇌졸중 있음",
    ],
})

mean_table["평균 나이"] = mean_table[
    "뇌졸중 여부"
].map(age_means)

mean_table["평균 혈당"] = mean_table[
    "뇌졸중 여부"
].map(glucose_means)

mean_table["평균 나이"] = mean_table[
    "평균 나이"
].round(2)

mean_table["평균 혈당"] = mean_table[
    "평균 혈당"
].round(2)

st.subheader("두 그룹의 평균값")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# 3. 고혈압 여부에 따른 뇌졸중 비율
# -----------------------------
st.header("3. 고혈압 여부에 따른 뇌졸중 비율")

st.write(
    "각 그룹에서 뇌졸중 발생 여부가 확인된 사람 중 "
    "stroke가 1인 사람의 비율을 계산합니다."
)

hypertension_df = stroke_df[
    stroke_df["hypertension"].isin([0, 1])
].copy()

hypertension_df["고혈압 여부"] = hypertension_df[
    "hypertension"
].map({
    0: "고혈압 없음",
    1: "고혈압 있음",
})

hypertension_rate = (
    hypertension_df
    .groupby("고혈압 여부")
    .agg(
        전체 인원=("stroke", "size"),
        뇌졸중 인원=("stroke", "sum"),
        뇌졸중 비율=("stroke", "mean"),
    )
    .reset_index()
)

hypertension_rate["뇌졸중 비율(%)"] = (
    hypertension_rate["뇌졸중 비율"] * 100
)

hypertension_rate["뇌졸중 비율(%)"] = (
    hypertension_rate["뇌졸중 비율(%)"].round(2)
)

fig_hypertension = px.bar(
    hypertension_rate,
    x="고혈압 여부",
    y="뇌졸중 비율(%)",
    text="뇌졸중 비율(%)",
    title="고혈압 여부에 따른 뇌졸중 비율",
    labels={
        "고혈압 여부": "고혈압 여부",
        "뇌졸중 비율(%)": "뇌졸중 비율(%)",
    },
)

fig_hypertension.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
)

fig_hypertension.update_layout(
    yaxis_title="뇌졸중 비율(%)",
    xaxis_title="고혈압 여부",
    yaxis_rangemode="tozero",
)

st.plotly_chart(
    fig_hypertension,
    use_container_width=True,
)

st.dataframe(
    hypertension_rate[
        [
            "고혈압 여부",
            "전체 인원",
            "뇌졸중 인원",
            "뇌졸중 비율(%)",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# 4. 심장병 여부에 따른 뇌졸중 비율
# -----------------------------
st.header("4. 심장병 여부에 따른 뇌졸중 비율")

heart_df = stroke_df[
    stroke_df["heart_disease"].isin([0, 1])
].copy()

heart_df["심장병 여부"] = heart_df[
    "heart_disease"
].map({
    0: "심장병 없음",
    1: "심장병 있음",
})

heart_rate = (
    heart_df
    .groupby("심장병 여부")
    .agg(
        전체 인원=("stroke", "size"),
        뇌졸중 인원=("stroke", "sum"),
        뇌졸중 비율=("stroke", "mean"),
    )
    .reset_index()
)

heart_rate["뇌졸중 비율(%)"] = (
    heart_rate["뇌졸중 비율"] * 100
)

heart_rate["뇌졸중 비율(%)"] = (
    heart_rate["뇌졸중 비율(%)"].round(2)
)

fig_heart = px.bar(
    heart_rate,
    x="심장병 여부",
    y="뇌졸중 비율(%)",
    text="뇌졸중 비율(%)",
    title="심장병 여부에 따른 뇌졸중 비율",
    labels={
        "심장병 여부": "심장병 여부",
        "뇌졸중 비율(%)": "뇌졸중 비율(%)",
    },
)

fig_heart.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
)

fig_heart.update_layout(
    yaxis_title="뇌졸중 비율(%)",
    xaxis_title="심장병 여부",
    yaxis_rangemode="tozero",
)

st.plotly_chart(
    fig_heart,
    use_container_width=True,
)

st.dataframe(
    heart_rate[
        [
            "심장병 여부",
            "전체 인원",
            "뇌졸중 인원",
            "뇌졸중 비율(%)",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# 5. BMI 빈 값과 뇌졸중 비율
# -----------------------------
st.header("5. BMI가 비어 있는 사람들의 뇌졸중 비율")

st.write(
    "BMI가 비어 있는 사람들 중 뇌졸중 발생 여부가 "
    "확인된 사람을 대상으로 비율을 계산하고, "
    "전체 데이터의 뇌졸중 비율과 비교합니다."
)

missing_bmi_df = df[df["bmi"].isna()].copy()

missing_bmi_known = missing_bmi_df[
    missing_bmi_df["stroke"].isin([0, 1])
]

all_known = df[df["stroke"].isin([0, 1])]

missing_bmi_people = len(missing_bmi_df)
missing_bmi_stroke_known = len(missing_bmi_known)
all_stroke_known = len(all_known)

missing_bmi_stroke_count = int(
    (missing_bmi_known["stroke"] == 1).sum()
)

all_stroke_count = int(
    (all_known["stroke"] == 1).sum()
)

missing_bmi_rate = (
    missing_bmi_stroke_count / missing_bmi_stroke_known * 100
    if missing_bmi_stroke_known > 0
    else None
)

all_stroke_rate = (
    all_stroke_count / all_stroke_known * 100
    if all_stroke_known > 0
    else None
)

bmi_comparison = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람",
    ],
    "해당 인원": [
        missing_bmi_people,
        len(df),
    ],
    "뇌졸중 발생 여부 확인 인원": [
        missing_bmi_stroke_known,
        all_stroke_known,
    ],
    "뇌졸중 인원": [
        missing_bmi_stroke_count,
        all_stroke_count,
    ],
    "뇌졸중 비율(%)": [
        (
            round(missing_bmi_rate, 2)
            if missing_bmi_rate is not None
            else None
        ),
        (
            round(all_stroke_rate, 2)
            if all_stroke_rate is not None
            else None
        ),
    ],
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "※ BMI가 비어 있는 사람 수는 BMI 결측 행 전체를 센 값입니다. "
    "뇌졸중 비율은 stroke 값이 0 또는 1로 확인되는 사람만 "
    "분모에 포함해 계산합니다."
)


# -----------------------------
# 6. 흡연 상태별 사람 수
# -----------------------------
st.header("6. 흡연 상태별 사람 수")

st.write(
    "흡연 상태의 각 값에 해당하는 사람 수를 집계합니다. "
    "흡연 상태가 비어 있는 경우도 별도로 표시합니다."
)

smoking_data = df["smoking_status"].fillna(
    "빈 값"
).replace("", "빈 값")

smoking_counts = (
    smoking_data
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True,
)

fig_smoking = px.bar(
    smoking_counts,
    x="흡연 상태",
    y="사람 수",
    text="사람 수",
    title="흡연 상태별 사람 수",
    labels={
        "흡연 상태": "흡연 상태",
        "사람 수": "사람 수",
    },
)

fig_smoking.update_traces(
    textposition="outside",
)

fig_smoking.update_layout(
    xaxis_title="흡연 상태",
    yaxis_title="사람 수",
    yaxis_rangemode="tozero",
)

st.plotly_chart(
    fig_smoking,
    use_container_width=True,
)


# -----------------------------
# 하단 안내
# -----------------------------
st.divider()

st.caption(
    "주의: 이 페이지는 데이터의 분포와 통계적 관련성을 "
    "탐색하기 위한 학습용 도구입니다. 그래프에서 나타나는 "
    "차이만으로 특정 요인이 뇌졸중의 원인이라고 단정할 수 없습니다."
)
