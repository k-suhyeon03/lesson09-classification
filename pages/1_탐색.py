
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
    "뇌졸중 발생의 관계를 살펴봅니다."
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
    st.error("데이터를 불러오지 못했습니다.")
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
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error("필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()


# 숫자형 데이터 변환
for col in [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "bmi",
    "stroke",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# 뇌졸중 발생 여부가 0 또는 1로 확인되는 행
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
    "히스토그램을 이용해 나이와 평균 혈당이 "
    "어떤 구간에 많이 분포하는지 확인합니다."
)

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df.dropna(subset=["age"]),
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
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df.dropna(subset=["avg_glucose_level"]),
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
    st.plotly_chart(fig_glucose, use_container_width=True)


# -----------------------------
# 2. 뇌졸중 여부에 따른 상자그림
# -----------------------------
st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

st.write(
    "상자그림으로 두 그룹의 중앙값과 자료의 퍼짐을 비교합니다. "
    "아래 표에서는 두 그룹의 평균값도 확인할 수 있습니다."
)

col1, col2 = st.columns(2)

with col1:
    age_box_data = stroke_df.dropna(
        subset=["뇌졸중 여부", "age"]
    )

    fig_age_box = px.box(
        age_box_data,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders={
            "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]
        },
        title="뇌졸중 여부에 따른 나이",
        labels={
            "age": "나이",
            "뇌졸중 여부": "그룹",
        },
        points="outliers",
    )
    fig_age_box.update_layout(showlegend=False)
    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    glucose_box_data = stroke_df.dropna(
        subset=["뇌졸중 여부", "avg_glucose_level"]
    )

    fig_glucose_box = px.box(
        glucose_box_data,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders={
            "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]
        },
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "그룹",
        },
        points="outliers",
    )
    fig_glucose_box.update_layout(showlegend=False)
    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 평균값 표
mean_table = (
    stroke_df
    .groupby("뇌졸중 여부")
    .agg(
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .reset_index()
)

mean_table = mean_table.rename(columns={
    "뇌졸중 여부": "뇌졸중 여부",
    "평균_나이": "평균 나이",
    "평균_혈당": "평균 혈당",
})

mean_table[["평균 나이", "평균 혈당"]] = (
    mean_table[["평균 나이", "평균 혈당"]].round(2)
)

st.subheader("두 그룹의 평균값")
st.dataframe(mean_table, use_container_width=True, hide_index=True)


# -----------------------------
# 3. 고혈압 여부에 따른 뇌졸중 비율
# -----------------------------
st.header("3. 고혈압 여부에 따른 뇌졸중 비율")

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
        전체_인원=("stroke", "size"),
        뇌졸중_인원=("stroke", "sum"),
        뇌졸중_비율=("stroke", "mean"),
    )
    .reindex(["고혈압 없음", "고혈압 있음"])
    .reset_index()
)

hypertension_rate["뇌졸중 비율(%)"] = (
    hypertension_rate["뇌졸중_비율"] * 100
).round(2)

hypertension_display = hypertension_rate.rename(columns={
    "고혈압 여부": "고혈압 여부",
    "전체_인원": "전체 인원",
    "뇌졸중_인원": "뇌졸중 인원",
})

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
fig_hypertension.update_layout(yaxis_rangemode="tozero")
st.plotly_chart(fig_hypertension, use_container_width=True)

st.dataframe(
    hypertension_display[
        [
            "고혈압 여부",
            "전체_인원",
            "뇌졸중_인원",
            "뇌졸중 비율(%)",
        ]
    ].rename(columns={
        "전체_인원": "전체 인원",
        "뇌졸중_인원": "뇌졸중 인원",
    }),
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

heart_df["심장병 여부"] = heart_df["heart_disease"].map({
    0: "심장병 없음",
    1: "심장병 있음",
})

heart_rate = (
    heart_df
    .groupby("심장병 여부")
    .agg(
        전체_인원=("stroke", "size"),
        뇌졸중_인원=("stroke", "sum"),
        뇌졸중_비율=("stroke", "mean"),
    )
    .reindex(["심장병 없음", "심장병 있음"])
    .reset_index()
)

heart_rate["뇌졸중 비율(%)"] = (
    heart_rate["뇌졸중_비율"] * 100
).round(2)

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
fig_heart.update_layout(yaxis_rangemode="tozero")
st.plotly_chart(fig_heart, use_container_width=True)

st.dataframe(
    heart_rate[
        [
            "심장병 여부",
            "전체_인원",
            "뇌졸중_인원",
            "뇌졸중 비율(%)",
        ]
    ].rename(columns={
        "전체_인원": "전체 인원",
        "뇌졸중_인원": "뇌졸중 인원",
    }),
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# 5. BMI 빈 값과 뇌졸중 비율
# -----------------------------
st.header("5. BMI가 비어 있는 사람들의 뇌졸중 비율")

st.write(
    "BMI가 비어 있는 사람들의 뇌졸중 비율을 "
    "전체 데이터의 비율과 비교합니다."
)

missing_bmi_df = df[df["bmi"].isna()].copy()

missing_bmi_known = missing_bmi_df[
    missing_bmi_df["stroke"].isin([0, 1])
]

all_known = df[df["stroke"].isin([0, 1])]

missing_bmi_stroke_count = int(
    (missing_bmi_known["stroke"] == 1).sum()
)
all_stroke_count = int(
    (all_known["stroke"] == 1).sum()
)

missing_bmi_known_count = len(missing_bmi_known)
all_known_count = len(all_known)

missing_bmi_rate = (
    missing_bmi_stroke_count / missing_bmi_known_count * 100
    if missing_bmi_known_count > 0
    else None
)

all_stroke_rate = (
    all_stroke_count / all_known_count * 100
    if all_known_count > 0
    else None
)

bmi_comparison = pd.DataFrame({
    "구분": ["BMI가 비어 있는 사람", "전체 사람"],
    "해당 인원": [len(missing_bmi_df), len(df)],
    "뇌졸중 여부 확인 인원": [
        missing_bmi_known_count,
        all_known_count,
    ],
    "뇌졸중 인원": [
        missing_bmi_stroke_count,
        all_stroke_count,
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_rate, 2)
        if missing_bmi_rate is not None else None,
        round(all_stroke_rate, 2)
        if all_stroke_rate is not None else None,
    ],
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "※ BMI가 비어 있는 사람 수는 결측 행 전체를 센 값입니다. "
    "뇌졸중 비율은 stroke 값이 0 또는 1로 확인된 사람만 "
    "대상으로 계산했습니다."
)


# -----------------------------
# 6. 흡연 상태별 사람 수
# -----------------------------
st.header("6. 흡연 상태별 사람 수")

smoking_data = (
    df["smoking_status"]
    .fillna("빈 값")
    .replace("", "빈 값")
)

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

fig_smoking.update_traces(textposition="outside")
fig_smoking.update_layout(yaxis_rangemode="tozero")

st.plotly_chart(fig_smoking, use_container_width=True)


# -----------------------------
# 하단 안내
# -----------------------------
st.divider()

st.caption(
    "이 페이지는 데이터 탐색을 위한 학습용 도구입니다. "
    "그룹 간 차이만으로 특정 요인이 뇌졸중의 원인이라고 "
    "단정할 수 없습니다."
)
