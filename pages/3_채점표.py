
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="채점표",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 분류 모델 채점표")
st.write(
    "학습한 분류 모델이 테스트 데이터의 뇌졸중 여부를 "
    "얼마나 정확하게 예측했는지 평가합니다."
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "k-suhyeon03/data/refs/heads/main/stroke.csv"
)


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


# --------------------------------------------------
# 3. 변수 설정
# --------------------------------------------------

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

DISPLAY_LABELS = {
    "id": "ID",
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]


# --------------------------------------------------
# 4. 사용할 변수 선택
# --------------------------------------------------

st.subheader("1. 모델에 사용할 변수 선택")

st.write(
    "아래 변수 중 2개 이상을 선택해 주세요. "
    "체질량지수(BMI)는 선택하지 않아도 됩니다."
)

selected_features = st.multiselect(
    "분류 모델에 사용할 변수",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x]
)

if len(selected_features) < 2:
    st.warning("분류 모델을 만들려면 변수를 2개 이상 선택해야 합니다.")
    st.stop()


# --------------------------------------------------
# 5. 데이터 전처리
# --------------------------------------------------

required_columns = [
    "id",
    "stroke",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()


# 원본 데이터의 모든 열을 유지
data = df.copy()

# 숫자형으로 변환
numeric_columns = [
    "id",
    "stroke",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# ID와 정답이 없는 행 제거
data = data.dropna(subset=["id", "stroke"])

# 정답이 0 또는 1인 데이터만 사용
data = data[data["stroke"].isin([0, 1])].copy()

# ID 순서대로 정렬
data = data.sort_values("id").reset_index(drop=True)


if len(data) == 0:
    st.error("분석에 사용할 데이터가 없습니다.")
    st.stop()


# 선택한 변수에 결측치가 있는지 확인
# BMI는 아래에서 학습 데이터의 중앙값으로 처리
for feature in selected_features:
    if feature != "bmi" and data[feature].isna().any():
        st.error(
            f"선택한 변수 '{FEATURE_LABELS[feature]}'에 "
            "결측치가 있습니다."
        )
        st.stop()


# --------------------------------------------------
# 6. 학습 데이터와 테스트 데이터 분리
# --------------------------------------------------

st.subheader("2. 학습 데이터와 테스트 데이터 분리")

st.write(
    "ID 순서대로 10명씩 묶은 뒤, "
    "각 묶음에서 앞의 3명은 테스트 데이터, "
    "나머지 7명은 학습 데이터로 사용합니다."
)

# 10명 중 앞의 3명을 테스트 데이터로 설정
test_mask = np.arange(len(data)) % 10 < 3

# 원본 데이터의 모든 열을 그대로 보존
train_df = data.loc[~test_mask].copy().reset_index(drop=True)
test_df = data.loc[test_mask].copy().reset_index(drop=True)

# 선택한 변수만 모델 입력값으로 사용
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)


# --------------------------------------------------
# 7. BMI 결측치 처리
# --------------------------------------------------

if "bmi" in selected_features:

    # 학습 데이터의 BMI 중앙값 계산
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("학습 데이터에서 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    # 학습 데이터와 테스트 데이터 모두 학습 데이터 중앙값 사용
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)


# --------------------------------------------------
# 8. 모델 학습
# --------------------------------------------------

st.subheader("3. 분류 모델 학습")

# 로지스틱 회귀
# 표준화는 학습 데이터에 대해서만 학습
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        random_state=42,
        max_iter=1000
    )
)

logistic_model.fit(X_train, y_train)

# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)


# --------------------------------------------------
# 9. 테스트 데이터 예측
# --------------------------------------------------

logistic_pred = logistic_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

# 로지스틱 회귀의 뇌졸중 예측 확률
logistic_probability = logistic_model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 10. 기준 모델
# --------------------------------------------------

# 학습 데이터에서 더 많은 비율을 차지하는 클래스를 확인
majority_class = int(y_train.value_counts().idxmax())

# 모든 테스트 데이터를 다수 클래스라고 예측
baseline_pred = np.full(
    len(y_test),
    majority_class
)


# --------------------------------------------------
# 11. 혼동행렬 및 평가 지표 계산 함수
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else None

    actual_positive = tp + fn
    predicted_positive = tp + fp

    if actual_positive > 0:
        recall = tp / actual_positive
    else:
        recall = None

    if predicted_positive > 0:
        precision = tp / predicted_positive
    else:
        precision = None

    if precision is not None and recall is not None:
        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = None
    else:
        f1 = None

    return {
        "cm": cm,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }


logistic_result = calculate_metrics(
    y_test,
    logistic_pred
)

tree_result = calculate_metrics(
    y_test,
    tree_pred
)

baseline_result = calculate_metrics(
    y_test,
    baseline_pred
)


# --------------------------------------------------
# 12. 테스트 데이터 정보
# --------------------------------------------------

st.subheader("4. 테스트 데이터 구성")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "테스트 데이터 수",
        f"{len(y_test):,}명"
    )

with col2:
    st.metric(
        "실제 뇌졸중 인원",
        f"{int((y_test == 1).sum()):,}명"
    )

with col3:
    st.metric(
        "실제 정상 인원",
        f"{int((y_test == 0).sum()):,}명"
    )


# --------------------------------------------------
# 13. 혼동행렬 출력 함수
# --------------------------------------------------

def display_confusion_matrix(result, title):

    st.markdown(f"#### {title}")

    tn = result["TN"]
    fp = result["FP"]
    fn = result["FN"]
    tp = result["TP"]

    matrix_df = pd.DataFrame(
        [
            [
                f"TN\n{tn:,}명",
                f"FP\n{fp:,}명",
                f"{tn + fp:,}명"
            ],
            [
                f"FN\n{fn:,}명",
                f"TP\n{tp:,}명",
                f"{fn + tp:,}명"
            ],
            [
                f"{tn + fn:,}명",
                f"{fp + tp:,}명",
                f"{tn + fp + fn + tp:,}명"
            ]
        ],
        index=[
            "실제 정상 (0)",
            "실제 뇌졸중 (1)",
            "합계"
        ],
        columns=[
            "예측 정상 (0)",
            "예측 뇌졸중 (1)",
            "합계"
        ]
    )

    st.table(matrix_df)


# --------------------------------------------------
# 14. 혼동행렬 비교
# --------------------------------------------------

st.subheader("5. 혼동행렬 비교")

st.write(
    "혼동행렬은 실제 정답과 모델의 예측 결과를 비교한 표입니다."
)

col1, col2 = st.columns(2)

with col1:
    display_confusion_matrix(
        logistic_result,
        "로지스틱 회귀 (확률로 답하는 모델)"
    )

with col2:
    display_confusion_matrix(
        tree_result,
        "의사결정트리 (질문으로 답하는 모델)"
    )


# --------------------------------------------------
# 15. 혼동행렬 용어 설명
# --------------------------------------------------

with st.expander("혼동행렬의 항목 알아보기"):

    st.markdown("""
    - **TN (진짜 음성)**: 실제 정상인 사람을 정상으로 예측
    - **FP (거짓 양성)**: 실제 정상인 사람을 뇌졸중으로 예측
    - **FN (거짓 음성)**: 실제 뇌졸중인 사람을 정상으로 예측
    - **TP (진짜 양성)**: 실제 뇌졸중인 사람을 뇌졸중으로 예측
    """)


# --------------------------------------------------
# 16. 평가 지표 계산 및 비교
# --------------------------------------------------

st.subheader("6. 모델 성능 평가")

st.write(
    "정확도, 재현율, 정밀도, F1 점수를 이용해 "
    "세 가지 모델의 성능을 비교합니다."
)


def format_metric(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


metrics_df = pd.DataFrame(
    {
        "모델": [
            "로지스틱 회귀",
            "의사결정트리",
            "기준 모델"
        ],
        "정확도": [
            format_metric(logistic_result["accuracy"]),
            format_metric(tree_result["accuracy"]),
            format_metric(baseline_result["accuracy"])
        ],
        "재현율": [
            format_metric(logistic_result["recall"]),
            format_metric(tree_result["recall"]),
            format_metric(baseline_result["recall"])
        ],
        "정밀도": [
            format_metric(logistic_result["precision"]),
            format_metric(tree_result["precision"]),
            format_metric(baseline_result["precision"])
        ],
        "F1 점수": [
            format_metric(logistic_result["f1"]),
            format_metric(tree_result["f1"]),
            format_metric(baseline_result["f1"])
        ]
    }
)

st.dataframe(
    metrics_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 17. 평가 지표 설명
# --------------------------------------------------

with st.expander("평가 지표의 의미와 계산 방법"):

    st.markdown("""
    **① 정확도 (Accuracy)**

    전체 예측 중 실제 정답을 맞힌 비율입니다.

    \[
    정확도 = \\frac{TP+TN}{TP+TN+FP+FN}
    \]

    **② 재현율 (Recall)**

    실제 뇌졸중인 사람 중 모델이 뇌졸중이라고 예측한 비율입니다.

    \[
    재현율 = \\frac{TP}{TP+FN}
    \]

    **③ 정밀도 (Precision)**

    모델이 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율입니다.

    \[
    정밀도 = \\frac{TP}{TP+FP}
    \]

    **④ F1 점수 (F1 Score)**

    정밀도와 재현율의 조화평균입니다.

    \[
    F1 = 2 \\times \\frac{정밀도 \\times 재현율}
    {정밀도+재현율}
    \]

    정밀도 또는 재현율을 계산할 수 없는 경우 F1 점수도 계산할 수 없습니다.
    """)


# --------------------------------------------------
# 18. 기준 모델 설명
# --------------------------------------------------

with st.expander("기준 모델이란?"):

    st.write(
        f"기준 모델은 학습 데이터에서 더 많은 비율을 차지하는 "
        f"클래스인 {majority_class}만을 모든 테스트 데이터의 "
        "예측 결과로 사용하는 모델입니다."
    )

    if majority_class == 0:
        st.write(
            "현재 학습 데이터에서는 정상(0)이 더 많으므로 "
            "모든 사람을 정상이라고 예측합니다."
        )
    else:
        st.write(
            "현재 학습 데이터에서는 뇌졸중(1)이 더 많으므로 "
            "모든 사람을 뇌졸중이라고 예측합니다."
        )

    st.write(
        "이 모델의 성능과 다른 분류 모델의 성능을 비교하면 "
        "분류 모델이 단순한 다수 클래스 예측보다 얼마나 나은지 "
        "확인할 수 있습니다."
    )


# --------------------------------------------------
# 19. 오답 데이터 확인
# --------------------------------------------------

st.subheader("7. 모델의 오답 데이터 확인")

st.write(
    "선택한 모델이 테스트 데이터에서 잘못 예측한 사례를 확인합니다."
)

selected_model_name = st.selectbox(
    "오답을 확인할 모델 선택",
    [
        "로지스틱 회귀 (확률로 답하는 모델)",
        "의사결정트리 (질문으로 답하는 모델)"
    ]
)


if selected_model_name.startswith("로지스틱"):

    selected_pred = logistic_pred
    selected_result = logistic_result

else:

    selected_pred = tree_pred
    selected_result = tree_result


# --------------------------------------------------
# 20. 테스트 데이터의 예측 결과 정리
# --------------------------------------------------

# test_df에는 선택한 변수 외의 원본 정보도 그대로 남아 있음
test_details = test_df[
    [
        "id",
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ]
].copy()

test_details["실제 결과"] = y_test.to_numpy()
test_details["예측 결과"] = selected_pred

# 실제 뇌졸중인데 정상으로 예측한 경우
false_negative = test_details[
    (test_details["실제 결과"] == 1)
    & (test_details["예측 결과"] == 0)
].copy()

# 실제 정상인데 뇌졸중으로 예측한 경우
false_positive = test_details[
    (test_details["실제 결과"] == 0)
    & (test_details["예측 결과"] == 1)
].copy()


# --------------------------------------------------
# 21. 오답 데이터 표시 함수
# --------------------------------------------------

def display_error_table(error_df, error_type, description):

    st.markdown(f"#### {error_type} ({len(error_df):,}명)")

    st.write(description)

    if len(error_df) == 0:
        st.success("해당 유형의 오답이 없습니다.")
        return

    # 나이가 많은 순서대로 정렬
    error_df = error_df.sort_values(
        "age",
        ascending=False
    )

    # 표시할 열 이름 변경
    display_df = error_df[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease"
        ]
    ].rename(
        columns={
            "id": "ID",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압",
            "heart_disease": "심장병"
        }
    )

    st.caption(
        f"총 {len(display_df):,}명 중 최대 20명까지 표시합니다. "
        "나이가 많은 순서입니다."
    )

    st.dataframe(
        display_df.head(20),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 22. 거짓 음성 및 거짓 양성 출력
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    display_error_table(
        false_negative,
        "거짓 음성 (FN)",
        "실제로 뇌졸중인 사람을 정상이라고 잘못 예측한 사례입니다."
    )

with col2:

    display_error_table(
        false_positive,
        "거짓 양성 (FP)",
        "실제로 정상인 사람을 뇌졸중이라고 잘못 예측한 사례입니다."
    )


# --------------------------------------------------
# 23. 오답 수와 혼동행렬 비교
# --------------------------------------------------

st.markdown("#### 오답 수 확인")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "거짓 음성 (FN)",
        f"{len(false_negative):,}명",
        help="혼동행렬에서 FN에 해당하는 수와 같습니다."
    )

with col2:
    st.metric(
        "거짓 양성 (FP)",
        f"{len(false_positive):,}명",
        help="혼동행렬에서 FP에 해당하는 수와 같습니다."
    )

st.caption(
    "거짓 음성의 인원은 혼동행렬의 FN과, "
    "거짓 양성의 인원은 혼동행렬의 FP와 일치합니다."
)


# --------------------------------------------------
# 24. 마무리 안내
# --------------------------------------------------

st.divider()

st.info(
    "이 페이지는 분류 모델의 작동 원리와 평가 방법을 학습하기 위한 "
    "실습 자료입니다. 실제 뇌졸중 진단이나 의료적 판단에 "
    "사용할 수 있는 프로그램은 아닙니다."
)
