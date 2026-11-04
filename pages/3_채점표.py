
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="채점표",
    page_icon="📋",
    layout="wide"
)

st.title("📋 뇌졸중 분류 모델 채점표")
st.write(
    "테스트 데이터를 이용하여 두 분류 모델의 예측 결과를 평가합니다."
)

st.info(
    "뇌졸중(stroke)이 1이면 양성(뇌졸중), "
    "0이면 음성(뇌졸중 아님)입니다."
)


# =========================================================
# 2. 데이터 불러오기
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "k-suhyeon03/data/refs/heads/main/stroke.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 3. 입력 속성 선택
# =========================================================
st.header("1. 분류에 사용할 속성 선택")

feature_labels = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease"
}

default_features = [
    "나이",
    "평균 혈당",
    "고혈압",
    "심장병"
]

selected_labels = st.multiselect(
    "모델의 입력으로 사용할 속성을 선택하세요.",
    options=list(feature_labels.keys()),
    default=default_features
)

if len(selected_labels) < 2:
    st.warning("분류 모델을 만들려면 속성을 두 개 이상 선택해야 합니다.")
    st.stop()

selected_features = [
    feature_labels[label]
    for label in selected_labels
]

st.write("선택한 속성: " + ", ".join(selected_labels))


# =========================================================
# 4. 데이터 정리 및 학습용·테스트용 분리
# =========================================================
st.header("2. 학습용 데이터와 테스트용 데이터 분리")

required_columns = ["id", "stroke"] + selected_features

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

data = df[required_columns].copy()

data["id"] = pd.to_numeric(data["id"], errors="coerce")
data["stroke"] = pd.to_numeric(data["stroke"], errors="coerce")

data = data.dropna(subset=["id", "stroke"])
data = data[data["stroke"].isin([0, 1])].copy()

data = data.sort_values("id").reset_index(drop=True)

for col in selected_features:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 번호 순으로 정렬한 뒤 10명 중 앞 3명을 테스트용으로 지정
test_mask = np.arange(len(data)) % 10 < 3

train_df = data.loc[~test_mask].copy().reset_index(drop=True)
test_df = data.loc[test_mask].copy().reset_index(drop=True)

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

st.write(
    f"전체 데이터 **{len(data):,}명** 중 "
    f"학습용 **{len(train_df):,}명**, "
    f"테스트용 **{len(test_df):,}명**으로 나누었습니다."
)

if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르다면 원본 데이터의 행 수를 확인하세요."
    )


# =========================================================
# 5. 결측치 처리
# =========================================================
st.header("3. 결측치 처리")

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("학습용 체질량지수 데이터에 유효한 값이 없습니다.")
        st.stop()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.write(
        f"체질량지수의 결측값은 학습용 데이터의 중앙값 "
        f"**{bmi_median:.2f}**으로 채웠습니다."
    )
else:
    st.write("체질량지수를 선택하지 않아 BMI 결측치 처리는 생략했습니다.")

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 속성에 BMI 이외의 결측값이 있습니다. "
        "원본 데이터를 확인해 주세요."
    )
    st.stop()


# =========================================================
# 6. 두 모델 학습
# =========================================================
st.header("4. 분류 모델 학습")

# 로지스틱 회귀: 표준화 포함
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# 의사결정트리: 최대 깊이 3, 최소 잎 노드 인원 5명
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

st.success("두 모델의 학습이 완료되었습니다.")


# =========================================================
# 7. 예측 결과 계산
# =========================================================
logistic_train_pred = logistic_model.predict(X_train)
logistic_test_pred = logistic_model.predict(X_test)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

# 기준 모델: 학습용 데이터에서 더 많은 쪽만 예측
majority_class = int(y_train.value_counts().idxmax())

baseline_train_pred = np.full(
    len(y_train),
    majority_class
)

baseline_test_pred = np.full(
    len(y_test),
    majority_class
)


# =========================================================
# 8. 혼동행렬 계산 함수
# =========================================================
def get_confusion_values(y_true, y_pred):
    """
    양성: 뇌졸중(1)
    음성: 뇌졸중 아님(0)

    반환 순서: TN, FP, FN, TP
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn = int(matrix[0, 0])
    fp = int(matrix[0, 1])
    fn = int(matrix[1, 0])
    tp = int(matrix[1, 1])

    return tn, fp, fn, tp


def make_confusion_table(y_true, y_pred):
    tn, fp, fn, tp = get_confusion_values(
        y_true,
        y_pred
    )

    # 행은 실제, 열은 예측
    table = pd.DataFrame(
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
            "실제 뇌졸중 아님 (0)",
            "실제 뇌졸중 (1)",
            "열 합계"
        ],
        columns=[
            "예측 뇌졸중 아님 (0)",
            "예측 뇌졸중 (1)",
            "행 합계"
        ]
    )

    return table


# =========================================================
# 9. 혼동행렬 표시
# =========================================================
st.header("5. 테스트 데이터 혼동행렬")

test_positive_count = int((y_test == 1).sum())

st.write(
    f"테스트용 데이터는 **{len(y_test):,}명**이며, "
    f"그중 실제 뇌졸중인 사람은 **{test_positive_count:,}명**입니다."
)

st.caption(
    "행은 실제 정답, 열은 모델의 예측입니다. "
    "TP는 실제 뇌졸중을 뇌졸중으로 예측한 사람, "
    "TN은 실제 뇌졸중 아님을 뇌졸중 아님으로 예측한 사람입니다. "
    "FP는 헛짚은 사람, FN은 놓친 사람입니다."
)

st.subheader("로지스틱 회귀 (확률로 답하는 모델)")

logistic_matrix = make_confusion_table(
    y_test,
    logistic_test_pred
)

st.table(logistic_matrix)

st.subheader("의사결정트리 (질문으로 답하는 모델)")

tree_matrix = make_confusion_table(
    y_test,
    tree_test_pred
)

st.table(tree_matrix)


# =========================================================
# 10. 정확도·재현율·정밀도·F1 계산
# =========================================================
def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = get_confusion_values(
        y_true,
        y_pred
    )

    total = tn + fp + fn + tp

    # 정확도: 전체 중 올바르게 예측한 비율
    accuracy = (
        (tp + tn) / total
        if total > 0 else None
    )

    # 재현율: 실제 뇌졸중인 사람 중 뇌졸중이라고 예측한 비율
    actual_positive = tp + fn

    recall = (
        tp / actual_positive
        if actual_positive > 0 else None
    )

    # 정밀도: 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율
    predicted_positive = tp + fp

    precision = (
        tp / predicted_positive
        if predicted_positive > 0 else None
    )

    # F1 = 2 * 정밀도 * 재현율 / (정밀도 + 재현율)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = None
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# 각 모델의 훈련용·테스트용 지표
logistic_train_metrics = calculate_metrics(
    y_train,
    logistic_train_pred
)

logistic_test_metrics = calculate_metrics(
    y_test,
    logistic_test_pred
)

tree_train_metrics = calculate_metrics(
    y_train,
    tree_train_pred
)

tree_test_metrics = calculate_metrics(
    y_test,
    tree_test_pred
)

baseline_train_metrics = calculate_metrics(
    y_train,
    baseline_train_pred
)

baseline_test_metrics = calculate_metrics(
    y_test,
    baseline_test_pred
)


# =========================================================
# 11. 세 모델의 평가 지표 비교
# =========================================================
st.header("6. 세 모델의 평가 지표 비교")

st.write(
    "정확도, 재현율, 정밀도, F1은 모두 테스트 데이터를 기준으로 "
    "비교합니다. 훈련용에서 많은 쪽만 예측하는 모델도 함께 표시합니다."
)

metric_table = pd.DataFrame([
    {
        "모델": "로지스틱 회귀 (확률로 답하는 모델)",
        "정확도": format_metric(logistic_test_metrics["정확도"]),
        "재현율": format_metric(logistic_test_metrics["재현율"]),
        "정밀도": format_metric(logistic_test_metrics["정밀도"]),
        "F1": format_metric(logistic_test_metrics["F1"])
    },
    {
        "모델": "의사결정트리 (질문으로 답하는 모델)",
        "정확도": format_metric(tree_test_metrics["정확도"]),
        "재현율": format_metric(tree_test_metrics["재현율"]),
        "정밀도": format_metric(tree_test_metrics["정밀도"]),
        "F1": format_metric(tree_test_metrics["F1"])
    },
    {
        "모델": "훈련용에서 많은 쪽만 예측하는 모델",
        "정확도": format_metric(baseline_test_metrics["정확도"]),
        "재현율": format_metric(baseline_test_metrics["재현율"]),
        "정밀도": format_metric(baseline_test_metrics["정밀도"]),
        "F1": format_metric(baseline_test_metrics["F1"])
    }
])

st.dataframe(
    metric_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 12. 놓친 환자와 헛짚은 사람 목록
# =========================================================
st.header("7. 잘못 예측한 사람 확인")

st.write(
    "모델이 잘못 예측한 사람을 확인합니다. "
    "모델을 선택하면 해당 모델의 놓친 환자와 헛짚은 사람 목록이 표시됩니다."
)

model_options = {
    "로지스틱 회귀 (확률로 답하는 모델)": logistic_test_pred,
    "의사결정트리 (질문으로 답하는 모델)": tree_test_pred
}

selected_model_name = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    options=list(model_options.keys())
)

selected_prediction = model_options[selected_model_name]


# =========================================================
# 13. 오류 예측 데이터 구성
# =========================================================
# 테스트 데이터 원본에서 번호와 표시할 속성을 가져옴
display_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

available_display_columns = [
    col for col in display_columns
    if col in df.columns
]

test_details = df.loc[
    test_df.index if False else test_df.index
].copy()

# 분할된 테스트 데이터의 원래 행을 기준으로 표시 데이터 구성
# data는 번호 순으로 정렬된 데이터이므로 id를 기준으로 연결
test_details = test_df[
    ["id", "stroke"]
].merge(
    data[available_display_columns + ["id"]],
    on="id",
    how="left"
)

test_details["실제"] = y_test.to_numpy()
test_details["예측"] = selected_prediction

# 실제 뇌졸중인데 뇌졸중 아님으로 예측한 사람(FN)
missed_people = test_details[
    (test_details["실제"] == 1)
    & (test_details["예측"] == 0)
].copy()

# 실제 뇌졸중 아님인데 뇌졸중으로 예측한 사람(FP)
false_alarm_people = test_details[
    (test_details["실제"] == 0)
    & (test_details["예측"] == 1)
].copy()

# 나이가 많은 순서로 정렬
missed_people = missed_people.sort_values(
    "age",
    ascending=False
)

false_alarm_people = false_alarm_people.sort_values(
    "age",
    ascending=False
)


# =========================================================
# 14. 목록 표시용 데이터 가공
# =========================================================
def prepare_display_table(person_df):
    result = person_df[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease"
        ]
    ].copy()

    result = result.rename(columns={
        "id": "번호",
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "hypertension": "고혈압 여부",
        "heart_disease": "심장병 여부"
    })

    result["고혈압 여부"] = result["고혈압 여부"].map({
        0: "없음",
        1: "있음"
    }).fillna("정보 없음")

    result["심장병 여부"] = result["심장병 여부"].map({
        0: "없음",
        1: "있음"
    }).fillna("정보 없음")

    result["나이"] = pd.to_numeric(
        result["나이"],
        errors="coerce"
    )

    result["평균 혈당"] = pd.to_numeric(
        result["평균 혈당"],
        errors="coerce"
    ).round(2)

    result["번호"] = result["번호"].astype(int)

    return result


missed_display = prepare_display_table(missed_people)
false_alarm_display = prepare_display_table(false_alarm_people)


# =========================================================
# 15. 두 목록 나란히 표시
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader(
        f"놓친 환자 (FN): {len(missed_display):,}명"
    )

    st.caption(
        "실제 뇌졸중이지만 모델이 뇌졸중 아님으로 예측한 사람"
    )

    st.dataframe(
        missed_display.head(20),
        use_container_width=True,
        hide_index=True
    )

    if len(missed_display) > 20:
        st.caption(
            f"전체 {len(missed_display):,}명 중 "
            "나이가 많은 순서로 앞 20명만 표시했습니다."
        )

with col2:
    st.subheader(
        f"헛짚은 사람 (FP): {len(false_alarm_display):,}명"
    )

    st.caption(
        "실제 뇌졸중은 아니지만 모델이 뇌졸중이라고 예측한 사람"
    )

    st.dataframe(
        false_alarm_display.head(20),
        use_container_width=True,
        hide_index=True
    )

    if len(false_alarm_display) > 20:
        st.caption(
            f"전체 {len(false_alarm_display):,}명 중 "
            "나이가 많은 순서로 앞 20명만 표시했습니다."
        )


# =========================================================
# 16. 혼동행렬과 목록 인원수 확인
# =========================================================
st.divider()

selected_tn, selected_fp, selected_fn, selected_tp = (
    get_confusion_values(y_test, selected_prediction)
)

st.write(
    f"선택한 모델의 혼동행렬에서 FN은 **{selected_fn:,}명**, "
    f"FP는 **{selected_fp:,}명**이며, "
    f"위 목록의 놓친 환자 **{len(missed_display):,}명**, "
    f"헛짚은 사람 **{len(false_alarm_display):,}명**과 일치합니다."
)

st.caption(
    "※ 이 페이지는 분류 알고리즘의 성능을 비교하는 교육용 실습입니다. "
    "실제 의료 진단이나 뇌졸중 위험 판단에 사용할 수 없습니다."
)
