
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 분류 모델 성능 높이기")

st.write(
    "뇌졸중 환자 데이터가 정상 데이터보다 훨씬 적다는 점에 주목하여, "
    "클래스 가중치를 적용하기 전과 후의 분류 모델 성능을 비교합니다."
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
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 3. 변수 이름 설정
# --------------------------------------------------

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

REQUIRED_COLUMNS = [
    "id",
    "stroke",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease"
]


# --------------------------------------------------
# 4. 입력 속성 선택
# --------------------------------------------------

st.subheader("1. 모델에 사용할 속성 선택")

st.write(
    "분류 모델의 입력값으로 사용할 속성을 선택해 주세요. "
    "2개 이상 선택해야 합니다."
)

selected_features = st.multiselect(
    "입력 속성",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x]
)

if len(selected_features) < 2:
    st.warning("속성을 2개 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 5. 데이터 확인 및 전처리
# --------------------------------------------------

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()


data = df.copy()

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
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

data = data.dropna(
    subset=["id", "stroke"]
)

data = data[
    data["stroke"].isin([0, 1])
].copy()

data = data.sort_values(
    "id"
).reset_index(drop=True)

if len(data) == 0:
    st.error("분석에 사용할 데이터가 없습니다.")
    st.stop()


for feature in selected_features:

    if feature != "bmi" and data[feature].isna().any():

        st.error(
            f"'{FEATURE_LABELS[feature]}'에 결측치가 있습니다. "
            "해당 데이터를 확인해 주세요."
        )

        st.stop()


# --------------------------------------------------
# 6. 학습 데이터와 테스트 데이터 분리
# --------------------------------------------------

st.subheader("2. 학습 데이터와 테스트 데이터 분리")

st.write(
    "ID 순서대로 10명씩 묶고, 각 묶음에서 앞의 3명은 테스트 데이터, "
    "나머지 7명은 학습 데이터로 사용합니다."
)

test_mask = np.arange(len(data)) % 10 < 3

train_df = data.loc[
    ~test_mask
].copy().reset_index(drop=True)

test_df = data.loc[
    test_mask
].copy().reset_index(drop=True)

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)


# --------------------------------------------------
# 7. BMI 결측치 처리
# --------------------------------------------------

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련 데이터에서 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )


# --------------------------------------------------
# 8. 클래스 가중치 적용 전후 모델 학습
# --------------------------------------------------

st.subheader("3. 클래스 가중치 적용 전후 모델 학습")

st.write(
    "클래스 가중치를 적용하지 않은 모델과 "
    "클래스별 데이터 수의 차이를 고려한 모델을 각각 학습합니다."
)

# 로지스틱 회귀 - 적용 전
logistic_before = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight=None
    )
)

logistic_before.fit(
    X_train,
    y_train
)


# 로지스틱 회귀 - 적용 후
logistic_after = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )
)

logistic_after.fit(
    X_train,
    y_train
)


# 의사결정트리 - 적용 전
tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight=None
)

tree_before.fit(
    X_train,
    y_train
)


# 의사결정트리 - 적용 후
tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

tree_after.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 9. 테스트 데이터 예측
# --------------------------------------------------

logistic_pred_before = logistic_before.predict(X_test)
logistic_pred_after = logistic_after.predict(X_test)

tree_pred_before = tree_before.predict(X_test)
tree_pred_after = tree_after.predict(X_test)


# --------------------------------------------------
# 10. 평가 지표 계산
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0 else None
    )

    actual_positive = tp + fn
    predicted_positive = tp + fp

    recall = (
        tp / actual_positive
        if actual_positive > 0 else None
    )

    precision = (
        tp / predicted_positive
        if predicted_positive > 0 else None
    )

    if (
        precision is not None
        and recall is not None
        and precision + recall > 0
    ):

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:
        f1 = None

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }


logistic_result_before = calculate_metrics(
    y_test,
    logistic_pred_before
)

logistic_result_after = calculate_metrics(
    y_test,
    logistic_pred_after
)

tree_result_before = calculate_metrics(
    y_test,
    tree_pred_before
)

tree_result_after = calculate_metrics(
    y_test,
    tree_pred_after
)


# --------------------------------------------------
# 11. 비교표 생성
# --------------------------------------------------

def format_metric(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


def make_comparison_table(before, after):

    rows = [
        ("TN (진짜 음성)", "TN"),
        ("FP (거짓 양성)", "FP"),
        ("FN (거짓 음성)", "FN"),
        ("TP (진짜 양성)", "TP"),
        ("정확도", "accuracy"),
        ("재현율", "recall"),
        ("정밀도", "precision"),
        ("F1 점수", "f1")
    ]

    table_data = []

    for label, key in rows:

        if key in ["TN", "FP", "FN", "TP"]:

            before_value = f"{before[key]:,}명"
            after_value = f"{after[key]:,}명"

        else:

            before_value = format_metric(before[key])
            after_value = format_metric(after[key])

        table_data.append({
            "평가 항목": label,
            "가중치 적용 전": before_value,
            "가중치 적용 후": after_value
        })

    return pd.DataFrame(table_data)


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
        "실제 뇌졸중 환자 수",
        f"{int((y_test == 1).sum()):,}명"
    )

with col3:
    st.metric(
        "실제 정상인 수",
        f"{int((y_test == 0).sum()):,}명"
    )


# --------------------------------------------------
# 13. 로지스틱 회귀 결과
# --------------------------------------------------

st.subheader(
    "5. 로지스틱 회귀 (확률로 답하는 모델)"
)

st.write(
    "클래스 가중치를 적용하기 전과 적용한 후의 "
    "혼동행렬 및 평가 지표를 비교합니다."
)

logistic_table = make_comparison_table(
    logistic_result_before,
    logistic_result_after
)

st.dataframe(
    logistic_table,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 14. 의사결정트리 결과
# --------------------------------------------------

st.subheader(
    "6. 의사결정트리 (질문으로 답하는 모델)"
)

st.write(
    "동일한 방법으로 의사결정트리의 "
    "가중치 적용 전후 결과를 비교합니다."
)

tree_table = make_comparison_table(
    tree_result_before,
    tree_result_after
)

st.dataframe(
    tree_table,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 15. 환자 예측 결과 변화
# --------------------------------------------------

st.subheader("7. 뇌졸중 환자 예측 결과 변화")

st.write(
    "찾아낸 환자는 실제 뇌졸중 환자를 뇌졸중이라고 예측한 사람(TP), "
    "놓친 환자는 실제 뇌졸중 환자를 정상이라고 예측한 사람(FN), "
    "헛짚은 사람은 실제 정상인을 뇌졸중이라고 예측한 사람(FP)입니다."
)


def show_change_line(title, before, after):

    st.markdown(
        f"**{title}**  \n"
        f"찾아낸 환자: {before['TP']:,}명 → {after['TP']:,}명 "
        f" | 놓친 환자: {before['FN']:,}명 → {after['FN']:,}명 "
        f" | 헛짚은 사람: {before['FP']:,}명 → {after['FP']:,}명"
    )


show_change_line(
    "로지스틱 회귀",
    logistic_result_before,
    logistic_result_after
)

show_change_line(
    "의사결정트리",
    tree_result_before,
    tree_result_after
)


# --------------------------------------------------
# 16. 경계선 및 평면 시각화 준비
# --------------------------------------------------

st.subheader("8. 가중치 적용 전후의 분류 경계 비교")

st.write(
    "테스트 데이터의 실제 뇌졸중 여부를 점의 색으로 나타내고, "
    "각 모델의 가중치 적용 전후 0.5 경계를 겹쳐 표시합니다."
)

st.caption(
    "선택한 속성 중 그림의 축으로 사용하지 않는 속성은 "
    "훈련 데이터의 중앙값으로 고정합니다."
)

plot_features = selected_features[:3]


# --------------------------------------------------
# 17. 속성 범위 계산
# --------------------------------------------------

def get_feature_range(feature):

    values = X_train[feature].dropna()

    minimum = float(values.min())
    maximum = float(values.max())

    if minimum == maximum:
        minimum -= 1
        maximum += 1

    padding = (maximum - minimum) * 0.08

    return (
        minimum - padding,
        maximum + padding
    )


# --------------------------------------------------
# 18. 예측용 데이터 생성
# --------------------------------------------------

def make_prediction_data(feature_values):

    # 수정된 부분:
    # 입력 배열의 실제 길이를 기준으로 행 개수를 결정
    if not feature_values:
        raise ValueError("예측용 속성값이 없습니다.")

    lengths = [
        len(values)
        for values in feature_values.values()
    ]

    if len(set(lengths)) != 1:
        raise ValueError("예측용 속성 배열의 길이가 서로 다릅니다.")

    row_count = lengths[0]

    grid_df = pd.DataFrame(
        index=range(row_count)
    )

    for feature in selected_features:

        if feature in feature_values:

            grid_df[feature] = np.asarray(
                feature_values[feature]
            )

        else:

            grid_df[feature] = float(
                X_train[feature].median()
            )

    return grid_df[selected_features]


# --------------------------------------------------
# 19. 2차원 산점도 및 경계선
# --------------------------------------------------

def draw_2d_plot():

    x_feature = plot_features[0]
    y_feature = plot_features[1]

    x_min, x_max = get_feature_range(x_feature)
    y_min, y_max = get_feature_range(y_feature)

    x_values = np.linspace(x_min, x_max, 120)
    y_values = np.linspace(y_min, y_max, 120)

    xx, yy = np.meshgrid(
        x_values,
        y_values
    )

    grid_values = {
        x_feature: xx.ravel(),
        y_feature: yy.ravel()
    }

    grid_df = make_prediction_data(
        grid_values
    )

    probabilities = {
        "로지스틱 회귀 - 적용 전": (
            logistic_before.predict_proba(grid_df)[:, 1]
        ),
        "로지스틱 회귀 - 적용 후": (
            logistic_after.predict_proba(grid_df)[:, 1]
        ),
        "의사결정트리 - 적용 전": (
            tree_before.predict_proba(grid_df)[:, 1]
        ),
        "의사결정트리 - 적용 후": (
            tree_after.predict_proba(grid_df)[:, 1]
        )
    }

    colors = {
        "로지스틱 회귀 - 적용 전": "#2563EB",
        "로지스틱 회귀 - 적용 후": "#DC2626",
        "의사결정트리 - 적용 전": "#16A34A",
        "의사결정트리 - 적용 후": "#9333EA"
    }

    fig = go.Figure()

    # 실제 정상 데이터
    normal_mask = y_test.to_numpy() == 0

    fig.add_trace(
        go.Scatter(
            x=X_test.loc[normal_mask, x_feature],
            y=X_test.loc[normal_mask, y_feature],
            mode="markers",
            name="실제 정상",
            marker=dict(
                color="#64748B",
                size=7,
                opacity=0.65
            )
        )
    )

    # 실제 뇌졸중 데이터
    stroke_mask = y_test.to_numpy() == 1

    fig.add_trace(
        go.Scatter(
            x=X_test.loc[stroke_mask, x_feature],
            y=X_test.loc[stroke_mask, y_feature],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#F59E0B",
                size=9,
                opacity=0.9,
                symbol="diamond"
            )
        )
    )

    boundary_messages = []

    for name, probability in probabilities.items():

        z = probability.reshape(xx.shape)

        p_min = float(np.min(probability))
        p_max = float(np.max(probability))

        if p_min <= 0.5 <= p_max:

            fig.add_trace(
                go.Contour(
                    x=x_values,
                    y=y_values,
                    z=z,
                    name=name,
                    showscale=False,
                    showlegend=True,
                    contours=dict(
                        start=0.5,
                        end=0.5,
                        size=1,
                        coloring="lines"
                    ),
                    line=dict(
                        color=colors[name],
                        width=3,
                        dash=(
                            "solid"
                            if "적용 전" in name
                            else "dash"
                        )
                    ),
                    hoverinfo="skip"
                )
            )

        else:

            boundary_messages.append(
                f"- **{name}**: 0.5 경계가 현재 그림 범위 "
                f"밖에 있습니다. (확률 범위: "
                f"{p_min:.3f}~{p_max:.3f})"
            )

    fig.update_layout(
        title="실제 데이터와 가중치 적용 전후의 0.5 경계",
        xaxis_title=FEATURE_LABELS[x_feature],
        yaxis_title=FEATURE_LABELS[y_feature],
        height=650,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if boundary_messages:

        st.warning(
            "다음 경계는 현재 그림의 확률 범위 안에서 "
            "찾을 수 없습니다.\n\n"
            + "\n".join(boundary_messages)
        )

    else:

        st.success(
            "네 가지 모델의 0.5 경계가 모두 그림 범위 안에서 확인됩니다."
        )


# --------------------------------------------------
# 20. 3차원 산점도 및 경계면
# --------------------------------------------------

def draw_3d_plot():

    x_feature = plot_features[0]
    y_feature = plot_features[1]
    z_feature = plot_features[2]

    x_min, x_max = get_feature_range(x_feature)
    y_min, y_max = get_feature_range(y_feature)
    z_min, z_max = get_feature_range(z_feature)

    x_values = np.linspace(x_min, x_max, 28)
    y_values = np.linspace(y_min, y_max, 28)
    z_values = np.linspace(z_min, z_max, 28)

    xx, yy, zz = np.meshgrid(
        x_values,
        y_values,
        z_values,
        indexing="ij"
    )

    grid_values = {
        x_feature: xx.ravel(),
        y_feature: yy.ravel(),
        z_feature: zz.ravel()
    }

    # 수정된 make_prediction_data 함수가
    # 28 x 28 x 28개의 격자 데이터를 모두 생성함
    grid_df = make_prediction_data(
        grid_values
    )

    probabilities = {
        "로지스틱 회귀 - 적용 전": (
            logistic_before.predict_proba(grid_df)[:, 1]
        ),
        "로지스틱 회귀 - 적용 후": (
            logistic_after.predict_proba(grid_df)[:, 1]
        ),
        "의사결정트리 - 적용 전": (
            tree_before.predict_proba(grid_df)[:, 1]
        ),
        "의사결정트리 - 적용 후": (
            tree_after.predict_proba(grid_df)[:, 1]
        )
    }

    colors = {
        "로지스틱 회귀 - 적용 전": "#2563EB",
        "로지스틱 회귀 - 적용 후": "#DC2626",
        "의사결정트리 - 적용 전": "#16A34A",
        "의사결정트리 - 적용 후": "#9333EA"
    }

    fig = go.Figure()

    # 실제 정상 데이터
    normal_mask = y_test.to_numpy() == 0

    fig.add_trace(
        go.Scatter3d(
            x=X_test.loc[normal_mask, x_feature],
            y=X_test.loc[normal_mask, y_feature],
            z=X_test.loc[normal_mask, z_feature],
            mode="markers",
            name="실제 정상",
            marker=dict(
                color="#64748B",
                size=4,
                opacity=0.55
            )
        )
    )

    # 실제 뇌졸중 데이터
    stroke_mask = y_test.to_numpy() == 1

    fig.add_trace(
        go.Scatter3d(
            x=X_test.loc[stroke_mask, x_feature],
            y=X_test.loc[stroke_mask, y_feature],
            z=X_test.loc[stroke_mask, z_feature],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#F59E0B",
                size=5,
                opacity=0.9,
                symbol="diamond"
            )
        )
    )

    boundary_messages = []

    # 각 모델의 0.5 경계면
    for name, probability in probabilities.items():

        p_min = float(np.min(probability))
        p_max = float(np.max(probability))

        if p_min <= 0.5 <= p_max:

            fig.add_trace(
                go.Isosurface(
                    x=xx.ravel(),
                    y=yy.ravel(),
                    z=zz.ravel(),
                    value=probability,
                    isomin=0.5,
                    isomax=0.5,
                    surface_count=1,
                    opacity=0.30,
                    colorscale=[
                        [0, colors[name]],
                        [1, colors[name]]
                    ],
                    showscale=False,
                    caps=dict(
                        x_show=False,
                        y_show=False,
                        z_show=False
                    ),
                    name=name,
                    showlegend=True,
                    hoverinfo="skip"
                )
            )

        else:

            boundary_messages.append(
                f"- **{name}**: 0.5 경계면이 현재 그림 범위 "
                f"밖에 있습니다. (확률 범위: "
                f"{p_min:.3f}~{p_max:.3f})"
            )

    fig.update_layout(
        title="3차원 실제 데이터와 가중치 적용 전후의 0.5 경계면",
        scene=dict(
            xaxis_title=FEATURE_LABELS[x_feature],
            yaxis_title=FEATURE_LABELS[y_feature],
            zaxis_title=FEATURE_LABELS[z_feature]
        ),
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if boundary_messages:

        st.warning(
            "다음 경계면은 현재 그림의 확률 범위 안에서 "
            "찾을 수 없습니다.\n\n"
            + "\n".join(boundary_messages)
        )

    else:

        st.success(
            "네 가지 모델의 0.5 경계면이 모두 그림 범위 안에서 확인됩니다."
        )


# --------------------------------------------------
# 21. 선택 속성 수에 따라 그래프 표시
# --------------------------------------------------

if len(selected_features) == 2:

    st.info(
        "선택한 속성이 2개이므로 2차원 산점도와 "
        "0.5 경계선을 표시합니다."
    )

    draw_2d_plot()

else:

    st.info(
        "선택한 속성이 3개 이상이므로 처음 선택한 3개 속성을 "
        "축으로 사용하는 3차원 산점도와 경계면을 표시합니다. "
        "나머지 속성은 훈련 데이터의 중앙값으로 고정합니다."
    )

    draw_3d_plot()


# --------------------------------------------------
# 22. 마무리 안내
# --------------------------------------------------

st.divider()

st.subheader("결과 해석 시 참고할 점")

st.markdown("""
- 클래스 가중치를 적용하면 적은 수의 뇌졸중 환자를 더 중요하게 고려하도록 학습할 수 있습니다.
- 가중치를 적용한 뒤 재현율이 높아지면 실제 뇌졸중 환자를 더 많이 찾아냈다는 의미입니다.
- 반면 정상인을 뇌졸중이라고 잘못 예측하는 거짓 양성(FP)이 증가할 수도 있습니다.
- 정확도만으로 모델의 성능을 판단하기보다 재현율, 정밀도, F1 점수도 함께 살펴보는 것이 좋습니다.
- 가중치를 적용했다고 해서 모든 평가 지표가 반드시 좋아지는 것은 아닙니다.
""")

st.warning(
    "이 프로그램은 분류 모델의 원리와 성능을 학습하기 위한 실습 자료입니다. "
    "실제 뇌졸중 진단이나 의료적 판단에 사용할 수 없습니다."
)
