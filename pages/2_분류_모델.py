
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="분류 모델",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 분류 모델")
st.write(
    "나이, 평균 혈당, 체질량지수, 고혈압, 심장병 데이터를 활용하여 "
    "뇌졸중 여부를 예측하는 분류 모델을 만들어 봅니다."
)

st.info(
    "뇌졸중 여부(stroke)가 1이면 뇌졸중(양성), "
    "0이면 뇌졸중 아님(음성)으로 분류합니다."
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
# 3. 사용할 속성 선택
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

selected_feature_names = {
    feature_labels[label]: label
    for label in selected_labels
}

st.write(
    "선택한 속성: " + ", ".join(selected_labels)
)


# =========================================================
# 4. 데이터 정리 및 학습용·테스트용 분리
# =========================================================
st.header("2. 학습용 데이터와 테스트용 데이터 분리")

required_columns = [
    "id",
    "stroke"
] + selected_features

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

# 필요한 열만 사용
data = df[required_columns].copy()

# 번호와 정답을 숫자형으로 변환
data["id"] = pd.to_numeric(data["id"], errors="coerce")
data["stroke"] = pd.to_numeric(data["stroke"], errors="coerce")

# 번호 또는 정답이 없는 행 제거
data = data.dropna(subset=["id", "stroke"])

# 정답이 0 또는 1인 데이터만 사용
data = data[data["stroke"].isin([0, 1])].copy()

# 번호 순서대로 정렬
data = data.sort_values("id").reset_index(drop=True)

# 입력 속성을 숫자형으로 변환
for col in selected_features:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 10명씩 묶었을 때 각 묶음의 앞 3명을 테스트용으로 지정
test_mask = np.arange(len(data)) % 10 < 3

train_df = data.loc[~test_mask].copy().reset_index(drop=True)
test_df = data.loc[test_mask].copy().reset_index(drop=True)

# 입력 데이터와 정답 데이터 분리
X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

st.write(f"- 전체 데이터: **{len(data):,}명**")
st.write(f"- 학습용 데이터: **{len(train_df):,}명**")
st.write(f"- 테스트용 데이터: **{len(test_df):,}명**")

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

    # 학습용 중앙값만 사용하여 학습용·테스트용 결측치 처리
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.write(
        f"체질량지수(bmi)의 결측값은 학습용 데이터의 중앙값 "
        f"**{bmi_median:.2f}**으로 채웠습니다."
    )
else:
    st.write("체질량지수를 선택하지 않아 BMI 결측치 처리는 생략했습니다.")

# 다른 선택 속성에 결측값이 있는지 확인
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 속성에 BMI 이외의 결측값이 있습니다. "
        "원본 데이터를 확인해 주세요."
    )
    st.stop()


# =========================================================
# 6. 모델 만들기
# =========================================================
st.header("4. 분류 모델 만들기")

# 로지스틱 회귀: 표준화는 학습 데이터에만 적합
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

# 학습용 데이터만 사용하여 학습
logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

st.success("두 분류 모델의 학습이 완료되었습니다.")


# =========================================================
# 7. 기준 모델 및 정확도 계산
# =========================================================
st.header("5. 모델 정확도 비교")

# 로지스틱 회귀
logistic_train_pred = logistic_model.predict(X_train)
logistic_test_pred = logistic_model.predict(X_test)

logistic_train_acc = accuracy_score(
    y_train, logistic_train_pred
)

logistic_test_acc = accuracy_score(
    y_test, logistic_test_pred
)

# 의사결정트리
tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

tree_train_acc = accuracy_score(
    y_train, tree_train_pred
)

tree_test_acc = accuracy_score(
    y_test, tree_test_pred
)

# 기준 모델: 훈련용 데이터에서 더 많은 정답을 항상 예측
majority_class = int(y_train.value_counts().idxmax())

baseline_train_pred = np.full(
    len(y_train), majority_class
)

baseline_test_pred = np.full(
    len(y_test), majority_class
)

baseline_train_acc = accuracy_score(
    y_train, baseline_train_pred
)

baseline_test_acc = accuracy_score(
    y_test, baseline_test_pred
)

baseline_name = (
    "뇌졸중"
    if majority_class == 1
    else "뇌졸중 아님"
)

st.caption(
    f"기준 모델은 입력 속성을 전혀 사용하지 않고 "
    f"훈련용 데이터에서 더 많은 쪽인 '{baseline_name}'만 예측합니다."
)


def accuracy_card(title, train_acc, test_acc):
    with st.container(border=True):
        st.markdown(f"**{title}**")

        st.metric(
            label="테스트 데이터 정확도",
            value=f"{test_acc * 100:.2f}%"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.caption("훈련 데이터 정확도")
            st.write(f"**{train_acc * 100:.2f}%**")

        with col2:
            st.caption("테스트 데이터 정확도")
            st.write(f"**{test_acc * 100:.2f}%**")


col1, col2, col3 = st.columns(3)

with col1:
    accuracy_card(
        "로지스틱 회귀 (확률로 답하는 모델)",
        logistic_train_acc,
        logistic_test_acc
    )

with col2:
    accuracy_card(
        "의사결정트리 (질문으로 답하는 모델)",
        tree_train_acc,
        tree_test_acc
    )

with col3:
    accuracy_card(
        "입력을 보지 않는 기준 모델",
        baseline_train_acc,
        baseline_test_acc
    )


# =========================================================
# 8. 산점도 축 선택
# =========================================================
st.header("6. 두 속성으로 분류 결과 시각화")

st.write(
    "선택한 속성 중 두 개를 골라 가로축과 세로축으로 사용합니다."
)

default_x = selected_labels[0]
default_y = selected_labels[1]

col1, col2 = st.columns(2)

with col1:
    x_label = st.selectbox(
        "가로축 속성",
        options=selected_labels,
        index=0
    )

with col2:
    y_label = st.selectbox(
        "세로축 속성",
        options=selected_labels,
        index=1
    )

if x_label == y_label:
    st.warning("가로축과 세로축은 서로 다른 속성을 선택해야 합니다.")
    st.stop()

x_feature = feature_labels[x_label]
y_feature = feature_labels[y_label]


# =========================================================
# 9. 결정 영역 계산
# =========================================================
# 테스트 데이터의 중앙값으로 나머지 속성을 고정
fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(X_test[feature].median())

st.subheader("그림에서 고정한 나머지 속성")

if fixed_values:
    for feature, value in fixed_values.items():
        st.write(
            f"- {selected_feature_names[feature]}: "
            f"테스트 데이터 중앙값 **{value:.2f}**"
        )
else:
    st.write("선택한 속성이 두 개뿐이므로 고정한 속성이 없습니다.")

# 테스트 데이터 범위에 맞춰 격자 생성
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

# 값이 모두 같을 경우에도 그림이 그려지도록 범위 보정
if x_min == x_max:
    x_min -= 1
    x_max += 1

if y_min == y_max:
    y_min -= 1
    y_max += 1

x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

x_values = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    100
)

y_values = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    100
)

xx, yy = np.meshgrid(x_values, y_values)

# 선택한 속성은 격자값으로, 나머지는 테스트 중앙값으로 설정
grid_data = pd.DataFrame(
    np.zeros((xx.size, len(selected_features))),
    columns=selected_features
)

for feature in selected_features:
    if feature == x_feature:
        grid_data[feature] = xx.ravel()
    elif feature == y_feature:
        grid_data[feature] = yy.ravel()
    else:
        grid_data[feature] = fixed_values[feature]

# 로지스틱 회귀의 양성 확률
grid_probability = logistic_model.predict_proba(
    grid_data
)[:, list(logistic_model.classes_).index(1)]

probability_grid = grid_probability.reshape(xx.shape)

# 의사결정트리의 예측 결과
grid_tree_pred = tree_model.predict(
    grid_data
)

tree_grid = grid_tree_pred.reshape(xx.shape)


# =========================================================
# 10. 산점도 그리기
# =========================================================
st.subheader("테스트 데이터의 실제 결과와 분류 경계")

fig = go.Figure()

# 의사결정트리의 영역을 옅은 색으로 표시
fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=tree_grid,
        colorscale=[
            [0, "rgba(255, 190, 190, 0.35)"],
            [1, "rgba(170, 205, 255, 0.35)"]
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역"
    )
)

# 로지스틱 회귀의 확률 0.5 경계선
fig.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=probability_grid,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="none"
        ),
        line=dict(
            color="black",
            width=3
        ),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계선"
    )
)

# 실제 뇌졸중 여부에 따라 테스트 데이터 표시
test_plot = X_test.copy()
test_plot["stroke"] = y_test.to_numpy()

negative = test_plot[test_plot["stroke"] == 0]
positive = test_plot[test_plot["stroke"] == 1]

fig.add_trace(
    go.Scatter(
        x=negative[x_feature],
        y=negative[y_feature],
        mode="markers",
        name="실제 뇌졸중 아님 (0)",
        marker=dict(
            color="#E57373",
            size=7,
            opacity=0.7
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=positive[x_feature],
        y=positive[y_feature],
        mode="markers",
        name="실제 뇌졸중 (1)",
        marker=dict(
            color="#1976D2",
            size=7,
            opacity=0.7
        )
    )
)

fig.update_layout(
    title="테스트 데이터와 두 모델의 분류 영역",
    xaxis_title=x_label,
    yaxis_title=y_label,
    height=650,
    legend_title="실제 뇌졸중 여부"
)

st.plotly_chart(fig, use_container_width=True)

# 로지스틱 회귀의 0.5 경계선이 그림 안에 있는지 확인
boundary_exists = (
    np.nanmin(probability_grid) <= 0.5
    and np.nanmax(probability_grid) >= 0.5
)

if boundary_exists:
    st.write(
        "검은색 선은 로지스틱 회귀 모델이 뇌졸중 확률을 "
        "0.5로 판단하는 경계선입니다."
    )
else:
    st.write(
        "로지스틱 회귀의 0.5 경계선은 현재 그림의 범위 밖에 있습니다."
    )

st.caption(
    "의사결정트리의 옅은 배경색은 각 영역에서 예측하는 결과를 나타냅니다. "
    "점의 색은 테스트 데이터의 실제 뇌졸중 여부를 나타냅니다."
)


# =========================================================
# 11. 의사결정트리 가지 그림
# =========================================================
st.header("7. 의사결정트리의 질문 구조")

st.write(
    "의사결정트리는 속성에 관한 질문을 최대 세 번까지 하며, "
    "각 마지막 마디에서 뇌졸중 여부를 예측합니다."
)

tree_structure = tree_model.tree_
tree_classes = tree_model.classes_

# 각 노드에 도달한 훈련 데이터의 실제 인원과 양성 인원 계산
decision_path = tree_model.decision_path(X_train)

node_statistics = {}

for node_id in range(tree_structure.node_count):
    reached = (
        decision_path[:, node_id]
        .toarray()
        .ravel()
        .astype(bool)
    )

    node_y = y_train.to_numpy()[reached]

    total_count = int(len(node_y))
    positive_count = int(np.sum(node_y == 1))

    positive_ratio = (
        positive_count / total_count * 100
        if total_count > 0 else 0
    )

    node_statistics[node_id] = {
        "total": total_count,
        "positive": positive_count,
        "ratio": positive_ratio
    }


# DOT 문자열 생성
dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, nodesep=0.45, ranksep=0.65];',
    'node [shape=box, style="filled,rounded", fontname="Malgun Gothic", fontsize=11];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]

leaf_count = 0
negative_leaf_count = 0
used_features = set()


def build_tree_dot(node_id):
    global leaf_count, negative_leaf_count

    stats = node_statistics[node_id]

    total = stats["total"]
    positive = stats["positive"]
    ratio = stats["ratio"]

    left_child = tree_structure.children_left[node_id]
    right_child = tree_structure.children_right[node_id]

    is_leaf = left_child == right_child

    if is_leaf:
        leaf_count += 1

        class_index = int(
            np.argmax(tree_structure.value[node_id][0])
        )
        predicted_class = int(tree_classes[class_index])

        if predicted_class == 0:
            negative_leaf_count += 1
            answer = "뇌졸중 아님"
            fill_color = "#F8CACA"
        else:
            answer = "뇌졸중"
            fill_color = "#C8DDF8"

        label = (
            f"답: {answer}\\n"
            f"전체 {total}명\\n"
            f"뇌졸중 {positive}명 ({ratio:.1f}%)"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
        )

    else:
        feature_index = tree_structure.feature[node_id]
        feature = selected_features[feature_index]
        feature_name = selected_feature_names[feature]

        used_features.add(feature)

        threshold = tree_structure.threshold[node_id]

        label = (
            f"{feature_name} ≤ {threshold:.2f}?\\n"
            f"전체 {total}명\\n"
            f"뇌졸중 {positive}명 ({ratio:.1f}%)"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="#EEEEEE"];'
        )

        # 왼쪽 가지: 조건을 만족하는 경우
        dot_lines.append(
            f'{node_id} -> {left_child} [label="예"];'
        )

        # 오른쪽 가지: 조건을 만족하지 않는 경우
        dot_lines.append(
            f'{node_id} -> {right_child} [label="아니요"];'
        )

        build_tree_dot(left_child)
        build_tree_dot(right_child)


build_tree_dot(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# =========================================================
# 12. 의사결정트리 요약
# =========================================================
st.subheader("의사결정트리 요약")

used_feature_names = [
    selected_feature_names[feature]
    for feature in selected_features
    if feature in used_features
]

if used_feature_names:
    used_text = ", ".join(used_feature_names)
else:
    used_text = "없음 (질문 없이 바로 예측)"

st.write(
    f"- 답을 내리는 마디는 모두 **{leaf_count}개**이며, "
    f"그중 **{negative_leaf_count}개**가 '뇌졸중 아님'이라고 답합니다."
)

st.write(
    f"- 의사결정트리가 실제로 질문에 사용한 속성: **{used_text}**"
)

st.caption(
    "마디의 인원수와 뇌졸중 비율은 해당 마디까지 도달한 "
    "훈련용 데이터만을 기준으로 계산했습니다."
)

st.divider()

st.caption(
    "※ 이 실습은 분류 알고리즘의 작동 원리를 이해하기 위한 교육용입니다. "
    "실제 의료 진단이나 뇌졸중 위험 판단에 사용할 수 없습니다."
)
