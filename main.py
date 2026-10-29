import json
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "k-suhyeon03/data/refs/heads/main/stroke.csv"
)

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "뇌졸중_데이터_기록.json"


# -----------------------------
# 앱 제목
# -----------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고 각 열의 의미를 직접 정리해 보는 공간입니다.")


# -----------------------------
# 기록 파일 불러오기
# -----------------------------
def load_notes():
    if RECORD_FILE.exists():
        try:
            with RECORD_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                return {
                    "meanings": data.get("meanings", {}),
                    "source": data.get("source", ""),
                }

        except (json.JSONDecodeError, OSError):
            st.warning(
                "기존 기록 파일을 읽지 못해 "
                "빈 기록으로 시작합니다."
            )

    return {
        "meanings": {},
        "source": "",
    }


# -----------------------------
# 기록 파일 저장하기
# -----------------------------
def save_notes(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source": source,
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


if "saved_notes" not in st.session_state:
    st.session_state.saved_notes = load_notes()


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


# 필요한 열 확인
required_columns = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "CSV에서 예상한 열을 찾지 못했습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# -----------------------------
# 1. 데이터 소개
# -----------------------------
st.subheader("1. 데이터 소개")

st.write(
    "이 데이터는 사람별 특성과 건강 관련 정보를 "
    "열로 정리한 CSV 자료입니다. 각 행은 한 사람의 기록이며, "
    "`stroke` 열은 뇌졸중 발생 여부를 나타내는 값으로 "
    "사용됩니다. 아래에서 데이터의 크기와 열 구성을 "
    "확인해 보세요."
)


# -----------------------------
# 2. 숫자 카드 4개
# -----------------------------
total_people = len(df)
column_count = len(df.columns)

stroke_values = pd.to_numeric(
    df["stroke"],
    errors="coerce",
)

stroke_count = int((stroke_values == 1).sum())

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)

card1, card2, card3, card4 = st.columns(4)

card1.metric(
    "전체 사람 수",
    f"{total_people:,}명",
)

card2.metric(
    "열 개수",
    f"{column_count:,}개",
)

card3.metric(
    "stroke가 1인 사람 수",
    f"{stroke_count:,}명",
)

card4.metric(
    "stroke가 1인 비율",
    f"{stroke_ratio:.2f}%",
)

st.caption(
    "※ 비율은 stroke가 1인 사람 수를 전체 사람 수로 "
    "나누어 계산했습니다."
)


# -----------------------------
# 3. 열 이름과 데이터 특성
# -----------------------------
st.subheader("2. 열 이름과 데이터 특성")

st.write(
    "‘우리말 뜻’ 칸에 교재를 참고해 직접 입력한 다음, "
    "아래 저장 버튼을 눌러 주세요."
)

meanings = st.session_state.saved_notes.get(
    "meanings", {}
)

rows = []

for column in df.columns:
    non_null = df[column].dropna()
    unique_count = non_null.nunique()

    if pd.api.types.is_numeric_dtype(df[column]):
        kind = (
            f"숫자형 "
            f"(서로 다른 값 {unique_count:,}종)"
        )
    else:
        kind = (
            f"문자형 "
            f"(서로 다른 값 {unique_count:,}종)"
        )

    rows.append({
        "열 이름": column,
        "우리말 뜻": meanings.get(column, ""),
        "값의 종류": kind,
        "빈 값 개수": int(df[column].isna().sum()),
    })


info_df = pd.DataFrame(rows)

edited_info = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수",
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 열의 뜻을 직접 입력하세요.",
            width="large",
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
            width="large",
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d",
        ),
    },
    key="column_info_editor",
)


# -----------------------------
# 4. 데이터 처음 다섯 줄
# -----------------------------
st.subheader("3. 데이터 처음 다섯 줄")

st.write(
    "아래는 원본 CSV에서 처음 다섯 행을 표시한 것입니다. "
    "빈 값은 빈칸으로 나타납니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# 5. 데이터 출처 입력
# -----------------------------
st.subheader("4. 데이터 출처")

st.write(
    "교재에 적힌 데이터 출처를 입력하세요. "
    "입력한 내용은 우리말 뜻과 함께 JSON 파일로 저장됩니다."
)

source_value = st.text_area(
    "데이터 출처",
    value=st.session_state.saved_notes.get(
        "source", ""
    ),
    placeholder="교재에 적힌 데이터 출처를 입력하세요.",
    height=100,
    key="source_editor",
)


# -----------------------------
# 6. 저장 버튼
# -----------------------------
if st.button(
    "💾 우리말 뜻과 출처 저장",
    type="primary",
    use_container_width=True,
):
    meaning_dict = {}

    for _, row in edited_info.iterrows():
        meaning = row["우리말 뜻"]

        if pd.isna(meaning):
            meaning = ""
        else:
            meaning = str(meaning)

        meaning_dict[str(row["열 이름"])] = meaning

    try:
        save_notes(
            meaning_dict,
            source_value,
        )

        st.session_state.saved_notes = {
            "meanings": meaning_dict,
            "source": source_value,
        }

        st.success(
            f"저장했습니다: {RECORD_FILE.as_posix()}"
        )

    except OSError as error:
        st.error(
            "기록을 저장하지 못했습니다. "
            "실행 환경의 폴더 쓰기 권한을 확인해 주세요."
        )
        st.code(str(error))


# -----------------------------
# 7. 저장 파일 안내
# -----------------------------
with st.expander("저장 파일 안내"):
    st.write(
        f"저장 위치: `{RECORD_FILE.as_posix()}`"
    )

    st.write(
        "새로고침하면 JSON 파일에서 기록을 다시 불러옵니다. "
        "단, Streamlit Cloud에서는 앱 재배포나 재시작 후 "
        "파일이 유지되지 않을 수 있으므로, "
        "필요하면 JSON 파일을 별도로 보관하세요."
    )


# -----------------------------
# 하단 안내
# -----------------------------
st.divider()

st.caption(
    "학습용 데이터 탐색 화면입니다. "
    "이 앱은 실제 뇌졸중 위험을 진단하거나 예측하지 않습니다."
)
