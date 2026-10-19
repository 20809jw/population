import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page Config (제목, 아이콘, 탭 타이틀)
st.set_page_config(
    page_title="영화 유형 나누기", page_icon="🎬", layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means Clustering)")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df_raw = load_data()
total_count = len(df_raw)

# 정제: 필수 속성 결측치 제거 및 first_week_audi > 0 조건
required_cols = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
df = df_raw.dropna(subset=required_cols).copy()
df = df[df["first_week_audi"] > 0].copy()

# 파생 변수 계산
# 1) 스크린 수 상용로그
df["log_first_scrn"] = np.log10(df["first_scrn"])
# 2) 누적 관객 상용로그
df["log_total_audi"] = np.log10(df["total_audi"])
# 3) 롱런 지수 (누적 관객 / 첫 주 관객, 최대 20)
df["long_run_index"] = np.clip(df["total_audi"] / df["first_week_audi"], a_min=0, a_max=20)
# 4) 10위권 일수 (기존 컬럼 사용)

valid_count = len(df)

# 편수 안내 메시지
st.info(f"전체 영화 편수: **{total_count:,}**편 | 분석 대상(유효 데이터) 편수: **{valid_count:,}**편")

# 속성 매핑 (화면 표시용 이름 <-> 실제 칼럼명)
feature_map = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객수 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

# 원래 단위 평균 계산용 원본 속성 매핑
raw_feature_map = {
    "스크린 수 (로그)": "first_scrn",
    "누적 관객수 (로그)": "total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

# 2. 속성 선택
st.subheader("1. 군집화에 사용할 속성 선택")
selected_features_labels = st.multise
