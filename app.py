import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 설정 (제목, 아이콘, 브라우저 탭)
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

st.title("🎬 영화 유형 나누기 (k-Means 군집화)")

# 데이터 불러오기 함수 (캐싱)
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df_raw = load_data()

# 2. 전처리 및 속성 생성
df_clean = df_raw.copy()
df_clean = df_clean.dropna(
    subset=["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
)
df_clean = df_clean[df_clean["first_week_audi"] > 0].copy()

# 파생 변수 생성
df_clean["log_first_scrn"] = np.log10(df_clean["first_scrn"])
df_clean["log_total_audi"] = np.log10(df_clean["total_audi"])
df_clean["long_run_index"] = df_clean["total_audi"] / df_clean["first_week_audi"]
df_clean["long_run_index"] = df_clean["long_run_index"].clip(upper=20)

# 표시용 속성 명칭 매핑
feature_options = {
    "log_first_scrn": "스크린 수 (로그)",
    "log_total_audi": "누적 관객수 (로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수",
}

# 3. 화면 UI: 속성 및 묶음 수 선택
col_feat, col_k = st.columns([2, 1])

with col_feat:
    selected_feature_keys = st.multiselect(
        "군집화에 사용할 속성을 선택하세요 (최소 2개 이상):",
        options=list(feature_options.keys()),
        default=list(feature_options.keys()),
        format_func=lambda x: feature_options[x],
    )

with col_k:
    n_clusters = st.number_input(
        "묶음 수 선택 (2 ~ 7):", min_value=2, max_value=7, value=3, step=1
    )

st.write(
    f"전체 영화 편수: **{len(df_raw)}**편 | 군집화에 사용된 영화 편수: **{len(df_clean)}**편"
)

if len(selected_feature_keys) < 2:
    st.warning("군집화를 진행하려면 속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

# 4. 선택된 속성 데이터 표준화
X = df_clean[selected_feature_keys].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. K-Means 군집화 (선택된 k값 적용)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_clean["cluster_temp"] = clusters

# 6. 묶음 번호 재정의 (누적 관객 평균이 큰 순서대로 ㉮~㉴ 부여)
avg_audi_per_cluster = (
    df_clean.groupby("cluster_temp")["total_audi"].mean().sort_values(ascending=False)
)
cluster_mapping = {}
labels_pool = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
active_labels = labels_pool[:n_clusters]

for idx, temp_cluster in enumerate(avg_audi_per_cluster.index):
    cluster_mapping[temp_cluster] = active_labels[idx]

df_clean["cluster"] = df_clean["cluster_temp"].map(cluster_mapping)

# 실루엣 점수 계산
sil_score = silhouette_score(X_scaled, clusters)
st.info(
    f"💡 현재 설정된 묶음 수(**{n_clusters}개**)의 **실루엣 점수**: **{sil_score:.4f}** (점수는 -1에서 1 사이이며, 1에 가까울수록 묶음이 뚜렷하게 잘 나뉘었음을 의미합니다.)"
)

st.markdown("---")

# 7. 2차원 산점도
st.subheader("📌 2차원 산점도")
col1, col2 = st.columns(2
