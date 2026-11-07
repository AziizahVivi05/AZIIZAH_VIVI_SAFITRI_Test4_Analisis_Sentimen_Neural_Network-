import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Dashboard Clustering Gempa", layout="wide")
st.title("🌍 Dashboard Analisis Clustering K-Means Gempa Bumi")

@st.cache_data
def load_data():
    return pd.read_csv('global_earthquakes_seismic.csv')

try:
    df = load_data()
    st.subheader("Cuplikan Dataset")
    st.dataframe(df.head())

    # 1. Preprocessing
    num_cols = ['latitude', 'longitude', 'depth', 'mag', 'sig', 'rms', 'gap']
    df_num = df[num_cols].dropna().drop_duplicates()
    
    Q1 = df_num.quantile(0.25)
    Q3 = df_num.quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_num[~((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).any(axis=1)].copy()

    # 2. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # Sidebar untuk parameter interaktif
    st.sidebar.header("Pengaturan K-Means")
    k_clusters = st.sidebar.slider("Pilih Jumlah Cluster (K)", min_value=2, max_value=7, value=3)

    # 3. K-Means Clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df_clean['cluster'] = kmeans.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, df_clean['cluster'])
    
    # 4. Menampilkan Metrik
    st.subheader("Metrik Evaluasi")
    st.metric(label="Silhouette Score", value=f"{sil_score:.4f}")
    
    st.subheader("Rata-rata Karakteristik per Cluster")
    st.dataframe(df_clean.groupby('cluster').mean())

    # 5. Visualisasi
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2D Scatter Plot (Longitude vs Latitude)")
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='longitude', y='latitude', hue='cluster', data=df_clean, palette='viridis', s=15, alpha=0.8, ax=ax_scatter)
        st.pyplot(fig_scatter)

    with col2:
        st.subheader("Correlation Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_clean.drop('cluster', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heat)
        st.pyplot(fig_heat)

    st.subheader("Distribusi Fitur (Boxplots)")
    fig_box, axes_box = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(x='cluster', y='mag', data=df_clean, palette='Set2', ax=axes_box[0])
    axes_box[0].set_title('Distribusi Magnitudo (mag)')
    sns.boxplot(x='cluster', y='depth', data=df_clean, palette='Set2', ax=axes_box[1])
    axes_box[1].set_title('Distribusi Kedalaman (depth)')
    st.pyplot(fig_box)

except FileNotFoundError:
    st.error("File 'global_earthquakes_seismic.csv' tidak ditemukan. Pastikan file berada di folder yang sama dengan skrip ini.")