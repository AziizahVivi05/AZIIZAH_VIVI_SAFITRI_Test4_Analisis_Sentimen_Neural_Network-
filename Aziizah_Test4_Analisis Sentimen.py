import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dashboard Sentimen", layout="wide")

st.title("Dashboard Analisis Sentimen menggunakan Neural Network")
st.write("Topik: Kualitas Udara & Karhutla")

# --- Langkah 1 & 2: Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(r'Kualitas_Udara___28-10-2026_18-24-04.csv')
    return df

df = load_data()

# --- Langkah 3: Pembersihan Teks ---
def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\@\w+|\#', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

df['clean_text'] = df['full_text'].apply(clean_text)

# --- Langkah 4: Pelabelan Sentimen ---
kata_positif = ['aman', 'lega', 'selamat', 'bantuan', 'pulih', 'baik', 'bersih', 'sehat', 'segar', 'cerah', 'bagus', 'terkendali', 'normal', 'penanganan', 'selesai', 'bebas', 'teguh']
kata_negatif = ['banjir', 'parah', 'rusak', 'macet', 'bahaya', 'asap', 'kebakaran', 'karhutla', 'buruk', 'anjlok', 'kabut', 'terbatas', 'ispa', 'mencemari', 'mengancam', 'kuning', 'merah', 'sesak', 'sakit', 'absen']

def get_sentiment(text):
    words = text.split()
    pos_count = sum(1 for word in words if word in kata_positif)
    neg_count = sum(1 for word in words if word in kata_negatif)
    if pos_count > neg_count: return 'Positif'
    elif neg_count > pos_count: return 'Negatif'
    else: return 'Netral'

df['sentimen'] = df['clean_text'].apply(get_sentiment)

st.header("1. Data Hasil Preprocessing")
st.dataframe(df[['created_at', 'username', 'full_text', 'clean_text', 'sentimen']].head(10))

# --- Langkah 5: EDA & Visualisasi ---
st.header("2. Exploratory Data Analysis (EDA)")

tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Sentimen", "Panjang Tweet", "Top Kata", "Tren Waktu"])

with tab1:
    col1, col2 = st.columns(2)
    sentiment_counts = df['sentimen'].value_counts()
    with col1:
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette=['#ff9999','#66b3ff','#99ff99'])
        ax.set_title("Bar Chart Sentimen")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
        ax.set_title("Pie Chart Sentimen")
        st.pyplot(fig)

with tab2:
    df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x='word_count', hue='sentimen', kde=True, bins=30, palette={'Positif':'#2ecc71', 'Netral':'#3498db', 'Negatif':'#e74c3c'}, ax=ax)
    ax.set_title("Distribusi Jumlah Kata per Sentimen")
    st.pyplot(fig)

with tab3:
    all_words = ' '.join(df['clean_text']).split()
    word_freq = Counter(all_words).most_common(15)
    words, counts = zip(*word_freq)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=list(counts), y=list(words), palette='viridis', ax=ax)
    ax.set_title("Top 15 Kata Paling Sering Muncul")
    st.pyplot(fig)

with tab4:
    try:
        df['tanggal'] = pd.to_datetime(df['created_at']).dt.date
        tren_waktu = df.groupby(['tanggal', 'sentimen']).size().reset_index(name='jumlah')
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=tren_waktu, x='tanggal', y='jumlah', hue='sentimen', palette={'Positif':'#2ecc71', 'Netral':'#3498db', 'Negatif':'#e74c3c'}, marker='o', linewidth=2, ax=ax)
        ax.set_title("Tren Jumlah Tweet per Hari")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Gagal memuat tren waktu: {e}")

# --- Langkah 6: WordCloud ---
st.header("3. WordCloud per Sentimen")
col_wc1, col_wc2, col_wc3 = st.columns(3)

def generate_wordcloud(sentiment_label, colormap):
    text_data = " ".join(df[df['sentimen'] == sentiment_label]['clean_text'])
    if text_data.strip():
        wc = WordCloud(width=400, height=400, background_color='white', colormap=colormap).generate(text_data)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"WordCloud - {sentiment_label}")
        return fig
    return None

with col_wc1:
    fig_pos = generate_wordcloud('Positif', 'Greens')
    if fig_pos: st.pyplot(fig_pos)
with col_wc2:
    fig_net = generate_wordcloud('Netral', 'Blues')
    if fig_net: st.pyplot(fig_net)
with col_wc3:
    fig_neg = generate_wordcloud('Negatif', 'Reds')
    if fig_neg: st.pyplot(fig_neg)

# --- Langkah 7 & 8: Model Neural Network & Evaluasi ---
st.header("4. Model Neural Network & Evaluasi (Ternormalisasi)")

X = df['clean_text']
y = df['sentimen']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Normalisasi MaxAbsScaler
scaler = MaxAbsScaler()
X_train_norm = scaler.fit_transform(X_train_vec)
X_test_norm = scaler.transform(X_test_vec)

@st.cache_resource
def train_model(_X_train, _y_train):
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(_X_train, _y_train)
    return model

model = train_model(X_train_norm, y_train)
y_pred = model.predict(X_test_norm)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

col_eval1, col_eval2 = st.columns(2)
with col_eval1:
    st.subheader("Metrik Performa Model")
    metrics_df = pd.DataFrame({
        "Metrik": ["Akurasi", "Presisi", "Recall", "F1-Score"],
        "Nilai": [f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"]
    })
    st.table(metrics_df)

with col_eval2:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    ax_cm.set_xlabel('Prediksi')
    ax_cm.set_ylabel('Aktual')
    st.pyplot(fig_cm)

# --- Langkah 9: Prediksi Teks Baru ---
st.header("5. Prediksi Sentimen Teks Baru")
input_text = st.text_area("Masukkan teks tweet untuk diprediksi:")
if st.button("Prediksi"):
    if input_text.strip() != "":
        clean_input = clean_text(input_text)
        input_vec = vectorizer.transform([clean_input])
        # Gunakan scaler untuk normalisasi teks baru
        input_norm = scaler.transform(input_vec) 
        
        pred_label = model.predict(input_norm)[0]
        prob = model.predict_proba(input_norm)
        
        st.write(f"**Teks Bersih:** {clean_input}")
        
        emoji_dict = {"Positif": "✅ Positif", "Netral": "⚪ Netral", "Negatif": "❌ Negatif"}
        st.subheader(f"Hasil Prediksi: {emoji_dict[pred_label]}")
        
        st.write("**Probabilitas:**")
        prob_df = pd.DataFrame(prob, columns=model.classes_)
        st.dataframe(prob_df)
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")