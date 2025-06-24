"""
Sentix: Classificador de Sentimentos usando NLP.
"""
import re
import pandas as pd
import nltk
import logging
import argparse
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import lime.lime_text
from sklearn.pipeline import make_pipeline

import shap

# Baixar recursos do NLTK
nltk.download("stopwords")
nltk.download("rslp")

# Configurar logs
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# === 🔄 Leitura e preparação do dataset ===
csv_path = "dataset/data.csv"
df_raw = pd.read_csv(csv_path, encoding="ISO-8859-1")

# Filtrando dados que têm descrição válida
df_raw = df_raw[df_raw["Description"].notna() & (df_raw["Description"].str.strip() != "")]

# Criando rótulos baseados na coluna Quantity (exemplo simplificado)
def rotular_sentimento(qtd):
    if qtd > 10:
        return "positivo"
    elif qtd < 0:
        return "negativo"
    else:
        return "neutro"

df_raw = df_raw.copy()
df_raw["sentiment"] = df_raw["Quantity"].apply(rotular_sentimento)
df_raw = df_raw[["Description", "sentiment"]].rename(columns={"Description": "text"})

# === 🔧 Pré-processamento ===
stop_words = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()

def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = text.split()
    filtered = [t for t in tokens if t not in stop_words]
    stemmed = [stemmer.stem(t) for t in filtered]
    return " ".join(stemmed)

df_raw["clean_text"] = df_raw["text"].apply(preprocess_text)

# === 🔠 Vetorização + Divisão ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_raw["clean_text"])
y = df_raw["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 🧠 Treinamento ===
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# === 📊 Avaliação ===
y_pred = model.predict(X_test)
print("\n📊 Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))

# Gráfico de distribuição
sentiment_counts = df_raw["sentiment"].value_counts()
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind="bar", color=["green", "red", "gray"])
plt.title("Contagem de Sentimentos no Dataset")
plt.xlabel("Sentimento")
plt.ylabel("Quantidade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# === 📦 Pipeline para interpretação ===
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200))
pipeline.fit(df_raw["text"], df_raw["sentiment"])


# === 📘 LIME ===
def explain_with_lime(text):
    logging.info("[LIME] Explicando texto...")
    explainer = lime.lime_text.LimeTextExplainer(class_names=pipeline.classes_)
    exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=5)
    exp.save_to_file("lime_explanation.html")
    print("✅ LIME Explanation saved at 'lime_explanation.html'")
    print("🔍 LIME Top Words:")
    print(exp.as_list())


# === 🔴 SHAP ===
def explain_with_shap(text):
    logging.info("[SHAP] Explicando texto...")
    tfidf = pipeline.named_steps["tfidfvectorizer"]
    model = pipeline.named_steps["logisticregression"]

    explainer = shap.Explainer(model, tfidf.transform)
    vec_text = tfidf.transform([text])
    shap_values = explainer(vec_text)

    print("🔍 SHAP Words Importances:")
    shap.initjs()
    shap.plots.text(shap_values[0])


# === 🚀 CLI ===
def main():
    parser = argparse.ArgumentParser(description="Classificador de Sentimentos com NLP")
    parser.add_argument("--text", type=str, help="Frase para classificar e explicar")
    parser.add_argument(
        "--method", type=str, choices=["lime", "shap", "ambos"], default="ambos",
        help="Método de explicação: lime, shap ou ambos"
    )
    args = parser.parse_args()

    if args.text:
        sentiment = pipeline.predict([args.text])[0]
        print(f"\n🔮 Sentimento previsto: {sentiment}")

        if args.method in ("lime", "ambos"):
            explain_with_lime(args.text)

        if args.method in ("shap", "ambos"):
            explain_with_shap(args.text)


if __name__ == "__main__":
    main()
