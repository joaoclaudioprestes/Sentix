# Sentix

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&style=for-the-badge)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#)

---

## 🚀 Sobre o projeto

Sentix é um classificador de sentimentos de avaliações de produtos que utiliza técnicas avançadas de **Processamento de Linguagem Natural (NLP)** para compreender e interpretar opiniões reais de clientes.

O modelo foi treinado com um **dataset real de e-commerce do Kaggle** e incorpora pré-processamento eficiente para maximizar a qualidade dos dados.

Além disso, utiliza técnicas de explicação de modelos, como **LIME** e **SHAP**, para interpretar as decisões da máquina de forma transparente.

---

## 🧠 O que é NLP (Processamento de Linguagem Natural)?

NLP (Natural Language Processing) é uma área da Inteligência Artificial que tem como objetivo permitir que computadores **compreendam, interpretem, manipulem e gerem** a linguagem humana — escrita ou falada — de forma útil e eficiente. Para isso, combina linguística, estatística e machine learning.

### Como funciona o NLP no Sentix?

1. **Tokenização:** Divide o texto em palavras ou frases menores chamadas *tokens*.  
   Exemplo:  
   `"Eu gosto de Python"` → `["Eu", "gosto", "de", "Python"]`

2. **Remoção de Stopwords:** Remove palavras comuns que pouco agregam para análise (como "de", "o", "e").

3. **Stemming:** Reduz palavras às suas raízes para agrupar variações.  
   Exemplo: "gostando", "gostei" → "gost"

4. **Vetorização (TF-IDF):** Converte o texto em números que representam a importância das palavras no documento e no conjunto de textos.

5. **Treinamento:** Algoritmos de machine learning aprendem padrões para classificar os sentimentos das avaliações.

---

## 🔍 Interpretando modelos com LIME e SHAP

Para garantir que as decisões do modelo sejam transparentes e confiáveis, Sentix usa duas técnicas poderosas:

### 🟡 LIME (Local Interpretable Model-Agnostic Explanations)

- Explica previsões individuais do modelo, destacando quais palavras influenciaram a decisão.  
- Exemplo: No texto "O produto é incrível, eu amei", LIME destaca "incrível" e "amei" como palavras-chave para a classificação positiva.

### 🔴 SHAP (SHapley Additive exPlanations)

- Calcula a contribuição de cada palavra para a previsão, mostrando se aumentam ou diminuem a probabilidade de uma classe.  
- Exemplo: No texto "Não gostei do atendimento", SHAP mostra que a palavra "não" reduz fortemente a chance de ser classificado como positivo.

---

## 🛠️ Tecnologias e Ferramentas

- Python 3.11  
- NLP (tokenização, stopwords, stemming)  
- TF-IDF para vetorização  
- Machine Learning (ex: regressão logística, Naive Bayes)  
- LIME e SHAP para interpretação de modelos  
- Pandas, Numpy  
---

## ⚙️ Como rodar o projeto

```bash
# Clone o repositório
git clone https://github.com/seuusuario/sentix.git
cd sentix

# Instale as dependências
pip install -r requirements.txt

# Execute o script principal
python main.py --text "O produto levou muito tempo para ser entregue! Não compraria novamente..." 
````

---

## 🤝 Contribuição

Contribuições são bem-vindas! Faça um fork, crie sua branch com a feature e envie um pull request.

---

## ✉️ Contato

João Prestes – [joaoprestes@email.com](mailto:joaoprestesdev@gmail.com)
[LinkedIn](https://www.linkedin.com/in/joaoclaudioprestes) | [GitHub](https://github.com/joaoclaudioprestes)

---

Feito com ❤️ e Python! 🐍