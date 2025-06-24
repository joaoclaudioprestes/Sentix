# 🧠 O que é NLP (Processamento de Linguagem Natural)?

NLP (Natural Language Processing) é uma área da Inteligência Artificial que tem como objetivo permitir que computadores **compreendam, interpretem, manipulem e gerem** a linguagem humana — seja escrita ou falada — de maneira útil e eficiente. Para isso, o NLP combina conhecimentos de linguística, estatística e técnicas de machine learning.

---

# 🧩 Como funciona o NLP?

O processamento de linguagem natural envolve várias etapas que transformam textos ou falas em dados estruturados que máquinas podem entender. Veja as principais fases:

## 🔠 1. Tokenização

É o processo que **divide o texto em partes menores**, geralmente palavras ou frases, chamadas de *tokens*. Essa etapa é fundamental para que o texto possa ser analisado em unidades menores.

Exemplo:

```python
"Eu gosto de Python"  ➜  ["Eu", "gosto", "de", "Python"]
```

---

## 🛑 2. Remoção de Stopwords

Stopwords são palavras muito comuns e que, na maioria dos casos, não agregam muito significado para a análise, como artigos, preposições e conjunções ("de", "o", "e", "a"). Removê-las ajuda a focar nas palavras mais importantes do texto.

---

## 🔁 3. Stemming e Lemmatization

Ambos são técnicas para **reduzir palavras às suas formas básicas ou raízes**, para agrupar variações da mesma palavra.

* **Stemming:** Técnica mais simples e rápida que remove sufixos para chegar a um radical aproximado. Pode gerar palavras truncadas e não muito "corretas".
  Exemplo: "gostando", "gostei" ➜ "gost"

* **Lemmatization:** Técnica mais avançada que utiliza dicionários e regras gramaticais para encontrar a forma base correta da palavra (o *lem*).
  Exemplo: "gostando", "gostei" ➜ "gostar"

---

## 🧮 4. Vetorização (Representação numérica do texto)

Modelos de machine learning precisam de números, então o texto deve ser convertido em vetores numéricos que representem seu conteúdo.

Principais métodos:

* **Bag of Words (BoW):** Conta a frequência de cada palavra no texto, sem considerar a ordem ou contexto.

* **TF-IDF (Term Frequency-Inverse Document Frequency):** Calcula a importância de uma palavra no documento e em todo o conjunto de textos, dando peso maior para palavras relevantes e menos frequentes.

* **Word Embeddings:** Representações vetoriais densas que capturam o contexto e significado das palavras, como Word2Vec, GloVe ou modelos mais avançados como BERT.

---

## 🤖 5. Treinamento com Machine Learning

Depois de transformar os textos em números, algoritmos de machine learning podem aprender padrões e fazer previsões ou classificações.

Alguns modelos populares para NLP:

**Regressão Logística:** Modelo simples e eficiente para classificação binária ou multiclasse.

**Naive Bayes:** Baseado em probabilidade, funciona bem para textos e é rápido.

**SVM (Support Vector Machines):** Modelo poderoso para classificação com alta dimensionalidade.

**Árvores de Decisão:** Modelos interpretáveis que dividem os dados em regras.

**Redes Neurais:** Desde simples perceptrons até redes profundas, são a base de muitos avanços recentes.

**Transformers:** Arquiteturas modernas (como BERT, RoBERTa, GPT) que capturam contexto complexo e são usadas em tarefas avançadas.

---

# 🔍 Interpretando Modelos com LIME e SHAP

Mesmo que um modelo apresente boa performance, é essencial entender **como e por que ele toma certas decisões** para garantir confiança, detectar erros e melhorar o sistema.

## 🟡 LIME (Local Interpretable Model-Agnostic Explanations)

LIME é uma técnica que explica **previsões individuais** de qualquer modelo, mesmo os complexos, de forma simples e local.

Como funciona:

* Cria pequenas perturbações na entrada (ex: altera palavras do texto).

* Observa como a saída do modelo muda com essas perturbações.

* Ajusta um modelo simples (ex: regressão linear) que aproxima o comportamento do modelo original naquele ponto.

**No NLP**, LIME destaca as palavras que mais influenciaram a decisão para aquela instância específica.

Exemplo visual:

Texto: "O produto é incrível, eu amei"
→ LIME destaca "incrível" e "amei" como palavras que contribuíram para classificar como "positivo".

---

## 🔴 SHAP (SHapley Additive exPlanations)

SHAP é uma técnica baseada em teoria dos jogos que calcula a contribuição individual de cada característica (palavra, no caso de NLP) para a previsão do modelo.

Características do SHAP:

* Funciona para qualquer modelo.

* Oferece explicações **globais** (entender o modelo como um todo) e **locais** (explicar uma previsão específica).

* Garante propriedades matemáticas importantes, como consistência e justiça na atribuição de importância.

**No NLP**, SHAP mostra se cada palavra **aumenta ou diminui** a probabilidade de uma determinada classe, ajudando a identificar viés ou problemas.

Exemplo visual:

Texto: "Não gostei do atendimento"
→ SHAP mostra que a palavra "não" reduz fortemente a chance da classificação ser positiva.

---

## ✅ Por que usar LIME ou SHAP?

**Confiança e transparência:** Entender *por que* o modelo tomou determinada decisão.

**Detecção de erros e viés:** Identificar falhas ou preconceitos presentes nos dados ou no modelo.

**Melhoria do modelo:** Ajustar ou refinar o sistema com base no impacto das palavras/características.
