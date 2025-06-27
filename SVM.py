import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Baixar stopwords do nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

# =============================
# ETAPA 1 - Carregar dataset de reviews
# =============================

# Usaremos um dataset pequeno de exemplo com 2 classes: positivo e negativo
# Substitua isso depois por um maior se quiser
data = pd.read_csv("reviews.csv")

df = pd.DataFrame(data)

# =============================
# ETAPA 2 - Pré-processar e vetorizar texto
# =============================

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove pontuação
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['cleaned'] = df['text'].apply(preprocess)

# Vetorização com TF-IDF
vectorizer = TfidfVectorizer(max_features=50)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = np.array(df['label'])

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =============================
# ETAPA 3 - SVM SEM PCA
# =============================
print("\nTreinando SVM SEM PCA...")
start = time.time()
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
end = time.time()

y_pred = svm_clf.predict(X_test)

print("Acurácia SEM PCA:", accuracy_score(y_test, y_pred))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))
print("Tempo:", round(end - start, 4), "s")

# =============================
# ETAPA 4 - PCA + SVM
# =============================

print("\nAplicando PCA e treinando SVM COM PCA...")

# Reduzir para 2 componentes só pra exemplo
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

start = time.time()
svm_pca = SVC(kernel='linear')
svm_pca.fit(X_train_pca, y_train)
end = time.time()

y_pred_pca = svm_pca.predict(X_test_pca)

print("Acurácia COM PCA:", accuracy_score(y_test, y_pred_pca))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred_pca))
print("Tempo:", round(end - start, 4), "s")

# =============================
# ETAPA 5 - Visualização (opcional)
# =============================
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.title("Pontos em 2D após PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()
