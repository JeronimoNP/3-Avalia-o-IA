# 3-Avaliação-IA

Este projeto implementa e compara algoritmos de machine learning (SVM e K-Means) com e sem aplicação de PCA (Principal Component Analysis).

## 📋 Descrição do Projeto

O projeto consiste em duas análises principais:

### 1. SVM (Support Vector Machine) - Classificação de Reviews
- **Arquivo**: `SVM/SVM.py`
- **Dataset**: `reviews.csv` e `reviews1.csv`
- **Objetivo**: Classificar reviews de texto como positivos ou negativos
- **Comparação**: SVM sem PCA vs SVM com PCA

### 2. K-Means - Clustering de Clientes
- **Arquivo**: `K-Mean/k-means.py`
- **Dataset**: `Mall_Customers.csv`
- **Objetivo**: Segmentar clientes com base em idade, renda anual e pontuação de gastos
- **Comparação**: K-Means sem PCA vs K-Means com PCA

## 🛠️ Tecnologias Utilizadas

- **Python 3.11+**
- **Bibliotecas**:
  - pandas - Manipulação de dados
  - numpy - Computação numérica
  - scikit-learn - Algoritmos de ML
  - matplotlib - Visualização
  - seaborn - Visualização estatística
  - nltk - Processamento de linguagem natural

## 📦 Instalação

### 1. Clone o repositório
```bash
git clone <url-do-repositorio>
cd 3-Avalia-o-IA
```

### 2. Instale as dependências
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

### 3. Baixe os recursos do NLTK (necessário para o SVM)
```python
import nltk
nltk.download('stopwords')
```

## 🚀 Como Executar

### SVM - Classificação de Reviews

1. Navegue para o diretório SVM:
```bash
cd SVM
```

2. Execute o script SVM:
```bash
python SVM.py
```

**O que o script faz:**
- Carrega e pré-processa o dataset de reviews
- Treina SVM sem PCA
- Aplica PCA e treina SVM com PCA
- Compara métricas de performance (acurácia, matriz de confusão)
- Gera visualização 2D dos dados após PCA

### K-Means - Clustering de Clientes

1. Navegue para o diretório K-Mean:
```bash
cd K-Mean
```

2. Execute o script K-Means:
```bash
python k-means.py
```

**O que o script faz:**
- Carrega dataset de clientes do shopping
- Normaliza os dados
- Executa K-Means sem PCA
- Aplica PCA para 2 componentes
- Executa K-Means com PCA
- Compara métricas (Silhouette Score, Davies-Bouldin, etc.)
- Gera visualização dos clusters

## 📊 Estrutura do Projeto

```
3-Avalia-o-IA/
│
├── README.md                 # Este arquivo
├── reviews.csv              # Dataset de reviews
├── reviews1.csv             # Dataset alternativo de reviews
│
├── SVM/
│   └── SVM.py               # Classificação com SVM
│
└── K-Mean/
    ├── k-means.py           # Clustering com K-Means
    └── Mall_Customers.csv   # Dataset de clientes
```

## 📈 Métricas Avaliadas

### SVM (Classificação)
- **Acurácia**: Porcentagem de predições corretas
- **Matriz de Confusão**: Análise detalhada de acertos/erros
- **Relatório de Classificação**: Precision, Recall, F1-Score
- **Tempo de Execução**: Performance computacional

### K-Means (Clustering)
- **Silhouette Score**: Qualidade dos clusters (quanto maior, melhor)
- **Davies-Bouldin Score**: Separação entre clusters (quanto menor, melhor)
- **Calinski-Harabasz Score**: Razão de dispersão (quanto maior, melhor)
- **Inertia**: Soma das distâncias ao centroide (quanto menor, melhor)

## 🎯 Objetivos da Comparação

### PCA (Principal Component Analysis)
- **Vantagens**: Redução de dimensionalidade, remoção de ruído, visualização
- **Desvantagens**: Possível perda de informação, interpretabilidade reduzida

### Comparação Esperada
- **SVM**: PCA pode melhorar performance removendo features irrelevantes
- **K-Means**: PCA facilita visualização, mas pode afetar qualidade dos clusters

## 🔧 Possíveis Problemas e Soluções

### Erro: "No such file or directory"
- **Causa**: Executar script no diretório errado
- **Solução**: Navegar para o diretório correto antes de executar

### Erro: "n_components must be between 0 and min(n_samples, n_features)"
- **Causa**: Número de componentes PCA maior que features disponíveis
- **Solução**: Reduzir `n_components` no código

### Erro: NLTK resources not found
- **Causa**: Recursos do NLTK não baixados
- **Solução**: Executar `nltk.download('stopwords')`

## 👨‍💻 Autor

Projeto desenvolvido para avaliação de técnicas de Machine Learning com e sem redução de dimensionalidade.