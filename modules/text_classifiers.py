"""
==============================================================================
Module:      text_classifiers.py
Description: Supervised text classification (MultinomialNB) and unsupervised
             Gaussian Mixture Model (GMM) clustering with PCA and Silhouette analysis.
==============================================================================
"""

import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Tuple, Dict, Any, Optional


def run_naive_bayes_classification() -> Tuple[float, np.ndarray, list]:
    """Trainiert und evaluiert einen Naive-Bayes-Klassifikator auf dem 20 Newsgroups Datensatz."""
    categories = ['rec.sport.hockey', 'sci.space', 'talk.politics.mideast']
    train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    vectorizer = CountVectorizer(stop_words='english', max_features=3000)
    x_train = vectorizer.fit_transform(train_data.data)
    x_test = vectorizer.transform(test_data.data)

    model = MultinomialNB()
    model.fit(x_train, train_data.target)

    preds = model.predict(x_test)
    acc = float(accuracy_score(test_data.target, preds))
    conf_mat = confusion_matrix(test_data.target, preds)

    return acc, conf_mat, categories


def run_gmm_clustering(num_clusters: int = 4, save_plot: Optional[str] = None) -> Tuple[float, np.ndarray]:
    """Clusterung mittels Gaussian Mixture Model (GMM) auf TF-IDF Repräsentationen."""
    categories = ['rec.sport.hockey', 'sci.space', 'comp.graphics', 'talk.politics.mideast']
    dataset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)

    stop_words = set(stopwords.words('english'))

    def clean_text(raw_text: str) -> str:
        t = raw_text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = [w for w in word_tokenize(t) if w not in stop_words and w.isalpha()]
        return " ".join(tokens)

    cleaned = [clean_text(doc) for doc in dataset.data]
    vectorizer = TfidfVectorizer(max_features=1500)
    x_matrix = vectorizer.fit_transform(cleaned).toarray()

    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    labels = gmm.fit_predict(x_matrix)

    sil_score = float(silhouette_score(x_matrix, labels))

    # 2D-Projektion via PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_matrix)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels, palette='viridis', s=40, alpha=0.8)
    plt.title(f"GMM Clustering (2D PCA Projektion) | Silhouette: {sil_score:.4f}")
    plt.xlabel("Hauptkomponente 1")
    plt.ylabel("Hauptkomponente 2")

    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return sil_score, labels
