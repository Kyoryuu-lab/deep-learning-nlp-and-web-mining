"""
==============================================================================
Module:      web_and_social_mining.py
Description: Web graph scraping, server access log parsing, Reddit text mining,
             VADER sentiment scoring, and LDA topic discovery.
==============================================================================
"""

import re
import requests
import pandas as pd
import networkx as nx
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Any, Optional


def build_hyperlink_graph(start_url: str = "https://www.wikipedia.org/", max_pages: int = 15) -> nx.DiGraph:
    """Erzeugt einen gerichteten Graphen aus Hyperlinks einer Website."""
    graph = nx.DiGraph()
    visited = set()
    queue = [start_url]
    headers = {"User-Agent": "Mozilla/5.0"}

    while queue and len(visited) < max_pages:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        try:
            resp = requests.get(current, headers=headers, timeout=4)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if href.startswith('http'):
                        target = href
                    elif href.startswith('/'):
                        target = urljoin(current, href)
                    else:
                        continue
                    graph.add_edge(current, target)
                    if target not in visited and len(queue) < max_pages * 2:
                        queue.append(target)
        except Exception:
            continue

    return graph


def parse_access_log_entries(log_lines: List[str]) -> pd.DataFrame:
    """Extrahiert Host, Zeitstempel, HTTP-Methode, URL und Statuscode aus Webserver-Logs."""
    pattern = re.compile(
        r'(?P<host>\S+) \S+ \S+ \[(?P<time>.*?)\] "(?P<method>\S+) (?P<url>\S+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<size>\S+)'
    )
    records = []
    for line in log_lines:
        match = pattern.match(line.strip())
        if match:
            entry = match.groupdict()
            entry["size"] = int(entry["size"]) if entry["size"].isdigit() else 0
            records.append(entry)
    return pd.DataFrame(records)


def analyze_vader_and_lda(texts: List[str], num_topics: int = 2) -> Dict[str, Any]:
    """Berechnet VADER-Polaritäten und generiert LDA-Themenvektoren."""
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for t in texts:
        score = analyzer.polarity_scores(t)['compound']
        label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        sentiments.append(label)

    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    x_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(x_matrix)

    feature_names = vectorizer.get_feature_names_out()
    extracted_topics = []
    for idx, topic_dist in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic_dist.argsort()[:-6:-1]]
        extracted_topics.append((idx, ", ".join(top_words)))

    return {"sentiments": sentiments, "topics": extracted_topics}
