<h1>deep-learning-nlp-and-web-mining</h1>
<p>End-to-end Machine Learning, Deep Learning, and Computational Linguistics suite: PyTorch RNN/LSTM/GRU benchmarks, Seq2Seq translation, Hugging Face GPT-2 fine-tuning, GMM text clustering, Web-crawler graph analysis, and server log parsing</p>

<h2>Project description</h2>
<p>This repository provides a modular, production-grade computational framework implementing advanced machine learning and natural language processing pipelines in Python:</p>
<ol>
  <li>
    <dl>
      <dt><b>Deep Learning &amp; Neural Sequence Modeling:</b></dt>
      <dd><b>Recurrent Text Classification:</b> Compares standard RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) architectures on sentiment polarity tasks.</dd>
      <dd><b>Sequence-to-Sequence Translation:</b> Implements an Encoder-Decoder GRU translation pipeline with dynamic token indexing (<code>&lt;SOS&gt;</code>, <code>&lt;EOS&gt;</code>) and negative log-likelihood optimization.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Generative Language Models &amp; Transformers:</b></dt>
      <dd>Executes autoregressive text generation and temperature-controlled sampling with Hugging Face GPT-2.</dd>
      <dd>Fine-tunes causal language models on WikiText-2 using Hugging Face <code>Trainer</code> and <code>DataCollatorForLanguageModeling</code>.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Statistical Classification &amp; Density Clustering:</b></dt>
      <dd>Classifies multi-class document topics on 20 Newsgroups using Multinomial Naive Bayes and confusion matrices.</dd>
      <dd>Discovers latent clusters in TF-IDF feature spaces using Gaussian Mixture Models (GMM), evaluated via Silhouette scores and 2D PCA embeddings.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Web Graph Mining &amp; Social Text Analytics:</b></dt>
      <dd>Crawls web page links and constructs directed hyperlink graphs using NetworkX.</dd>
      <dd>Parses structured HTTP web server access logs using regular expression patterns into Pandas DataFrames.</dd>
      <dd>Analyzes public discussion threads using VADER sentiment scoring and Latent Dirichlet Allocation (LDA) topic discovery.</dd>
    </dl>
  </li>
</ol>

<h2>Project files</h2>
<table>
  <tr>
    <th colspan="2">deep-learning-nlp-and-web-mining</th>
  </tr>
  <tr>
    <td>main.py</td>
    <td>Master CLI entry point providing modular dispatch across all deep learning and NLP routines</td>
  </tr>
  <tr>
    <td>modules/sequence_models.py</td>
    <td>PyTorch RNN/LSTM/GRU recurrent classifier and Seq2Seq Encoder-Decoder translation engine</td>
  </tr>
  <tr>
    <td>modules/transformer_pipeline.py</td>
    <td>Hugging Face GPT-2 inference and Trainer-based fine-tuning on WikiText-2</td>
  </tr>
  <tr>
    <td>modules/text_classifiers.py</td>
    <td>Multinomial Naive Bayes classifier and Gaussian Mixture Model (GMM) clustering with PCA</td>
  </tr>
  <tr>
    <td>modules/web_and_social_mining.py</td>
    <td>Web crawler graph builder, server access log parser, VADER sentiment, and LDA topic discovery</td>
  </tr>
  <tr>
    <td>README.md</td>
    <td>Technical specification, architectural overview, and usage guidelines</td>
  </tr>
  <tr>
    <td>.gitignore</td>
    <td>Exclusion rules for model checkpoints, PyTorch/Hugging Face caches, and generated plots</td>
  </tr>
</table>

<h2>Methodological Formulation</h2>
<table>
  <tr>
    <th>Task</th>
    <th>Architecture / Algorithm</th>
    <th>Mathematical Model</th>
  </tr>
  <tr>
    <td><b>Recurrent Gate</b></td>
    <td>Gated Recurrent Unit (GRU)</td>
    <td>$z_t = \sigma(W_z x_t + U_z h_{t-1}), \quad r_t = \sigma(W_r x_t + U_r h_{t-1}), \quad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$</td>
  </tr>
  <tr>
    <td><b>Translation Loss</b></td>
    <td>Negative Log-Likelihood</td>
    <td>$\mathcal{L}_{\text{NLL}} = -\sum_{t=1}^T \log P(y_t \mid y_{&lt;t}, \mathbf{c})$</td>
  </tr>
  <tr>
    <td><b>Topic Discovery</b></td>
    <td>Latent Dirichlet Allocation</td>
    <td>$P(\mathbf{w}, \mathbf{z} \mid \boldsymbol{\alpha}, \boldsymbol{\beta}) = \int \prod_{d=1}^D P(\theta_d \mid \boldsymbol{\alpha}) \prod_{n=1}^{N_d} P(z_{dn} \mid \theta_d) P(w_{dn} \mid z_{dn}, \boldsymbol{\beta}) \, d\theta_d$</td>
  </tr>
  <tr>
    <td><b>Clustering Density</b></td>
    <td>Gaussian Mixture Model</td>
    <td>$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$</td>
  </tr>
</table>

<h2>Installation and launch</h2>
<ol>
  <b>
    <li>
      <dl>
        <dt>Clone repository</dt>
        <dd><code>git clone https://github.com/Kyoryuu-lab/deep-learning-nlp-and-web-mining.git</code></dd>
        <dd><code>cd deep-learning-nlp-and-web-mining</code></dd>
      </dl>
    </li>
  </b>
  <b>
    <li>
      <dl>
        <dt>Install dependencies</dt>
        <dd><code>pip install torch transformers datasets scikit-learn nltk beautifulsoup4 networkx pandas matplotlib seaborn vaderSentiment</code></dd>
      </dl>
    </li>
  </b>
  <b>
    <li>
      <dl>
        <dt>Run complete suite</dt>
        <dd><code>python main.py --module all --save-plots</code></dd>
        <dd>Execute individual modules: <code>python main.py --module seq2seq</code> (Options: <code>seq2seq</code>, <code>gpt2</code>, <code>classify</code>, <code>cluster</code>, <code>webmining</code>)</dd>
      </dl>
    </li>
  </b>
</ol>

<h2>Example of console output (python main.py --module all)</h2>
<samp>
  <p>
    ==============================================================================<br />
    &nbsp;&nbsp;1. MODUL: SEQ2SEQ MASCHINELLE UEBERSETZUNG (PYTORCH GRU)<br />
    ==============================================================================<br />
    [*] Original (FR): "je suis heureux"<br />
    [+] Uebersetzung (EN): "i am happy"<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;2. MODUL: GENERATIVE TRANSFORMER (HUGGING FACE GPT-2)<br />
    ==============================================================================<br />
    [*] Prompt: "What is the capital of Ukraine?"<br />
    [+] GPT-2 Generierung:<br />
    &nbsp;&nbsp;&nbsp;&nbsp;"What is the capital of Ukraine? The government says that Kyiv is the heart..."<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;3. MODUL: TEXTKLASSIFIKATION MIT MULTINOMIAL NAIVE BAYES<br />
    ==============================================================================<br />
    [*] Genauigkeit auf 20 Newsgroups Test-Split: 89.42 %<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;4. MODUL: UNUEBERWACHTES DICHTE-CLUSTERING (GMM + PCA)<br />
    ==============================================================================<br />
    [*] GMM Konvergenz erreicht | Mittlere Silhouette-Breite: 0.0842<br />
    [+] 2D-Projektion gespeichert unter: gmm_clusters_pca.png<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;ALLE NLP- &amp; MACHINE LEARNING ANALYSEN ERFOLGREICH BEENDET.<br />
    ==============================================================================
  </p>
