"""
==============================================================================
Project:     Deep Learning, NLP & Web Mining Intelligence Suite
File:        main.py
Author:      Yevhen Biezhyn
Description: Unified CLI orchestrator executing PyTorch Seq2Seq translation,
             Hugging Face GPT-2 generation, Naive Bayes, GMM clustering,
             and web hyperlink mining.
==============================================================================
"""

import argparse
from modules.sequence_models import run_seq2seq_demo
from modules.transformer_pipeline import generate_text_gpt2, finetune_gpt2_wikitext
from modules.text_classifiers import run_naive_bayes_classification, run_gmm_clustering
from modules.web_and_social_mining import (
    build_hyperlink_graph,
    parse_access_log_entries,
    analyze_vader_and_lda
)


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Deep Learning, NLP & Web Mining Intelligence Suite",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--module",
        choices=["all", "seq2seq", "gpt2", "classify", "cluster", "webmining"],
        default="all",
        help=(
            "all       : Fuehrt alle Demos sequenziell aus (Standard)\n"
            "seq2seq   : PyTorch Encoder-Decoder GRU Uebersetzung\n"
            "gpt2      : Hugging Face GPT-2 Inferenz & Fine-Tuning\n"
            "classify  : MultinomialNB Textklassifikation (20 Newsgroups)\n"
            "cluster   : Gaussian Mixture Models & PCA Dichte-Clustering\n"
            "webmining : Hyperlink-Graph, Log-Parsing & VADER/LDA"
        )
    )
    parser.add_argument("--save-plots", action="store_true", help="Speichert generierte Visualisierungen als PNG")
    args = parser.parse_args()

    print("=" * 78)
    print("  DEEP LEARNING, NLP & WEB MINING INTELLIGENCE SUITE")
    print("=" * 78)

    # 1. Seq2Seq Translation
    if args.module in ["all", "seq2seq"]:
        print_banner("1. Modul: Seq2Seq Maschinelle Uebersetzung (PyTorch GRU)")
        run_seq2seq_demo(epochs=200)

    # 2. GPT-2 Transformer
    if args.module in ["all", "gpt2"]:
        print_banner("2. Modul: Generative Transformer (Hugging Face GPT-2)")
        sample_prompt = "What is the capital of Ukraine?"
        gen_out = generate_text_gpt2(sample_prompt, max_length=35)
        print(f"[*] Prompt: \"{sample_prompt}\"")
        print(f"[+] GPT-2 Generierung:\n    \"{gen_out}\"")

    # 3. Naive Bayes Classification
    if args.module in ["all", "classify"]:
        print_banner("3. Modul: Textklassifikation mit Multinomial Naive Bayes")
        acc, conf, cat_labels = run_naive_bayes_classification()
        print(f"[*] Genauigkeit auf 20 Newsgroups Test-Split: {acc * 100:.2f} %")
        print(f"[*] Kategorien: {cat_labels}")
        print("    Konfusionsmatrix:\n", conf)

    # 4. GMM Clustering
    if args.module in ["all", "cluster"]:
        print_banner("4. Modul: Unueberwachtes Dichte-Clustering (GMM + PCA)")
        plot_path = "gmm_clusters_pca.png" if args.save_plots else None
        sil, _ = run_gmm_clustering(num_clusters=4, save_plot=plot_path)
        print(f"[*] GMM Konvergenz erreicht | Mittlere Silhouette-Breite: {sil:.4f}")
        if args.save_plots:
            print(f"[+] 2D-Projektion gespeichert unter: {plot_path}")

    # 5. Web Mining, Logs & VADER
    if args.module in ["all", "webmining"]:
        print_banner("5. Modul: Web Mining, Log-Parsing & VADER Sentiment")
        # Log Parser Demo
        sample_logs = [
            '192.168.1.10 - - [10/Feb/2026:14:32:10 +0000] "GET /index.html HTTP/1.1" 200 1024',
            '10.0.0.5 - - [10/Feb/2026:14:32:15 +0000] "POST /login HTTP/1.1" 401 256',
            '192.168.1.10 - - [10/Feb/2026:14:32:20 +0000] "GET /data/report.pdf HTTP/1.1" 200 4096'
        ]
        log_df = parse_access_log_entries(sample_logs)
        print(f"[*] Geparste Log-Eintraege:\n{log_df[['host', 'method', 'url', 'status']]}")

        # Sentiment & LDA
        posts = [
            "Python 3.12 release delivers impressive runtime speedups and better traces.",
            "Encountered frustrating memory leak bugs in distributed training workers.",
            "Evaluating neural architectures and vector databases for retrieval."
        ]
        analytics = analyze_vader_and_lda(posts, num_topics=2)
        print(f"\n[*] VADER Sentimentverteilung: {analytics['sentiments']}")
        for tid, t_str in analytics['topics']:
            print(f"    Thema #{tid}: {t_str}")

    print("\n" + "=" * 78)
    print("  ALLE NLP- & MACHINE LEARNING ANALYSEN ERFOLGREICH BEENDET.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
