import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define method configurations
METHODS = {
    "role_agent": {
        "glob_pattern": "results/role_agent_summary_experiments/fold_*/window_10/role_agent_summary_chunk_semantic/retrieval_traces.csv",
        "label": "Role Agent",
    },
    "seekpolicy": {
        "glob_pattern": "results/seekpolicy_experiments/fold_*/window_10/rag_summary_semantic/retrieval_traces.csv",
        "label": "SeekPolicy",
    },
    "siamese": {
        "glob_pattern": [
            "results/retrieval_experiments/fold_*/window_10/climatebert_distilroberta-base-climate-f/contrastive/retrieval_traces.csv",
            "results/retrieval_experiments/fold_*/window_10/sentence-transformers_all-distilroberta-v1/contrastive/retrieval_traces.csv",
        ],
        "label": "Siamese (Contrastive)",
    },
}

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# Plotting function
def plot_similarity_distribution(df, method_label, output_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="retrieved_score",
        hue="is_relevant_doc",
        bins=30,
        palette={1.0: "green", 0.0: "red"},
        alpha=0.7,
        stat="density",
        common_norm=False,
    )
    plt.title(f"Cosine Similarity Distribution\n{method_label} (Window=10, All Folds)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend(["Incorrect", "Correct"])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


for method, config in METHODS.items():
    if isinstance(config["glob_pattern"], list):
        files = []
        for pattern in config["glob_pattern"]:
            files.extend(glob.glob(pattern))
    else:
        files = glob.glob(config["glob_pattern"])
    if not files:
        print(f"No files found for {method}")
        continue
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        # Standardize column names if needed
        if "retrieved_score" not in df.columns:
            continue
        if "is_relevant_doc" not in df.columns:
            continue
        dfs.append(df[["retrieved_score", "is_relevant_doc"]])
    if not dfs:
        print(f"No valid data for {method}")
        continue
    all_df = pd.concat(dfs, ignore_index=True)
    # Ensure is_relevant_doc is float (0.0 or 1.0)
    all_df["is_relevant_doc"] = all_df["is_relevant_doc"].astype(float)
    output_path = os.path.join(FIGURES_DIR, f"{method}_similarity_distribution.png")
    plot_similarity_distribution(all_df, config["label"], output_path)
    print(f"Saved: {output_path}")
