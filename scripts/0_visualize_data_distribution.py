import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("data/csv")
KFOLD_DIR = DATA_DIR / "kfold"
FIGURE_DIR = Path("figures")
FOLD_HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo", "--", "||", "**"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize grouped K-fold train/test distributions."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional fold index to visualize. If omitted, all folds are processed.",
    )
    return parser.parse_args()


def get_available_folds() -> list[int]:
    if not KFOLD_DIR.exists():
        raise FileNotFoundError(
            f"K-fold directory not found: {KFOLD_DIR}. Run scripts/0_split_test_set.py first."
        )

    fold_ids: set[int] = set()
    for train_file in KFOLD_DIR.glob("fold_*_train.csv"):
        parts = train_file.stem.split("_")
        if len(parts) != 3:
            continue
        _, fold_text, split_name = parts
        if split_name != "train" or not fold_text.isdigit():
            continue

        fold_id = int(fold_text)
        test_file = KFOLD_DIR / f"fold_{fold_id}_test.csv"
        if test_file.exists():
            fold_ids.add(fold_id)

    if not fold_ids:
        raise FileNotFoundError(
            f"No valid fold train/test pairs found in {KFOLD_DIR}. "
            "Expected files like fold_0_train.csv and fold_0_test.csv."
        )

    return sorted(fold_ids)


def load_fold_datasets(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = KFOLD_DIR / f"fold_{fold}_train.csv"
    test_path = KFOLD_DIR / f"fold_{fold}_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing K-fold split files for selected fold. "
            f"Expected: {train_path} and {test_path}."
        )

    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)

    if "Document ID" in train_set.columns and "Document ID" in test_set.columns:
        train_set = train_set[~train_set["Document ID"].isin(test_set["Document ID"])]

    return train_set, test_set


def percentage_distribution(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].value_counts(normalize=True) * 100


def sector_distribution(df: pd.DataFrame) -> pd.Series:
    expanded = df.copy()
    expanded["Sector"] = expanded["Sector"].fillna("").str.split(";")
    expanded = expanded.explode("Sector")
    expanded["Sector"] = expanded["Sector"].astype(str).str.strip()
    expanded = expanded[expanded["Sector"].ne("") & expanded["Sector"].ne("Other")]
    return expanded["Sector"].value_counts(normalize=True) * 100


def plot_multi_fold_bar(
    fold_series: dict[int, tuple[pd.Series, pd.Series]],
    title_prefix: str,
    output_path: Path,
    xlabel: str,
    ylabel: str,
    mapping_box_text: str | None = None,
) -> None:
    n_folds = len(fold_series)

    if mapping_box_text:
        n_cols = 3
        n_rows = max(3, math.ceil((n_folds + 1) / n_cols))
        preferred_mapping_index = 5
        max_axis_index = n_rows * n_cols - 1
        mapping_axis_index = min(preferred_mapping_index, max_axis_index)
    else:
        n_cols = 2
        n_rows = math.ceil(n_folds / n_cols)
        mapping_axis_index = -1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

    if hasattr(axes, "flatten"):
        axes_list = list(axes.flatten())
    else:
        axes_list = [axes]

    plot_axis_indices = list(range(len(axes_list)))
    if mapping_box_text and mapping_axis_index >= 0:
        plot_axis_indices.remove(mapping_axis_index)

    for idx, (fold, (train_series, test_series)) in enumerate(
        sorted(fold_series.items())
    ):
        ax = axes_list[plot_axis_indices[idx]]
        fold_hatch = FOLD_HATCHES[idx % len(FOLD_HATCHES)]
        combined = pd.concat(
            [train_series.rename("train"), test_series.rename("test")], axis=1
        ).fillna(0)

        combined.plot(
            kind="bar",
            ax=ax,
            width=0.8,
            color=["#1f77b4", "#ff7f0e"],
            legend=(idx == 0),
        )
        ax.set_title(f"Fold {fold}", fontsize=20)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis="x", rotation=45, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for patch in ax.patches:
            patch.set_hatch(fold_hatch)

    for axis_idx in plot_axis_indices[n_folds:]:
        axes_list[axis_idx].axis("off")

    fig.suptitle(title_prefix, fontsize=24, fontweight="bold")

    if mapping_box_text and mapping_axis_index >= 0:
        mapping_ax = axes_list[mapping_axis_index]
        mapping_ax.axis("off")
        mapping_ax.text(
            0.01,
            0.99,
            mapping_box_text,
            ha="left",
            va="top",
            fontsize=16,
            family="monospace",
            transform=mapping_ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        mapping_ax.set_title("Sector code mapping")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg")
    plt.close()


def build_short_label_map(
    fold_series: dict[int, tuple[pd.Series, pd.Series]], prefix: str
) -> dict[str, str]:
    labels: set[str] = set()
    for train_series, test_series in fold_series.values():
        labels.update([str(item) for item in train_series.index])
        labels.update([str(item) for item in test_series.index])

    sorted_labels = sorted(labels)
    return {
        full_label: f"{prefix}{idx:02d}"
        for idx, full_label in enumerate(sorted_labels, start=1)
    }


def apply_index_label_map(
    fold_series: dict[int, tuple[pd.Series, pd.Series]], label_map: dict[str, str]
) -> dict[int, tuple[pd.Series, pd.Series]]:
    remapped: dict[int, tuple[pd.Series, pd.Series]] = {}
    for fold, (train_series, test_series) in fold_series.items():
        train_copy = train_series.copy()
        test_copy = test_series.copy()
        train_copy.index = [
            label_map.get(str(item), str(item)) for item in train_copy.index
        ]
        test_copy.index = [
            label_map.get(str(item), str(item)) for item in test_copy.index
        ]
        remapped[fold] = (train_copy, test_copy)
    return remapped


def format_mapping_box(label_map: dict[str, str], title: str) -> str:
    lines = [title]
    for full_label, short_label in sorted(label_map.items(), key=lambda item: item[1]):
        lines.append(f"{short_label} = {full_label}")
    return "\n".join(lines)


def format_fold_hatch_box(folds: list[int], title: str) -> str:
    lines = [title]
    for idx, fold in enumerate(folds):
        hatch = FOLD_HATCHES[idx % len(FOLD_HATCHES)] or "(solid)"
        lines.append(f"Fold {fold} = {hatch}")
    return "\n".join(lines)


def get_alpha_counts(df: pd.DataFrame, label: str) -> pd.DataFrame:
    counts = df["alpha-3"].value_counts().reset_index()
    counts.columns = ["alpha-3", label]
    return counts


def plot_country_maps_all_folds(
    fold_data: dict[int, tuple[pd.DataFrame, pd.DataFrame]], output_path: Path
) -> None:
    try:
        import geopandas as gpd
    except ModuleNotFoundError:
        print("Skipping country map figure: geopandas is not installed.")
        return

    try:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
    except Exception:
        print("Skipping country map figure: Natural Earth dataset could not be loaded.")
        return

    n_folds = len(fold_data)
    fig, axes = plt.subplots(n_folds, 2, figsize=(20, 6 * n_folds))

    if n_folds == 1:
        axes = [axes]

    for row_idx, fold in enumerate(sorted(fold_data.keys())):
        train_set, test_set = fold_data[fold]
        train_counts = get_alpha_counts(train_set, "count")
        test_counts = get_alpha_counts(test_set, "count")

        world_train = world.merge(
            train_counts, left_on="ISO_A3", right_on="alpha-3", how="left"
        )
        world_test = world.merge(
            test_counts, left_on="ISO_A3", right_on="alpha-3", how="left"
        )

        ax_train = axes[row_idx][0]
        ax_test = axes[row_idx][1]

        world_train.boundary.plot(ax=ax_train, color="black", linewidth=0.4)
        world_train.plot(
            column="count",
            ax=ax_train,
            cmap="Blues",
            legend=False,
            missing_kwds={"color": "lightgrey"},
        )
        ax_train.set_title(f"Fold {fold} - Train")
        ax_train.set_axis_off()

        world_test.boundary.plot(ax=ax_test, color="black", linewidth=0.4)
        world_test.plot(
            column="count",
            ax=ax_test,
            cmap="Oranges",
            legend=False,
            missing_kwds={"color": "lightgrey"},
        )
        ax_test.set_title(f"Fold {fold} - Test")
        ax_test.set_axis_off()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg")
    plt.close()


def main() -> None:
    args = parse_args()
    folds = [args.fold] if args.fold is not None else get_available_folds()

    fold_data: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    region_series: dict[int, tuple[pd.Series, pd.Series]] = {}
    sector_series: dict[int, tuple[pd.Series, pd.Series]] = {}

    for fold in folds:
        train_set, test_set = load_fold_datasets(fold)
        fold_data[fold] = (train_set, test_set)

        region_series[fold] = (
            percentage_distribution(train_set, "region"),
            percentage_distribution(test_set, "region"),
        )
        sector_series[fold] = (
            sector_distribution(train_set),
            sector_distribution(test_set),
        )

    summary_dir = FIGURE_DIR / "kfold_summary"

    plot_multi_fold_bar(
        fold_series=region_series,
        title_prefix="Region Distribution by Fold (Train vs Test)",
        output_path=summary_dir / "region_distribution_all_folds.svg",
        xlabel="Region",
        ylabel="Percentage of Documents (%)",
        mapping_box_text=None,
    )

    sector_label_map = build_short_label_map(sector_series, prefix="S")
    sector_series_short = apply_index_label_map(sector_series, sector_label_map)
    sector_mapping_box = format_mapping_box(
        sector_label_map, title="Sector code mapping (short = full)"
    )

    plot_multi_fold_bar(
        fold_series=sector_series_short,
        title_prefix='Sector Distribution by Fold (Train vs Test, excluding "Other")',
        output_path=summary_dir / "sector_distribution_all_folds.svg",
        xlabel="Sector (short code)",
        ylabel="Percentage of Documents (%)",
        mapping_box_text=sector_mapping_box,
    )

    plot_country_maps_all_folds(
        fold_data=fold_data,
        output_path=summary_dir / "country_distribution_all_folds.svg",
    )

    print(f"Saved all summary figures to {summary_dir}")


if __name__ == "__main__":
    main()
