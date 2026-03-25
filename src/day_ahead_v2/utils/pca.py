import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from day_ahead_v2.utils.plot_settings import apply_plot_settings, color_palette_1

logger = logging.getLogger(__name__)


def apply_pca(features: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Reduce dimensionality of standardized numeric features using PCA.

    Args:
        features: DataFrame with the preprocessed features.
        n_components: Number of principal components to retain.

    Returns:
        Transformed feature set with principal components.
    """
    if features.empty:
        raise ValueError("Features DataFrame is empty.")

    # Fit PCA if not already fitted
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)

    # Create a DataFrame with principal components
    pca_df = pd.DataFrame(
        reduced,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=features.index,
    )

    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(
        f"PCA reduced features to {n_components} components "
        f"(explaining {explained_var:.2%} of variance)"
    )

    return pca_df

def plot_label_distribution(y_pred: pd.Series) -> None:
    """
    Plot the distribution of predicted labels.

    Args:
        y_pred: Series with predicted labels.
    """
    apply_plot_settings()
    label_counts = y_pred.value_counts().sort_index()
    labels = label_counts.index.tolist()
    counts = label_counts.values.tolist()

    fig, ax = plt.subplots()
    bars = ax.bar(labels, counts, color=[
        color_palette_1["reddish_purple"],
        color_palette_1["orange"],
        color_palette_1["bluish_green"]
    ])

    # Add counts above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xticks(labels)
    ax.set_xticklabels(['-1 (Deficit)', '0 (Uncertain)', '1 (Surplus)'])
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predicted Imbalance Directions')

    # Save figure
    project_root = Path(__file__).resolve().parents[3]
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "predicted_label_distribution.pdf",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def plot_decision_boundary(X_pca: pd.DataFrame,
                            y_pred: pd.Series,
                            y_true: pd.Series,
                            show_misclassified: bool = True,
                            y_magnitude: pd.Series = None,
                            save_path: Path = None,
                            uncertain_label: int = 0,
                            label_mapping: dict = None) -> None:
    """
    Visualize decision boundary in PCA-reduced feature space.

    Args:
        X_pca: DataFrame with PCA-reduced features (2 components).
        y_pred: Series with predicted labels.
        y_true: Series with true labels.
        show_misclassified: Whether to highlight misclassified points.
        y_magnitude: Series with imbalance magnitudes for marker sizing.
        save_path: Path to save the plot.
        label_mapping: Dictionary mapping label values to strings for plotting.
    """
    apply_plot_settings()
    fig, ax = plt.subplots()
    # Inner-align all on a common index
    common_index = X_pca.index.intersection(y_pred.index).intersection(y_true.index)
    if y_magnitude is not None:
        common_index = common_index.intersection(y_magnitude.index)
    X_pca = X_pca.loc[common_index]
    preds_aligned = y_pred.loc[common_index]
    true_aligned = y_true.loc[common_index]

    # Remap labels to {-1, 0, 1} so uncertain is always in the middle
    if label_mapping is not None:
        _role_to_plot = {"negative": -1, "uncertain": 0, "positive": 1}
        _value_to_plot = {orig: _role_to_plot[role] for orig, role in label_mapping.items()}
        preds_aligned = preds_aligned.map(_value_to_plot)
        true_aligned = true_aligned.map(_value_to_plot)
        uncertain_label = 0
        _role_to_orig = {role: orig for orig, role in label_mapping.items()}
        cbar_ticks = [-1, 0, 1]
        cbar_labels = [
            f"Negative ({_role_to_orig['negative']})",
            f"Uncertain ({_role_to_orig['uncertain']})",
            f"Positive ({_role_to_orig['positive']})",
        ]
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    else:
        cbar_ticks = [-0.5, 0.5, 1.5, 2.5]
        cbar_labels = None
        norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)

    # Colors: negative=reddish_purple, uncertain=orange, positive=bluish_green
    cmap_colors = [color_palette_1["reddish_purple"],
                    color_palette_1["orange"],
                    color_palette_1["bluish_green"]]
    custom_cmap = ListedColormap(cmap_colors)

    # Identify mismatches (excluding uncertain predictions)
    mismatch = (preds_aligned != true_aligned) & (preds_aligned != uncertain_label)
    n_mismatch = int(mismatch.sum())
    n_total = int((preds_aligned != uncertain_label).sum())
    n_correct = n_total - n_mismatch
    accuracy = n_correct / n_total if n_total > 0 else 0

    # Separate masks: uncertain, correct non-uncertain, misclassified non-uncertain
    uncertain_mask = preds_aligned == uncertain_label
    correct_mask = (~mismatch) & (~uncertain_mask)

    # Calculate marker sizes based on magnitude if provided
    if y_magnitude is not None:
        # Align magnitude with predictions
        magnitude_aligned = y_magnitude.loc[common_index].abs()
        # Scale to reasonable marker size range (10 to 200)
        min_size, max_size = 10, 200
        mag_min, mag_max = magnitude_aligned.min(), magnitude_aligned.max()
        if mag_max > mag_min:
            sizes = min_size + (magnitude_aligned - mag_min) / \
                    (mag_max - mag_min) * (max_size - min_size)
        else:
            sizes = pd.Series(50, index=magnitude_aligned.index)
    else:
        sizes = None

    # Plot based on show_misclassified flag
    if show_misclassified:
        # Plot uncertain predictions as normal dots
        if uncertain_mask.any():
            sc_correct = ax.scatter(
                X_pca.loc[uncertain_mask].iloc[:, 0],
                X_pca.loc[uncertain_mask].iloc[:, 1],
                c=preds_aligned[uncertain_mask],
                cmap=custom_cmap,
                norm=norm,
                alpha=0.6,
                s=sizes[uncertain_mask] if sizes is not None else 50,
                edgecolors="white",
                linewidths=0.5,
            )

        # Plot correct non-uncertain predictions as normal dots
        if correct_mask.any():
            sc_correct = ax.scatter(
                X_pca.loc[correct_mask].iloc[:, 0],
                X_pca.loc[correct_mask].iloc[:, 1],
                c=preds_aligned[correct_mask],
                cmap=custom_cmap,
                norm=norm,
                alpha=0.6,
                s=sizes[correct_mask] if sizes is not None else 50,
                edgecolors="white",
                linewidths=0.5,
                label="Correct",
            )

        # Plot mismatches (non-uncertain only) with prominent marker
        if mismatch.any():
            sc_mismatch = ax.scatter(
                X_pca.loc[mismatch].iloc[:, 0],
                X_pca.loc[mismatch].iloc[:, 1],
                c=preds_aligned[mismatch],
                cmap=custom_cmap,
                norm=norm,
                alpha=0.9,
                s=sizes[mismatch] if sizes is not None else 80,
                edgecolors="black",
                linewidths=1,
                marker="X",
                label=f"Misclassified ({n_mismatch})",
            )
    else:
        # Plot all predictions together without distinguishing misclassified
        sc_all = ax.scatter(
            X_pca.iloc[:, 0],
            X_pca.iloc[:, 1],
            c=preds_aligned,
            cmap=custom_cmap,
            norm=norm,
            alpha=0.7,
            s=sizes if sizes is not None else 50,
            edgecolors="white",
            linewidths=0.5,
        )

    # Title with accuracy
    title = (
        r"Decision Boundary in PCA Space "
        rf"(Accuracy={accuracy:.1%})"
    )
    ax.set_title(title, pad=15)

    # Colorbar with custom labels
    if show_misclassified:
        if correct_mask.any():
            scatter_ref = sc_correct
        elif mismatch.any():
            scatter_ref = sc_mismatch
        else:
            scatter_ref = None
    else:
        scatter_ref = sc_all

    if scatter_ref is None:
        return
    cbar = plt.colorbar(
        scatter_ref,
        ax=ax,
        ticks=cbar_ticks,
    )
    if cbar_labels is not None:
        cbar.ax.set_yticklabels(cbar_labels)

    # Custom legend - only show if misclassified points are displayed
    if show_misclassified:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='gray',
                    markersize=8, linestyle='None',
                    label=f'Correctly Classified ({n_correct})'),
            Line2D([0], [0], marker='X', color='w',
                    markerfacecolor='gray',
                    markeredgecolor='black', markersize=8,
                    linestyle='None',
                    label=f'Misclassified ({n_mismatch})')
        ]
        ax.legend(handles=legend_handles, loc="best",
                    framealpha=0.9, fontsize=10)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set axis labels
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')

    # Save figure
    project_root = Path(__file__).resolve().parents[3]
    save_path = project_root / save_path if save_path is not None else project_root / "reports"/ "figures"
    save_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        save_path / "decision_boundary.pdf",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
