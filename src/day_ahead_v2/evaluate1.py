import pandas as pd
import numpy as np


    def preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before prediction.
        Standardize features and fill NaNs by interpolation.
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty.")

        # Separate numeric and categorical features
        num_features = features.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        cat_features = features.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()
        features_num = features[num_features].copy()
        features_cat = features[cat_features].copy()

        # Numeric: interpolate
        if not features_num.empty:
            features_num = features_num.interpolate(
                method="linear", limit_direction="both"
            )
            features_num = features_num.fillna(features_num.mean())

        # Categorical: fill with mode
        if not features_cat.empty:
            for col in features_cat.columns:
                mode_series = features_cat[col].mode()
                mode_val = (
                    mode_series.iloc[0] if not mode_series.empty else "missing"
                )
                features_cat[col] = features_cat[col].fillna(mode_val)

        # Standardize numeric features
        if not features_num.empty:
            scaler = StandardScaler()
            features_num = pd.DataFrame(
                scaler.fit_transform(features_num),
                columns=num_features,
                index=features.index
            )

        # One-hot encode categorical features
        if not features_cat.empty:
            encoder = OneHotEncoder(drop="first", sparse_output=False)
            encoded = encoder.fit_transform(features_cat)
            cat_columns = encoder.get_feature_names_out(cat_features)
            features_cat = pd.DataFrame(encoded, columns=cat_columns,
                                        index=features.index)

        # Concatenate processed numeric and categorical features
        if not features_num.empty and not features_cat.empty:
            features_processed = pd.concat([features_num, features_cat],
                                           axis=1)
        elif not features_num.empty:
            features_processed = features_num
        else:
            features_processed = features_cat

        return features_processed

    def rfe(self, features: pd.DataFrame,
            target: pd.Series, n_features_to_select: int) -> pd.DataFrame:
        """
        Perform Recursive Feature Elimination (RFE) to select top features.

        Args:
            features: DataFrame with the preprocessed features.
            target: Series with the target labels.
            n_features_to_select: Number of top features to select.

        Returns:
            DataFrame with selected top features.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        if features.empty or target.empty:
            raise ValueError("Features or target DataFrame is empty.")

        rfe = RFE(estimator=self.model,
                  n_features_to_select=n_features_to_select)
        rfe.fit(features, target)

        selected_features = features.columns[rfe.support_]
        print(f"[INFO] Selected top {n_features_to_select} features via RFE.")

        return features[selected_features]

    def apply_pca(self, features: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
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
        self.pca = PCA(n_components=n_components)
        reduced = self.pca.fit_transform(features)

        # Create a DataFrame with principal components
        pca_df = pd.DataFrame(
            reduced,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=features.index,
        )

        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"[INFO] PCA reduced features to {n_components} components "
              f"(explaining {explained_var:.2%} of variance)")

        return pca_df

    def plot_label_distribution(self, y_pred: pd.Series) -> None:
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
        project_root = Path(__file__).resolve().parents[1]
        figures_dir = project_root / "figures"
        figures_dir.mkdir(exist_ok=True)

        plt.tight_layout()
        plt.savefig(
            figures_dir / "predicted_label_distribution.pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    def plot_decision_boundary(self, X_pca: pd.DataFrame,
                               y_pred: pd.Series, y_true: pd.Series,
                               alpha: float, show_misclassified: bool,
                               y_magnitude: pd.Series = None) -> None:
        """
        Visualize decision boundary in PCA-reduced feature space.

        Args:
            X_pca: DataFrame with PCA-reduced features (2 components).
            y_pred: Series with predicted labels.
            y_true: Series with true labels.
            alpha: Confidence threshold used for predictions.
            show_misclassified: Whether to highlight misclassified points.
            y_magnitude: Series with imbalance magnitudes for marker sizing.
        """
        apply_plot_settings()
        fig, ax = plt.subplots()
        # Align predictions with PCA dataframe index
        preds_aligned = y_pred.loc[X_pca.index]
        true_aligned = y_true.loc[X_pca.index]

        # Color normalization: -1 (shortage), 0 (uncertain), 1 (surplus)
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        # Use first 3 colors defined at module level
        cmap_colors = [color_palette_1["reddish_purple"],
                       color_palette_1["orange"],
                       color_palette_1["bluish_green"]]
        custom_cmap = ListedColormap(cmap_colors)

        # Identify mismatches (excluding uncertain predictions)
        mismatch = (preds_aligned != true_aligned) & (preds_aligned != 0)
        n_mismatch = int(mismatch.sum())
        n_total = int((preds_aligned != 0).sum())
        n_correct = n_total - n_mismatch
        accuracy = n_correct / n_total if n_total > 0 else 0

        # Separate correct and incorrect predictions
        correct_mask = ~mismatch

        # Calculate marker sizes based on magnitude if provided
        if y_magnitude is not None:
            # Align magnitude with predictions
            magnitude_aligned = y_magnitude.loc[X_pca.index].abs()
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
            # Plot correct predictions
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

            # Plot mismatches with prominent marker
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
            rf"($\alpha={alpha:.2f}$, Accuracy={accuracy:.1%})"
        )
        ax.set_title(title, pad=15)

        # Colorbar with custom labels
        if show_misclassified:
            scatter_ref = sc_correct if correct_mask.any() else sc_mismatch
        else:
            scatter_ref = sc_all

        cbar = plt.colorbar(
            scatter_ref,
            ax=ax,
            ticks=[-1, 0, 1]
        )
        cbar.ax.set_yticklabels(
            ["Deficit (-1)", "Uncertain (0)", "Surplus (1)"]
        )

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
        project_root = Path(__file__).resolve().parents[1]
        figures_dir = project_root / "figures"
        figures_dir.mkdir(exist_ok=True)

        plt.tight_layout()
        plt.savefig(
            figures_dir / f"decision_boundary_{alpha}.pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    def evaluate_classifier(
        self, y_pred: pd.Series, y_test: pd.Series, y_proba: pd.Series, alpha: float = 0.7) -> Dict[str, Any]:
        """
        Evaluate classifier predictions and plot a confusion matrix.

        Args:
            y_pred: The predicted labels.
            y_test: The true labels.
            y_proba: The predicted probabilities.
            print_metrics: Whether to print the evaluation metrics.

        Raises:
            ValueError: If y_pred or y_test is None.

        Returns:
            A dictionary containing evaluation metrics and the confusion matrix.
        """
        if y_pred is None or y_test is None:
            raise ValueError("y_pred and y_test must not be None")

        # Ensure aligned indices
        y_test = y_test.loc[y_pred.index]

        # Keep only confident predictions (exclude 0)
        mask = y_pred != 0
        if not mask.any():
            raise ValueError(
                "All predictions are 0 (uncertain); nothing to evaluate."
            )
        y_pred_filtered = y_pred[mask]
        y_test_filtered = y_test[mask]

        # Map labels {-1,1} -> {0,1}
        y_pred_bin = y_pred_filtered.replace({-1: 0, 1: 1})
        y_test_bin = y_test_filtered.replace({-1: 0, 1: 1})

        # Metrics
        acc = accuracy_score(y_test_bin, y_pred_bin)
        f1 = f1_score(y_test_bin, y_pred_bin, average="weighted")
        if y_proba is not None:
            auc = roc_auc_score(y_test_bin, y_proba.loc[y_pred_bin.index])
        else:
            auc = None

        #########################################################################
        # Confusion matrix (with original label names for display)
        cm = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["-1", "1"]
        )
        disp.plot(cmap="Blues")
        plt.title(r"Confusion Matrix ($\alpha={}$)".format(alpha))
        plt.tight_layout()
        # Path to project root (parent of src)
        project_root = Path(__file__).resolve().parents[1]
        figures_dir = project_root / "figures"
        figures_dir.mkdir(exist_ok=True)
        plt.savefig(figures_dir / f"confusion_matrix_{alpha}.pdf")
        plt.close()

        #######################################################################
        # ROC curve
        if y_proba is not None:

            # Align scores with evaluated (non-zero) predictions
            y_scores = y_proba.loc[y_pred_bin.index]

            # Compute ROC
            fpr, tpr, _ = roc_curve(y_test_bin, y_scores)

            # Plot ROC
            plt.figure()
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(r"ROC Curve ($\alpha={}$)".format(alpha))
            plt.legend(loc="lower right")
            plt.tight_layout()

            # Save ROC figure next to confusion matrix
            plt.savefig(figures_dir / f"roc_curve_{alpha}.pdf")
            plt.close()


        #######################################################################
        # Compile metrics
        metrics = {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "support": {
                "n_evaluated": int(len(y_test_bin)),
                "n_dropped_uncertain": int((~mask).sum()),
            },
        }

        if print_metrics:
            table = [
                ["Accuracy", f"{acc:.4f}"],
                ["F1 Score", f"{f1:.4f}"],
                ["ROC AUC", f"{auc:.4f}" if auc is not None else "N/A"],
            ]
            print(tabulate(table,
                           headers=["Metric", "Value"],
                           tablefmt="github"))
            print("Confusion Matrix:\n", pd.DataFrame(cm))
            print("Support:", metrics["support"])
        return metrics
