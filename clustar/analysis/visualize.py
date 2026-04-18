"""
Visualization Tools for CluSTAR Analysis.
t-SNE + confusion matrix for action classification evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler
from packaging import version


class CluSTARVisualizer:
    """
    Visualization toolkit for CluSTAR reservoir dynamics and predictions.
    """

    def __init__(self, reservoir: torch.nn.Module, action_names: List[str] = None):
        self.reservoir = reservoir
        self.action_names = action_names or [
            "moving",
            "spinning",
            "stationary",
        ]
        self.cluster_assignments = reservoir.cluster_assignments.cpu().numpy()
        self.num_clusters = reservoir.num_clusters

    def plot_tsne_by_action(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        model_accuracy: Optional[float] = None,
        perplexity: int = 50,
        save_path: Optional[str] = None,
    ):
        """
        t-SNE of reservoir features colored by action class.
        Side-by-side: t-SNE scatter (left) + linear probe confusion matrix (right).

        Args:
            features: [B, feature_dim] already aggregated and standardized features
            labels: [B] action labels
            model_accuracy: actual model test accuracy (displayed on plot for context)
            perplexity: t-SNE perplexity
            save_path: where to save plot
        """
        B = features.shape[0]
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        states_scaled = StandardScaler().fit_transform(features_np)

        effective_perplexity = min(perplexity, B - 1)
        if effective_perplexity != perplexity:
            print(
                f"  Note: Adjusting perplexity from {perplexity} to {effective_perplexity} (dataset size={B})"
            )

        print("Running t-SNE...")
        tsne_params = {
            "n_components": 2,
            "perplexity": effective_perplexity,
            "random_state": 42,
        }
        if version.parse(__import__("sklearn").__version__) >= version.parse("0.24"):
            tsne_params["max_iter"] = 1000
        else:
            tsne_params["n_iter"] = 1000
        tsne = TSNE(**tsne_params)
        states_2d = tsne.fit_transform(states_scaled)

        linear_probe = LogisticRegression(max_iter=1000)
        linear_probe.fit(states_2d, labels_np)
        tsne_accuracy = linear_probe.score(states_2d, labels_np)
        sil_score = silhouette_score(states_2d, labels_np)

        preds = linear_probe.predict(states_2d)
        cm = confusion_matrix(labels_np, preds)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # LEFT: t-SNE scatter
        scatter = axes[0].scatter(
            states_2d[:, 0], states_2d[:, 1], c=labels_np, cmap="tab10", alpha=0.6, s=20
        )
        axes[0].set_xlabel("t-SNE Component 1")
        axes[0].set_ylabel("t-SNE Component 2")

        title_lines = [f"2D Probe: {tsne_accuracy:.1%} | Silhouette: {sil_score:.3f}"]
        if model_accuracy is not None:
            title_lines.insert(0, f"Model Acc: {model_accuracy:.1%}")
        axes[0].set_title("t-SNE (per-cluster features)\n" + " | ".join(title_lines))
        axes[0].grid(True, alpha=0.3)

        handles, _ = scatter.legend_elements()
        unique_classes = np.unique(labels_np)
        legend_names = [self.action_names[c] for c in unique_classes]
        axes[0].legend(handles=handles, labels=legend_names, title="Action")

        # RIGHT: Confusion matrix
        axes[1].set_title("2D Linear Probe Confusion Matrix")
        axes[1].set_xlabel("Predicted label")
        axes[1].set_ylabel("True label")
        tick_marks = np.arange(len(self.action_names))
        axes[1].set_xticks(tick_marks)
        axes[1].set_yticks(tick_marks)
        axes[1].set_xticklabels(self.action_names, rotation=45, ha="right")
        axes[1].set_yticklabels(self.action_names)

        im = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                axes[1].text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                )

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved plot to {save_path}")
        plt.close()

    def plot_cluster_activity(
        self, states: torch.Tensor, save_path: Optional[str] = None
    ):
        """
        Plot average activation per cluster over time.
        states: [B, T, N]
        """
        states_np = states.mean(dim=0).cpu().numpy()

        cluster_means = np.zeros((states_np.shape[0], self.num_clusters))
        for c in range(self.num_clusters):
            mask = self.cluster_assignments == c
            if mask.sum() > 0:
                cluster_means[:, c] = states_np[:, mask].mean(axis=1)

        plt.figure(figsize=(10, 5))
        for c in range(self.num_clusters):
            plt.plot(cluster_means[:, c], label=f"Cluster {c}")
        plt.xlabel("Time")
        plt.ylabel("Avg Activation")
        plt.title("Cluster Activity Over Time")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_cluster_correlations(
        self,
        states: torch.Tensor,
        labels: torch.Tensor,
        save_path: Optional[str] = None,
    ):
        """
        Heatmap of average cluster activation per action class.
        states: [B, T, N]
        """
        import seaborn as sns

        states_np = states.mean(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        cluster_means_per_class = np.zeros((self.num_clusters, len(self.action_names)))

        for c in range(self.num_clusters):
            mask = self.cluster_assignments == c
            if mask.sum() > 0:
                cluster_states = states_np[:, mask].mean(axis=1)
                for a in range(len(self.action_names)):
                    class_mask = labels_np == a
                    cluster_means_per_class[c, a] = cluster_states[class_mask].mean()

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cluster_means_per_class,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            xticklabels=self.action_names,
            yticklabels=[f"Cluster {c}" for c in range(self.num_clusters)],
        )
        plt.xlabel("Action Class")
        plt.ylabel("Cluster")
        plt.title("Cluster Activation by Action")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
