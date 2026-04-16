"""
Visualization Tools for CluSTAR Analysis.
Reservoir state trajectories, cluster activations, predictions, t-SNE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from sklearn.manifold import TSNE
import seaborn as sns


class CluSTARVisualizer:
    """
    Visualization toolkit for CluSTAR reservoir dynamics and predictions.
    """

    def __init__(self, reservoir: torch.nn.Module, action_names: List[str] = None):
        self.reservoir = reservoir
        self.action_names = action_names or [
            "moving",
            "spinning",
            "collision",
            "stationary",
        ]
        self.cluster_assignments = reservoir.cluster_assignments.cpu().numpy()
        self.num_clusters = reservoir.num_clusters

    def plot_reservoir_dynamics(
        self,
        states: torch.Tensor,
        sequence_idx: int = 0,
        max_neurons: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Plot reservoir state trajectories over time for selected neurons.

        Args:
            states: [B, T, N] reservoir states
            sequence_idx: which sequence to visualize
            max_neurons: plot at most N neuron trajectories
        """
        seq_states = states[sequence_idx].cpu().numpy()  # [T, N]
        T, N = seq_states.shape

        # Select neurons to plot (first from each cluster)
        neurons_to_plot = []
        for c in range(self.num_clusters):
            cluster_neurons = np.where(self.cluster_assignments == c)[0]
            if len(cluster_neurons) > 0:
                neurons_to_plot.append(cluster_neurons[0])  # first neuron in cluster
        neurons_to_plot = neurons_to_plot[:max_neurons]

        plt.figure(figsize=(12, 6))
        time = np.arange(T)
        for i, neuron_idx in enumerate(neurons_to_plot):
            cluster_id = self.cluster_assignments[neuron_idx]
            plt.plot(
                time,
                seq_states[:, neuron_idx],
                label=f"Cluster {cluster_id}",
                alpha=0.7,
            )

        plt.xlabel("Time Step")
        plt.ylabel("Activation")
        plt.title(f"Reservoir Neuron Trajectories (Sequence {sequence_idx})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_cluster_activity(
        self, states: torch.Tensor, save_path: Optional[str] = None
    ):
        """
        Plot average activation per cluster over time.
        """
        # states: [B, T, N]
        states_np = states.mean(dim=0).cpu().numpy()  # [T, N] (avg over batch)

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

    def plot_tsne_by_action(
        self,
        states: torch.Tensor,
        labels: torch.Tensor,
        perplexity: int = 30,
        save_path: Optional[str] = None,
    ):
        """
        t-SNE of reservoir states colored by action class.
        """
        B = states.shape[0]
        # Use last timestep state for each sequence
        states_agg = states[:, -1, :].cpu().numpy()  # [B, N]
        labels_np = labels.cpu().numpy()

        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        states_2d = tsne.fit_transform(states_agg)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            states_2d[:, 0], states_2d[:, 1], c=labels_np, cmap="tab10", alpha=0.6, s=20
        )
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=self.action_names,
            title="Action",
        )
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title("Reservoir State Embeddings by Action Label")
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_frame_predictions(
        self,
        frames_true: torch.Tensor,
        frames_pred: torch.Tensor,
        num_examples: int = 5,
        save_path: Optional[str] = None,
    ):
        """
        Show true vs predicted frames side-by-side.
        """
        B = min(num_examples, frames_true.shape[0])
        fig, axes = plt.subplots(B, 3, figsize=(8, B * 2.5))

        if B == 1:
            axes = axes.reshape(1, -1)

        for i in range(B):
            # Show true frame at t
            axes[i, 0].imshow(frames_true[i, 0, 0].cpu(), cmap="gray")
            axes[i, 0].set_title("True (t)")
            axes[i, 0].axis("off")

            # Show true frame at t+1
            axes[i, 1].imshow(frames_true[i, 1, 0].cpu(), cmap="gray")
            axes[i, 1].set_title("True (t+1)")
            axes[i, 1].axis("off")

            # Show predicted frame
            axes[i, 2].imshow(frames_pred[i, 0].cpu().reshape(64, 64), cmap="gray")
            axes[i, 2].set_title("Predicted")
            axes[i, 2].axis("off")

        plt.suptitle("Frame Prediction Examples")
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
        """
        states_np = states.mean(dim=1).cpu().numpy()  # [B, N] (avg over time)
        labels_np = labels.cpu().numpy()

        cluster_means_per_class = np.zeros((self.num_clusters, len(self.action_names)))

        for c in range(self.num_clusters):
            mask = self.cluster_assignments == c
            if mask.sum() > 0:
                cluster_states = states_np[:, mask].mean(axis=1)  # [B]
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
        plt.title("Cluster Activation by Action (averaged over sequences)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_all_visualizations(
        self,
        states: torch.Tensor,
        frames: torch.Tensor,
        labels: torch.Tensor,
        preds: Optional[torch.Tensor] = None,
        pred_frames: Optional[torch.Tensor] = None,
        output_dir: str = "visualizations",
    ):
        """
        Generate and save all diagnostic plots.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating visualizations...")

        # 1. t-SNE colored by action
        self.plot_tsne_by_action(
            states, labels, save_path=output_dir / "tsne_actions.png"
        )

        # 2. Cluster activity over time (avg over batch)
        self.plot_cluster_activity(
            states, save_path=output_dir / "cluster_activity.png"
        )

        # 3. Sample neuron trajectories
        self.plot_reservoir_dynamics(
            states, sequence_idx=0, save_path=output_dir / "neuron_trajectories.png"
        )

        # 4. Cluster activation heatmap
        self.plot_cluster_correlations(
            states, labels, save_path=output_dir / "cluster_action_heatmap.png"
        )

        # 5. Frame predictions (if available)
        if pred_frames is not None:
            self.plot_frame_predictions(
                frames[:, :2],
                pred_frames[:5],
                save_path=output_dir / "frame_predictions.png",
            )

        print(f"Visualizations saved to {output_dir}/")


def test_visualizer():
    """Quick test."""
    from models.reservoir import ClusteredReservoir
    from models.encoder import SpatialEncoder

    reservoir = ClusteredReservoir(
        input_dim=128, reservoir_size=500, num_clusters=5, seed=42
    )
    encoder = SpatialEncoder(output_dim=128)

    # Dummy data
    B, T, N = 10, 30, 500
    states = torch.randn(B, T, N)
    labels = torch.randint(0, 4, (B,))

    viz = CluSTARVisualizer(reservoir)
    viz.plot_tsne_by_action(states, labels, save_path="test_tsne.png")
    viz.plot_reservoir_dynamics(states, save_path="test_trajectories.png")
    viz.plot_cluster_activity(states, save_path="test_cluster_activity.png")

    print("Visualizer tests passed!")


if __name__ == "__main__":
    test_visualizer()
