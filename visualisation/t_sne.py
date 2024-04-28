import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from scripts.train import DataConfig, SHHSConfig
from utils import get_data_dir_shhs
from src.datasets import SHHSDataset_t
from src.models import Sors_rip_hr
from pathlib import Path


class TSNE_Visualiser:

    def __init__(self, nsrrids: list, data_dir: str, device: torch.device, model):
        self.device = device
        self.model = model().to(device)
        self.data_dir = data_dir
        self.root_dir = Path(__file__).parent.parent
        self.nsrrids = nsrrids
        self.features = []
        self.labels = []
        self.tsne_results = None

    def get_features_hook(self, module, input, output):
        self.features.append(input[0].detach())  # Modules take tuples as input, so we need to extract the input tensor within, and detach it from the graph.

    def get_features(self):
        dataset = SHHSDataset_t(nsrrids=self.nsrrids, data_dir=self.data_dir)
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        model_state_dict = torch.load(self.root_dir / f"model_checkpoints/{self.model.__class__.__name__}.pt",
                                      map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        hook = self.model.fc1.register_forward_hook(self.get_features_hook)
        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                x = batch["features"].to(self.device)
                self.labels.extend(batch["label"].detach().cpu().numpy())
                output = self.model(x)
        hook.remove()

    def perform_tsne(self):
        tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=1000)
        self.tsne_results = tsne.fit_transform(torch.cat(self.features).cpu().numpy())

    def plot_embeddings(self, title: str=None):
        # Convert labels to an array for easy indexing
        labels = np.array(self.labels)
        classes = ["N3", "N1/N2", "REM", "W"]
        colours = ["yellow", "springgreen", "cornflowerblue", "indigo"]
        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        # Plot each class with its own color and label
        for i in np.unique(labels):
            subset = self.tsne_results[labels == i]
            plt.scatter(subset[:, 0], subset[:, 1], c=colours[i], label=f'{classes[i]}', alpha=0.5)

        # Add legend to the plot
        plt.legend( fontsize='12')
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_data_dir_shhs(data_type="rip_hr", art_rejection=True, filtering=True, prec_epochs=2, foll_epochs=1)
    nsrrids = [200053, 200306]#[200029, 200306, 200053]
    tsne_vis = TSNE_Visualiser(nsrrids, data_dir, device, Sors_rip_hr)
    tsne_vis.get_features()
    tsne_vis.perform_tsne()
    tsne_vis.plot_embeddings()

