import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from scripts.train import DataConfig, SHHSConfig
from utils import get_data_dir_shhs
from src.datasets import SHHSDataset_t
from src.models import Sors_rip_hr, Sors
from pathlib import Path


class TSNE_Visualiser:

    def __init__(self, nsrrids: list, data_dir: str, device: torch.device, model):
        self.device = device
        self.model = model().to(device)
        self.data_dir = data_dir
        self.root_dir = Path(__file__).parent.parent
        self.nsrrids = nsrrids
        self.dataset = SHHSDataset_t(nsrrids=self.nsrrids, data_dir=self.data_dir)
        self.features = []
        self.labels = []
        self.predictions = []
        self.tsne_results = None

    def get_features_hook(self, module, input, output):
        self.features.append(input[0].detach())  # Modules take tuples as input, so we need to extract the input tensor within, and detach it from the graph.

    def get_features(self):
        dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)
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
                self.predictions.extend(output.argmax(dim=1).detach().cpu().numpy())
        hook.remove()

    def perform_tsne(self):
        tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=1000)
        self.tsne_results = tsne.fit_transform(torch.cat(self.features).cpu().numpy())

    def save_embeddings(self, filename: str):
        np.save(f"saved_embeddings/{filename}_embeddings.npy",  self.tsne_results)
        np.save(f"saved_embeddings/{filename}_labels.npy",      self.labels)
        np.save(f"saved_embeddings/{filename}_predictions.npy", self.predictions)

    def load_embeddings(self, filename: str):
        self.tsne_results   = np.load(f"saved_embeddings/{filename}_embeddings.npy")
        self.labels         = np.load(f"saved_embeddings/{filename}_labels.npy")
        self.predictions    = np.load(f"saved_embeddings/{filename}_predictions.npy")

    def plot_embeddings(self, title: str = None):
        # Convert labels to an array for easy indexing
        labels = np.array(self.labels)
        classes = ["N3", "N1/N2", "REM", "W"]
        colours = ['navy', 'skyblue', 'magenta', 'darkorange']
        # Create a scatter plot
        plt.figure(figsize=(5, 4))
        # Plot each class with its own color and label
        for i in np.unique(labels):
            subset = self.tsne_results[labels == i]
            plt.scatter(subset[:, 0], subset[:, 1], c=colours[i], label=f'{classes[i]}', alpha=0.6, s=1)

        # Add legend to the plot
        legend = plt.legend()
        for handle in legend.legend_handles:
            handle._sizes = [6]
        plt.title(title, fontsize=14)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig("C:/Users/Alex/OneDrive/Documents/Uni/Year4/4YP/Latex/4YP Report/figure/t-sne/riphr.png", dpi=300)


    def plot_separately(self, title: str=None):
        # Plots embeddings of 8 recordings separately, with age and sex.
        nsrrids = self.dataset.index_df["nsrrids"]
        unique_ids = pd.unique(nsrrids)  # This gets all unique IDs
        labels = np.array(self.labels)
        predictions = np.array(self.predictions)


        # Create a dictionary of indices for each nsrrid
        indices_dict = {uid: np.where(nsrrids == uid)[0] for uid in unique_ids}

        fig, axs = plt.subplots(4,2, sharex=True, sharey=True)
        fig.set_size_inches(4,7)
        classes = ["N3", "N1/N2", "REM", "W"]
        colours = ['navy', 'skyblue', 'magenta', 'darkorange']
        df = pd.read_csv(Path(self.data_dir).parent.parent.parent.parent / "Raw/shhs/datasets/shhs-harmonized-dataset-0.20.0.csv")
        j = 0
        for nsrrid in unique_ids:
            ax = axs.flatten()[j]
            labels_subset       = labels[indices_dict[nsrrid]]
            predictions_subset  = predictions[indices_dict[nsrrid]]
            for i in np.unique(labels):
                subset = self.tsne_results[indices_dict[nsrrid]][labels_subset == i]
                ax.scatter(subset[:, 0], subset[:, 1], c=colours[i], label=f'{classes[i]}', alpha=1, s=1, linewidths=0)

            _ = df.loc[df['nsrrid'] == nsrrid, ['nsrr_age', 'nsrr_sex']]
            age = int(_['nsrr_age'].iloc[0])
            sex = _['nsrr_sex'].iloc[0]
            acc = np.sum(labels_subset == predictions_subset) / labels_subset.size
            ax.set_title(f"{sex}, {age}, ACC: {np.round(acc, decimals=2)}", fontsize=10)
            j += 1
        #legend = axs.flatten()[0].legend(fontsize=5)
        #for handle in legend.legend_handles:
        #    handle._sizes = [5]
        for a in axs.flatten():
            a.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,
                left=False,
                right=False  # ticks along the top edge are off
                )
            a.set_xticklabels([])
            a.set_yticklabels([])
        fig.supxlabel("t-SNE Dimension 1", y=0.08)
        fig.supylabel("t-SNE Dimension 2", x=0.08)
        plt.savefig("C:/Users/Alex/OneDrive/Documents/Uni/Year4/4YP/Latex/4YP Report/figure/t-sne/riphr8agesex.png", dpi=300)

    def plot_together(self, title: str):
        nsrrids = self.dataset.index_df["nsrrids"]
        unique_ids = pd.unique(nsrrids)  # This gets all unique IDs
        labels = np.array(self.labels)
        predictions = np.array(self.predictions)
        df = pd.read_csv(Path(self.data_dir).parent.parent.parent.parent / "Raw/shhs/datasets/shhs-harmonized-dataset-0.20.0.csv")
        # Create a dictionary of indices for each nsrrid
        indices_dict = {uid: np.where(nsrrids == uid)[0] for uid in unique_ids}

        classes = ["N3", "N1/N2", "REM", "W"]
        colours = ['navy', 'skyblue', 'magenta', 'darkorange']

        fig, ax_dict = plt.subplot_mosaic([['a', 'a'], ['a', 'a'], ['b', 'c'], ['d', 'e'], ['f', 'g'], ['h', 'i']],
                                          layout='constrained', gridspec_kw={"wspace": 0, "hspace": 0})
        fig.set_size_inches(4, 10)

        # Main plot configuration
        a = ax_dict['a']
        for i in np.unique(labels):
            subset = self.tsne_results[labels == i]
            a.scatter(subset[:, 0], subset[:, 1], c=colours[i], label=f'{classes[i]}', alpha=0.6, s=1)
        a.tick_params(axis='both',  # Changes apply to both x and y-axis
                      which='both',  # Apply to both major and minor ticks
                      direction='in',  # 'in', 'out', or 'inout'
                      length=4,  # Length of the ticks
                      width=1,  # Width of the ticks
                      colors='black',  # Color of the ticks, change if needed
                      pad=1,  # Distance between ticks and labels
                        )  # Ticks on the right side

        a.set_title(title, fontsize=14)
        a.set_xlabel('t-SNE Dimension 1', labelpad=1)
        a.set_ylabel('t-SNE Dimension 2', labelpad=1)
        legend = a.legend()
        for handle in legend.legend_handles:
            handle._sizes = [6]

        # Gather axis limits from the main plot
        xlims = a.get_xlim()
        ylims = a.get_ylim()

        # Apply axis limits to all subplots and disable ticks and labels
        subplots = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        for j, key in enumerate(subplots):
            ax = ax_dict[key]
            nsrrid = unique_ids[j % len(unique_ids)]  # Make sure to cycle through IDs if not enough
            labels_subset = labels[indices_dict[nsrrid]]
            predictions_subset = predictions[indices_dict[nsrrid]]
            for i in np.unique(labels_subset):
                subset = self.tsne_results[indices_dict[nsrrid]][labels_subset == i]
                ax.scatter(subset[:, 0], subset[:, 1], c=colours[i], label=f'{classes[i]}', alpha=1, s=1, linewidths=0)
            # Additional info such as age, sex, accuracy
            _ = df.loc[df['nsrrid'] == nsrrid, ['nsrr_age', 'nsrr_sex']]
            age = int(_['nsrr_age'].iloc[0])
            sex = _['nsrr_sex'].iloc[0]
            acc = np.sum(labels_subset == predictions_subset) / labels_subset.size
            ax.set_title(f"{sex}, {age}, ACC: {np.round(acc, decimals=2)}", fontsize=10)
            # Give all subplots same limits and no ticks
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.savefig(f"C:/Users/Alex/OneDrive/Documents/Uni/Year4/4YP/Latex/4YP Report/figure/t-sne/{title}_final.png", dpi=300)




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_data_dir_shhs(data_type="rip_hr", art_rejection=True, filtering=True, prec_epochs=2, foll_epochs=1)
    #nsrrids = [204622, 204904, 201470, 200478] # eeg
    #nsrrids = [200029, 200306, 203728, 204645, 201016, 200644, 201372, 202615]# [201016, 200644, 201372, 202615] # rip_hr#
    #nsrrids = [201761, 202350, 203842, 203870, 203312, 204631, 201463, 201242] # 45, 55, 65, 75 "riphr"
    #nsrrids = [200233, 204702, 204294, 204187, 203545, 204979, 202219, 202062] # 41, 53, 65, 77 "riphr1"
    #nsrrids = [200010, 201979, 205790, 200049, 203602, 203058, 201191, 201539] # 40, 53, 66, 79
    #nsrrids = [204726, 202319, 200046, 204936, 201574, 200876, 202480, 202790] # 40, 53, 65, 80
    nsrrids = [200010, 201979, 204294, 204187, 203312, 204631, 201463, 201242] # final

    tsne_vis = TSNE_Visualiser(nsrrids, data_dir, device, Sors_rip_hr)
    #tsne_vis.get_features()
    #tsne_vis.perform_tsne()
    #tsne_vis.save_embeddings("")
    tsne_vis.load_embeddings("eeg")
    #tsne_vis.plot_embeddings(title="RIP, HR")
    #tsne_vis.plot_separately()
    tsne_vis.plot_together("EEG")


