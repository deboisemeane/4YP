import os
import numpy as np
import torch
from scripts import Train
from scripts.train import DataConfig, OptimiserConfig, AdamConfig
from src.models import Sors
from utils import get_data_dir_shhs
import matplotlib.pyplot as plt
from pathlib import Path


class KFoldConfig(DataConfig):

    def __init__(self, k: int, split: dict, data_type: str, prec_epochs: int, foll_epochs: int,
                 equal_split: str, art_rejection: bool = True,
                 filtering: bool = True, resample: dict = None, **kwargs):
        """

        :param k: Number of folds for cross validation.
        :param split: dict containing proportions of total data for train, test. n_val is determined by n_total / k
        :param data_type: string determining signal(s) to be used.
        :param prec_epochs: context - number of preceeding 30s epochs.
        :param foll_epochs: context - number of following 30s epochs.
        :param equal_split: "train", "val", or "test" - which of these is split equally across folds such that examples are visited exactly once.
                            Examples for the other two datasets are taken from the remaining examples in that fold and divided according to the two relevant values in the split param.
        :param art_rejection: bool indicating artefact rejection.
        :param filtering: bool indicating filtering.
        :param resample: dict indicating resampling.
        :param kwargs:
        """
        super().__init__(k=k, resample=resample, data_type=data_type, split=split, art_rejection=art_rejection,
                         equal_split=equal_split, lpf=filtering,
                         prec_epochs=prec_epochs, foll_epochs=foll_epochs, **kwargs)
        # Check equal split
        assert equal_split in ["train", "val", "test"], "Invalid option for equal split across folds."

        # Get processed recordings that meet our desired preprocessing.
        data_dir = get_data_dir_shhs(data_type=data_type, art_rejection=art_rejection, filtering=filtering,
                                     prec_epochs=prec_epochs, foll_epochs=foll_epochs)
        # Read all filenames and extract nsrrids
        all_filenames = os.listdir(data_dir)
        all_nsrrids = [filename.split('.')[0][7::] for filename in all_filenames]
        n_total = len(all_nsrrids)

        # Generate the splits
        self.all_nsrrids = all_nsrrids
        self.splits = []
        print(f"Data_dir: {data_dir}")
        print(f"all_nsrrids: {all_nsrrids}")
        np.random.shuffle(self.all_nsrrids)  # Shuffle the data to ensure randomness

        segment_length = len(self.all_nsrrids) // k
        for fold in range(k):

            # These indeces will be for the data that is split equally across folds
            start_equal = fold * segment_length
            end_equal = (start_equal + segment_length) if fold < k - 1 else n_total
            equal_nsrrids = self.all_nsrrids[start_equal:end_equal]

            # The remaining nsrrids will be split according to the param argument
            other_nsrrids = self.all_nsrrids[:start_equal] + self.all_nsrrids[end_equal:]

            if equal_split == "train":
                split_train = equal_nsrrids
                n_test = int(np.floor(n_total * split["test"]))
                split_test = other_nsrrids[:n_test]
                split_val = other_nsrrids[n_test:]
            elif equal_split == "test":
                split_test = equal_nsrrids
                n_train = int(np.floor(n_total * split["train"]))
                split_train = other_nsrrids[:n_train]
                split_val = other_nsrrids[n_train:]
            else:  # equal_split == "val"
                split_val = equal_nsrrids
                n_train = int(np.floor(n_total * split["train"]))
                split_train = other_nsrrids[:n_train]
                split_test = other_nsrrids[n_train:]

            patients = {"train": split_train, "val": split_val, "test": split_test}
            self.splits.append(patients)

    def set_patients(self, fold):
        self.params.update({"patients": self.splits[fold]})


class KFold_CV:

    def __init__(self, data_config: KFoldConfig, optimiser_config: OptimiserConfig,
                 device: torch.device, model):
        # Training objects
        self.data_config = data_config
        self.optimiser_config = optimiser_config
        self.device = device
        self.model = model
        self.trainer = None
        # Inference objects
        self.TLs = []
        self.VLs = []
        self.ACCs = []
        self.Kappas = []
        self.confusion = np.zeros((4, 4))  # Total confusion matrix across all tests.
        self.best_accuracy = 0
        self.best_model_state = None # Save this for
        self.best_fold = 0

    def cross_validate(self, n_epochs: int, print_losses=True, weight_losses=True, weight_scalings=None):

        # Iterate over folds
        for fold in range(self.data_config.params["k"]):
            print(f"Training fold {fold+1}")
            # Iterate training split
            self.data_config.set_patients(fold)
            # Reset Trainer
            self.trainer = Train(data_config=self.data_config, optimiser_config=self.optimiser_config,
                                 device=self.device, model=self.model)
            # Train
            self.trainer.train(n_epochs=n_epochs, print_losses=print_losses, weight_losses=weight_losses, weight_scalings=weight_scalings)
            self.TLs.append(self.trainer.TL)
            self.VLs.append(self.trainer.VL)
            # Test
            self.trainer.test()
            self.confusion += self.trainer.confusion
            self.ACCs.append(self.trainer.metrics["Total ACC"])
            self.Kappas.append(self.trainer.metrics["Total Kappa"])
            if self.trainer.metrics["Total ACC"] > self.best_accuracy:
                self.best_accuracy = self.trainer.metrics["Total ACC"]
                self.best_model_state = self.trainer.best_model_state
                self.best_fold = fold
        # Save the best model overall
        self.trainer.best_model_state = self.best_model_state
        self.trainer.save_best_model()
        # Print Results
        print(f"TLs: {self.TLs}")
        print(f"VLs: {self.VLs}")
        print(f"Total Confusion: {self.confusion}")
        print(f"Accuracies: {self.ACCs}")
        print(f"Kappas: {self.Kappas}")
        print(f"Best fold: {self.best_fold+1}")

    def plot_loss(self, figure_dir: str, title: str):
        fig, ax = plt.subplots()

        # Convert lists of losses to numpy arrays for easier manipulation
        TLs_array = np.array(self.TLs)
        VLs_array = np.array(self.VLs)

        # Calculate median and IQR for training and validation losses
        TL_median = np.median(TLs_array, axis=0)
        VL_median = np.median(VLs_array, axis=0)
        TL_iqr = np.subtract(*np.percentile(TLs_array, [75, 25], axis=0))
        VL_iqr = np.subtract(*np.percentile(VLs_array, [75, 25], axis=0))

        epochs = np.arange(len(TL_median))

        # Plot training losses
        ax.plot(epochs, TL_median, label='Training Loss', color='darkblue')
        ax.fill_between(epochs, TL_median - 0.5 * TL_iqr, TL_median + 0.5 * TL_iqr, alpha=0.2, color='darkblue', label="Training Loss IQR")

        # Plot validation losses
        ax.plot(epochs, VL_median, label='Validation Loss', color='gold')
        ax.fill_between(epochs, VL_median - 0.5 * VL_iqr, VL_median + 0.5 * VL_iqr, alpha=0.2, color='gold', label="Validation Loss IQR")

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cross Entropy Loss')
        ax.set_title(title)
        ax.legend()
        plt.show()
        plt.savefig(figure_dir)


if __name__ == "__main__":
    k = 5
    split = {"train": 0.7, "val": 0.2, "test": 0.1}
    data_type = "t"
    prec_epochs = 2
    foll_epochs = 1

    kfold = KFoldConfig(k=k, data_type=data_type, split=split, equal_split="val", prec_epochs=prec_epochs, foll_epochs=foll_epochs)
    kfoldcv = KFold_CV(data_config=kfold, optimiser_config=AdamConfig(lr=0.0003), device=torch.device("cpu"), model=Sors)

    # Generate random loss data for testing the plot
    # Assuming there are 5 folds (k=5) and 12 epochs

    num_folds = 5
    num_epochs = 12

    # Random training and validation losses for each fold and each epoch
    # Generating losses between 0.0 and 1.0 as an example
    kfoldcv.TLs = [[0.7965220808982849, 0.7110756635665894, 0.6814250946044922, 0.6605067253112793, 0.6435421109199524, 0.6285405158996582, 0.6140121817588806, 0.5998182892799377, 0.5865230560302734, 0.5729584693908691, 0.5592545866966248, 0.5455743670463562], [0.7980103492736816, 0.7123120427131653, 0.6823107600212097, 0.6611058712005615, 0.6429933309555054, 0.6273648142814636, 0.6123096942901611, 0.59827721118927, 0.583893895149231, 0.5701468586921692, 0.5565385222434998, 0.5428838133811951], [0.7968530058860779, 0.7119198441505432, 0.6815794110298157, 0.6600721478462219, 0.64205402135849, 0.6265585422515869, 0.6115773916244507, 0.5973025560379028, 0.5837187767028809, 0.5693784356117249, 0.5555372834205627, 0.5422414541244507], [0.7941337823867798, 0.7079556584358215, 0.6775379180908203, 0.6562120318412781, 0.6384527683258057, 0.6226570010185242, 0.6078801155090332, 0.593803882598877, 0.5806005597114563, 0.566933810710907, 0.552931010723114, 0.5393954515457153], [0.7933440208435059, 0.7074292302131653, 0.6775052547454834, 0.6559755802154541, 0.6382274031639099, 0.6229872107505798, 0.6082162261009216, 0.5939199924468994, 0.5801597237586975, 0.5665446519851685, 0.552903950214386, 0.5392047762870789]]
    kfoldcv.VLs = [[0.7338491473020041, 0.7012541954236795, 0.6945738991700622, 0.6855948360307486, 0.6855291316594477, 0.6816213422408863, 0.6931917637264893, 0.6973846424531908, 0.6964171811234644, 0.7046131181420456, 0.712372197769424, 0.7209450669908563], [0.7370203118941973, 0.7094785457910355, 0.6961061576768026, 0.6830192536898064, 0.686644112476143, 0.6838474728464595, 0.6901782941135352, 0.690160656295298, 0.7008524451182575, 0.6953958510126494, 0.7199457082948246, 0.7207355092927704], [0.7431338250354128, 0.7082289167855398, 0.6876139433095491, 0.6994826902758655, 0.6907321584402806, 0.68759430338861, 0.6992447340653706, 0.7036193865644034, 0.7064316608045889, 0.7132285217424972, 0.712698203152777, 0.726954737777807], [0.7456764320188237, 0.736413237407843, 0.7383419867679648, 0.7142175121788922, 0.7037317266019896, 0.7146432782133415, 0.7245779718979082, 0.7202664250161219, 0.7241034779357252, 0.731744233646939, 0.7479552978327809, 0.7484626972985433], [0.7429402230172322, 0.7195928410179112, 0.6963903936926961, 0.6941345185118808, 0.6889236168742384, 0.6947420859840548, 0.6968127704285783, 0.7009895013507251, 0.700821274316107, 0.7120112539265718, 0.7153186503116282, 0.7311711760364021]]
    kfoldcv.plot_loss("RIP_HR_CV.png", "RIP-HR 5-Fold Cross Validation")
