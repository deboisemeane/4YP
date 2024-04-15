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

        plt.savefig(figure_dir)


if __name__ == "__main__":
    k = 5
    split = {"train": 0.7, "val": 0.2, "test": 0.1}
    data_type = "t"
    prec_epochs = 2
    foll_epochs = 1

    kfold = KFoldConfig(k=k, data_type=data_type, split=split, prec_epochs=prec_epochs, foll_epochs=foll_epochs)
    kfoldcv = KFold_CV(data_config=kfold, optimiser_config=AdamConfig(lr=0.0003), device=torch.device("cpu"), model=Sors)

    # Generate random loss data for testing the plot
    # Assuming there are 5 folds (k=5) and 12 epochs

    num_folds = 5
    num_epochs = 12

    # Random training and validation losses for each fold and each epoch
    # Generating losses between 0.0 and 1.0 as an example
    kfoldcv.TLs = [[0.7969807386398315, 0.7007883787155151, 0.6673288941383362, 0.6442959308624268, 0.6252745985984802, 0.6080255508422852, 0.5917069911956787, 0.576477587223053, 0.5617411732673645, 0.5466411113739014, 0.5319624543190002, 0.5167720913887024], [0.8001914620399475, 0.702507734298706, 0.668708324432373, 0.6453641653060913, 0.6265462636947632, 0.6099686026573181, 0.5941790342330933, 0.579436719417572, 0.5649137496948242, 0.5505386590957642, 0.5363479852676392, 0.5211670994758606], [0.7992476224899292, 0.7041249871253967, 0.6703822612762451, 0.6466144919395447, 0.627457320690155, 0.6103995442390442, 0.5942988395690918, 0.5791675448417664, 0.5637961030006409, 0.5494582056999207, 0.5343939065933228, 0.5195146799087524], [0.796255350112915, 0.6994864344596863, 0.6651181578636169, 0.6410764455795288, 0.6217517852783203, 0.6046133637428284, 0.5883994698524475, 0.5730615854263306, 0.5571037530899048, 0.5425118207931519, 0.5269349813461304, 0.5116919279098511], [0.7968116998672485, 0.6985489130020142, 0.6650015711784363, 0.6415765881538391, 0.6218569278717041, 0.6048905849456787, 0.5882777571678162, 0.5728940963745117, 0.5573227405548096, 0.5427756905555725, 0.5274370908737183, 0.5126480460166931]]
    kfoldcv.VLs = [[0.7750759937121899, 0.7296103866163901, 0.7214289239309758, 0.7050453131660352, 0.7293612070727835, 0.7216998970413471, 0.7216065968490087, 0.7192724170482881, 0.7360787813260244, 0.7276118288055647, 0.7541021485818159, 0.7521264828148804], [0.7580005826495692, 0.7336237743633188, 0.7154190650700656, 0.7218191850216119, 0.720518944014674, 0.718234557555616, 0.7204824464731149, 0.725291216430306, 0.7154579485957047, 0.7293114939752622, 0.7395950006661016, 0.7490344808460372], [0.753642979055231, 0.7248822227970376, 0.7069757916808519, 0.7022495805398055, 0.7006906229448995, 0.7023460832883027, 0.7135094631420827, 0.7097050009044534, 0.7157128958307647, 0.7114629325107759, 0.730114741015631, 0.735300558177966], [0.7773853277426052, 0.7359073801105981, 0.7184534597674923, 0.7199465693544052, 0.703351607850964, 0.7086027671753228, 0.7166374011427404, 0.7230792180668809, 0.7351482601808538, 0.726866077233557, 0.7480563138877776, 0.7568564552288104], [0.7420437941190026, 0.7073293824456764, 0.7200673896184626, 0.6969798247469788, 0.6927397451536755, 0.6967509773049153, 0.6980442500663985, 0.7042029173945737, 0.7073726524434426, 0.7204711856466746, 0.7316389685524355, 0.7279766934921434]]
    kfoldcv.plot_loss("RIP_HR_CV.png","RIP-HR 5-Fold Cross Validation")
