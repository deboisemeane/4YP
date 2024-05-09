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
        ax.fill_between(epochs, TL_median - 1 * TL_iqr, TL_median + 1 * TL_iqr, alpha=0.2, color='darkblue', label="Training Loss ± 1 IQR")
        #for i in range(len(TLs_array[0])):
        #    ax.plot(epochs, TLs_array[i], color='darkblue', label="Training Loss" if i==0 else None)
        #    ax.plot(epochs, VLs_array[i], color='gold', label="Validation Loss" if i==0 else None)
        # Plot validation losses
        ax.plot(epochs, VL_median, label='Validation Loss', color='gold')
        ax.fill_between(epochs, VL_median - 1 * VL_iqr, VL_median + 1 * VL_iqr, alpha=0.2, color='gold', label="Validation Loss ± 1 IQR")

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

    kfoldcv.TLs = [[1.3791443633933678, 0.799503743648529, 0.7095373272895813, 0.6785362362861633, 0.6563223004341125, 0.638363778591156, 0.6223539710044861, 0.6071639060974121], [1.4038314239913274, 0.8032811284065247, 0.7088157534599304, 0.6766912937164307, 0.6546294689178467, 0.6365655064582825, 0.6202941536903381, 0.6049183011054993], [1.385243077287263, 0.8011897206306458, 0.7106067538261414, 0.6792581081390381, 0.6573066115379333, 0.6390445828437805, 0.6229442358016968, 0.6076628565788269], [1.4076637089407373, 0.7938255667686462, 0.7048630118370056, 0.6735497117042542, 0.6521168947219849, 0.633781373500824, 0.6182246804237366, 0.6033772826194763], [1.4000444611469975, 0.796575665473938, 0.7069197297096252, 0.6761582493782043, 0.6543352603912354, 0.6362958550453186, 0.6202667951583862, 0.605073869228363], [1.3820307215357954, 0.8013017177581787, 0.7111063599586487, 0.67934250831604, 0.6579042077064514, 0.6397684216499329, 0.6235922574996948, 0.6083978414535522], [1.440706298161841, 0.7979435920715332, 0.7079405784606934, 0.676834762096405, 0.6547000408172607, 0.636396050453186, 0.6199188828468323, 0.6052637100219727], [1.405732489627583, 0.7995897531509399, 0.7082161903381348, 0.6772570013999939, 0.6551701426506042, 0.6367843151092529, 0.6208356022834778, 0.6057689189910889], [1.3820844471010327, 0.7990466952323914, 0.7091193199157715, 0.6770088076591492, 0.6548440456390381, 0.6365821957588196, 0.6206572651863098, 0.605350136756897], [1.3943125515338588, 0.7962278723716736, 0.7065858244895935, 0.6751255393028259, 0.6533746123313904, 0.6350502967834473, 0.6186179518699646, 0.6037407517433167]]
    kfoldcv.VLs = [[1.3680905345370005, 0.7426742836840701, 0.7209637655202437, 0.7104345413394745, 0.6993579858360967, 0.6956063531645604, 0.6978125921406917, 0.7064047585325699], [1.4110635924254042, 0.7605196765190932, 0.7169741401244495, 0.7148125895390561, 0.6922287150186113, 0.6959428007438623, 0.6935497685760947, 0.7059340143781664], [1.383293136311373, 0.7414867252964361, 0.707426254504869, 0.7134892997750888, 0.6896162372813224, 0.688973164569015, 0.6970346636587451, 0.7008581612146134], [1.4177049118941105, 0.7349382266810045, 0.7196395151154212, 0.7082554510434234, 0.6972404019498866, 0.7013374503231369, 0.705626342110836, 0.7027538892746823], [1.410156310840784, 0.7473789184521168, 0.7143222739724098, 0.7091519970151624, 0.7057497856242052, 0.7111228596587337, 0.696704034888566, 0.700729581107623], [1.3735177442058477, 0.7495184924616256, 0.7094008659618073, 0.6986256928749882, 0.6963732288476789, 0.6978500678127048, 0.6931668709889842, 0.6984400017095308], [1.4541910840800039, 0.7361120083955831, 0.7203786545945664, 0.7041596578913786, 0.7139915090200244, 0.6950499903379965, 0.6964383739234028, 0.6998193570402809], [1.4159643139827807, 0.7459684775423351, 0.7219192432733064, 0.6943966887270989, 0.7073651524025139, 0.7112540277440876, 0.7040677863254011, 0.7053962647763032], [1.3781753571031767, 0.7539744458959586, 0.7263211891691215, 0.7053045909266589, 0.7005629027261194, 0.7062016049071215, 0.7112981291255872, 0.7040095648565556], [1.3951467334730367, 0.7359957203847252, 0.7123301721948888, 0.6909317983970046, 0.689095676271277, 0.7035469136162817, 0.6941779817063564, 0.7095976703328649]]
    kfoldcv.plot_loss("RIP_HR.png", "10-Fold Cross Validation of Best-Performing Cardiorespiratory Model")
