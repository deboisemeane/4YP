from pathlib import Path
"""
# Function to get the data directory for processed data, depending on parameters such as data type,
# Whether artefact rejection was used, whether filtering was applied.

Args:
    data_type (str): defines whether the data is frequency based eeg features ("f"), time series eeg ("t")
    art_rejection (bool): whether or not Yasa's std-based artefact rejection was applied
    lpf (bool): whether or not low pass filtering was applied
    prec_epochs (int): number of preceeding epochs included with each example as context
    foll_epochs (int): number of following epochs included with each example as context

"""


def get_data_dir_shhs(data_type: str, art_rejection: bool, lpf: bool, prec_epochs: int, foll_epochs: int):

    assert data_type in ["f", "t"], "Data type should be f or t."

    root_dir = Path(__file__).parent.parent

    data_dir = root_dir / "data/Processed/shhs/"
    if data_type == "f":
        data_dir = data_dir / "Frequency_Features/"
    elif data_type == "t":
        data_dir = data_dir / "Time_Features/"

    data_dir = data_dir / f"art_rejection_{int(art_rejection)}_lpf_{int(lpf)}_prec{prec_epochs}_foll{foll_epochs}/"

    return data_dir
