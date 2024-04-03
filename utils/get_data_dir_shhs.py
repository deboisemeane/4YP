from pathlib import Path

"""
# Function to get the data directory for processed data, depending on parameters such as data type,
# Whether artefact rejection was used, whether filtering was applied.

Args:
    data_type (str): defines whether the data is frequency based eeg features ("f"), time series eeg ("t"), 
        thorax rip ("rip"), ecg ("ecg"), both ("ecg_rip"), or thorax rip and heart rate ("rip_hr")
    art_rejection (bool): whether or not Yasa's std-based artefact rejection for eeg was applied. For cardiorespiratory
        signals this refers to recording trimming/rejection based on pulseox HR. It is assumed to be true and the value not used.
    filtering (bool): whether or not filtering was applied (LPF for EEG and RIP, BPF for ECG). It is assumed to be true
        and not really used for cardiorespiratory signals.
    prec_epochs (int): number of preceeding epochs included with each example as context
    foll_epochs (int): number of following epochs included with each example as context

"""


def get_data_dir_shhs(data_type: str, art_rejection: bool, filtering: bool, prec_epochs: int, foll_epochs: int):
    # Check if we're on the IBME cluster
    if Path('/data/wadh6184/').is_dir():
        root_dir = Path("/data/wadh6184/")
    # Otherwise use local directory for processed data
    else:
        root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data/Processed/shhs/"

    if data_type in ["f", "t"]:
        if data_type == "f":
            data_dir = data_dir / "Frequency_Features/"
            data_dir = data_dir / f"art_rejection_{int(art_rejection)}_lpf_{int(filtering)}"
        elif data_type == "t":
            data_dir = data_dir / "Time_Features/"
            data_dir = data_dir / f"art_rejection_{int(art_rejection)}_lpf_{int(filtering)}_prec{prec_epochs}_foll{foll_epochs}/"

    elif data_type in ["ecg", "rip", "ecg_rip", "rip_hr"]:
        data_dir = data_dir / "Cardiorespiratory" / f"{data_type}_prec{prec_epochs}_foll{foll_epochs}"

    else:
        raise ValueError("Invalid data type.")

    return data_dir
