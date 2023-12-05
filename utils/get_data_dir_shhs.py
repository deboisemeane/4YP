from pathlib import Path


# Function to get the data directory for processed data, depending on parameters such as data type,
# Whether artefact rejection was used, whether filtering was applied.

def get_data_dir_shhs(data_type: str, art_rejection: bool, lpf: bool):

    assert data_type in ["f", "t"], "Data type should be f or t."

    root_dir = Path(__file__).parent.parent

    data_dir = root_dir / "data/Processed/shhs/"
    if data_type == "f":
        data_dir = data_dir / "Frequency_Features/"
    elif data_type == "t":
        data_dir = data_dir / "Time_Features/"

    data_dir = data_dir / f"art_rejection_{int(art_rejection)}_lpf_{int(lpf)}/"

    return data_dir
