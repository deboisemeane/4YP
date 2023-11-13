import mne
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


# This class is used to process raw SHHS-1 data for all (selected) participants.
class SHHSPreprocessor:

    def __init__(self, **params):

        self.demographics = pd.read_csv('data/Raw/shhs/datasets/shhs-harmonized-dataset-0.20.0.csv')
        self.choose_patients()
        self.raw_eegs: [mne.io.Raw] = self.load_raw_eegs(self)

    # Exclude unwanted patients.
    # Sleep disorders, second visits, and unavailable sleep scoring are excluded.
    def choose_patients(self):
        df = self.demographics
        full_scoring = df["nsrr_flag_spsw"] == 'full scoring'
        acceptable_ahi = df["nsrr_ahi_hp3r_aasm15"] <= 15
        first_visit = df["visitnumber"] == 1
        chosen_patients = np.logical_and(full_scoring, acceptable_ahi, first_visit)
        self.demographics = df[chosen_patients]

    # Creates a list of mne.io.Raw objects, one for each chosen nsrrid.
    @staticmethod
    def load_raw_eegs(self):
        raw_eegs = {}
        edfs_dir = 'data/Raw/shhs/polysomnography/edfs/shhs1/'
        chosen_patients = self.demographics["nsrrid"]

        for patient in chosen_patients:
            patient_path = edfs_dir+"shhs1-"+str(patient)+".edf"
            if os.path.isfile(patient_path):
                raw_eeg = mne.io.read_raw_edf(patient_path)
                raw_eegs.update({patient: raw_eeg})
            else:
                print(f"Patient {patient} .edf file unavailable.")
        return raw_eegs


    @staticmethod
    def load_stage_labels(self):
        stage = {}
        # Iterate over all the selected patients.
        for nsrrid in self.demographics["nsrrid"]:
            # Initialise labels array
            total_duration = self.raw_eegs[f'{nsrrid}'].times[-1]
            n_epochs = total_duration // 30
            labels = [None] * n_epochs
            # Create path string for current patient.
            annotations_path = f"data/Raw/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-{nsrrid}-nsrr.xml"
            # Check the annotations file is available for this patient.
            if os.path.isfile(annotations_path):
                annotations = ET.parse(f"data/Raw/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-{nsrrid}-nsrr.xml")
                root = annotations.getroot()
                # Check all ScoredEvents
                for event in root.findall(".//ScoredEvent"):    # . means starting from current node, // means ScoredEvent does not have to be a direct child of root
                    event_type = event.find("EventType").text
                    # Check if this event is a stage annotation
                    if "Stages|Stages" in event_type:
                        stage = event.find("EventConcept").text
                        # Label integer is at end of EventConcept string.
                        label = int(stage[-1])
                        # Convert to our own labelling convention ("0:N3, 1:N1/N2, 2:REM, 3:W")
                        if label == 3 or label == 4:  # Accounting for possibility of N4 in data.
                            label = 0
                        elif label == 2 or label == 1:
                            label = 1
                        elif label == 5:
                            label = 2
                        elif label == 0:
                            label = 3
                        else:
                            raise ValueError(f"Unsupported label '{event_type}.'")
                        # Store label in its corresponding positions in a numpy array, based on start and duration of this event.
                        start = int(event.find("Start").text)
                        duration = int(event.find("Duration").text)
                        for i in range(duration//30):
                            index = start // 30 + i
                            labels[index] = label






            else:
                print(f"Annotations file for id {nsrrid} not available.")



