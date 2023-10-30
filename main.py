from preprocess import ISRUCPreprocessor

for patient in range(1,11):
    pre = ISRUCPreprocessor(patient=patient)
    pre.save_features_labels_csv()