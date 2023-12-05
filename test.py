from preprocess import SHHSPreprocessor

pre = SHHSPreprocessor(art_rejection=True, lpf=False)
pre.process_t(incl_preceeding_epochs=2, incl_following_epochs=1)

