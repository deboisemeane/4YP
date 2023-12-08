from preprocess import SHHSPreprocessor
import torch
pre = SHHSPreprocessor(art_rejection=True, lpf=False)
pre.process_t(incl_preceeding_epochs=2, incl_following_epochs=1)

pre = SHHSPreprocessor(art_rejection=True, lpf=True)
pre.process_t(incl_preceeding_epochs=2, incl_following_epochs=1)

#print(torch.cuda.is_available())
