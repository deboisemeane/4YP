from preprocess import SHHSPreprocessor
import torch
pre = SHHSPreprocessor(art_rejection=True, lpf=True)
pre.process_t(incl_preceeding_epochs=0, incl_following_epochs=0)


#print(torch.cuda.is_available())
