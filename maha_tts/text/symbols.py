import sys
from maha_tts.config import config

labels=" abcdefghijklmnopqrstuvwxyz.,:;'()?!\""
labels=" !\"'(),-.:;?[]abcdefghijklmnopqrstuvwxyzàâèéêü’“”"
labels= [i for i in labels]

text_labels = [i for i in labels]
text_labels+='<S>','<E>','<PAD>'

code_labels= [str(i) for i in range(config.semantic_model_centroids)]
labels+=code_labels
code_labels+='<SST>','<EST>','<PAD>'

labels+='<S>','<E>','<SST>','<EST>','<PAD>'

tok_enc = {j:i for i,j in enumerate(labels)}
tok_dec = {i:j for i,j in enumerate(labels)}

#text encdec
text_enc = {j:i for i,j in enumerate(text_labels)}
text_dec = {i:j for i,j in enumerate(text_labels)}

#code encdec
code_enc = {j:i for i,j in enumerate(code_labels)}
code_dec = {i:j for i,j in enumerate(code_labels)}

# print('length of the labels: ',len(labels))