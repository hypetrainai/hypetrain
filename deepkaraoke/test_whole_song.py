
# coding: utf-8

# In[1]:


import torch
import numpy as np
import scipy.io as io
import importlib
import dataloader
from tensorboardX import SummaryWriter


# In[19]:


import pickle as pkl
with open('data/16k_LARGE/test.pkl.gz', 'rb') as file:
    test_this = pkl.load(file)
print(sorted(test_this.keys()))
with open('data/16k_LARGE/train.pkl.gz', 'rb') as file:
    test_this = pkl.load(file)
print(sorted(test_this.keys()))


# In[3]:


dataset = dataloader.KaraokeDataLoader('data/16k_LARGE/test.pkl.gz', batch_size = 1)


# In[8]:



from GLOBALS import FLAGS as FLAGS

NNModel = getattr(importlib.import_module(FLAGS.module_name), FLAGS.model_name)
model_state = torch.load('trained_models/largedataset_secondtest/model_22000.pt')
model = NNModel()
model = model.cuda()
model.load_state_dict(model_state['state_dict'])
model.current_step = 0
model.eval()


# In[9]:


import wave
import struct

for idx in range(10):
    song_data = [dataset.get_single_segment(extract_idx = idx,start_value = 0, sample_length = 0)]
    data_in = model.preprocess(song_data)
    with torch.no_grad():
        data_out = model.predict(data_in)
    
    tensor = np.clip(song_data[0].data[0],-1.0,1.0)
    fio = 'outputs/test_onvocal_%d.wav'%idx
    tensor_list = [int(32767.0 * x) for x in tensor]
    Wave_write = wave.open(fio, 'wb')
    Wave_write.setnchannels(1)
    Wave_write.setsampwidth(2)
    Wave_write.setframerate(16000)
    tensor_enc = b''
    bytelist = [struct.pack('<h',v) for v in tensor_list]
    tensor_enc = tensor_enc.join(bytelist)
    Wave_write.writeframes(tensor_enc)
    Wave_write.close()

    tensor = np.clip(data_out,-1.0,1.0)
    fio = 'outputs/test_predicted_%d.wav'%idx
    tensor_list = [int(32767.0 * x) for x in tensor]
    Wave_write = wave.open(fio, 'wb')
    Wave_write.setnchannels(1)
    Wave_write.setsampwidth(2)
    Wave_write.setframerate(16000)
    tensor_enc = b''
    bytelist = [struct.pack('<h',v) for v in tensor_list]
    tensor_enc = tensor_enc.join(bytelist)
    Wave_write.writeframes(tensor_enc)
    Wave_write.close()
    #io.wavfile.write('outputs/test_onvocal_%d.wav'%idx,16000,song_data[0].data[0])
    #io.wavfile.write('outputs/test_predicted_%d.wav'%idx,16000,np.clip(data_out,-1,1))


# In[76]:



tensor = np.clip(data_out,-1.0,1.0)
assert(tensor.ndim == 1), 'input tensor should be 1 dimensional.'

tensor_list = [int(32767.0 * x) for x in tensor]
import io
import wave
import struct
fio = 'testwav.wav'
Wave_write = wave.open(fio, 'wb')
Wave_write.setnchannels(1)
Wave_write.setsampwidth(2)
Wave_write.setframerate(16000)
tensor_enc = b''
bytelist = [struct.pack('<h',v) for v in tensor_list]
tensor_enc = tensor_enc.join(bytelist)
Wave_write.writeframes(tensor_enc)
Wave_write.close()

