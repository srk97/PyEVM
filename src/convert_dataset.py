from .EVM import magnify_color, magnify_motion
from ..hparams.registry import get_hparams
import os
from memory_profiler import profile

path_to_data = '/home/srk/NTU/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/Clips/'
deceptive_path = os.path.join(path_to_data,'Deceptive')
truth_path = os.path.join(path_to_data,'Truthful')
folders = ['Deceptive', 'Truthful']
hparams_dict = get_hparams()

@profile
def convert():
  for key in hparams_dict.keys():
      curr_path = os.path.join(os.path.dirname(__file__), '../Outputs', key)
      hps = hparams_dict[key]
      if not os.path.isdir(curr_path):
          os.mkdir(curr_path)
      for i, path in enumerate([deceptive_path, truth_path]):
          for f in os.listdir(path):
            write_path = os.path.join(curr_path, folders[i])
            if not os.path.isdir(write_path):
              os.mkdir(write_path)
            saved_name = os.path.join(write_path, f.strip('mp4')[:-1])

            for fn in hps.mode:
              print(fn)
              magnify_motion(os.path.join(path,f), hps, saved_name)

if __name__=='__main__':
  convert()