import sys
import numpy as np

if __name__ == '__main__':
  print(sys.argv)

  for k in np.arange(1,len(sys.argv)):
    with open(sys.argv[k], 'r') as f:
      print(sys.argv[k])
      lines = f.readlines()
      i = 0
      mean_evs = {}
      while i < (len(lines) - 1):
        while not lines[i].startswith('Training with train time delay') and i < (len(lines)-2):
          i += 1
        if lines[i].startswith('Training with train time delay'):
          delay_info = lines[i].split()
          train_delay = int(delay_info[6])
          test_delay = int(delay_info[12])
        if train_delay not in mean_evs:
          mean_evs[train_delay] = []
        i += 1
        while not lines[i].startswith('Training with train time delay') and i < (len(lines)):
          if lines[i].startswith('mean_EV'):
            mean_evs[train_delay].append(lines[i].strip().split()[-1])
          i += 1
          if i > (len(lines) - 1):
            break


      print(mean_evs.keys())
      print_str_1 = ''
      print_str_2 = ''
      for t in np.arange(-6,1,2):
        for ev in mean_evs[t][1:]:
          print_str_1 += str(ev)+"\t"
        print_str_2 += mean_evs[t][0]+"\t"*4

      print(print_str_1)
      print(print_str_2)