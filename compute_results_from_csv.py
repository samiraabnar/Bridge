import numpy as np
if __name__ == '__main__':

  labels = ['Temporal_Pole_Mid_L',
'Cerebelum_Crus1_L',
'Temporal_Pole_Sup_L',
'Parietal_Sup_L',
'Postcentral_L',
'ParaHippocampal_L',
'Amygdala_L',
'Pallidum_L',
'Thalamus_R',
'Hippocampus_R',
'Putamen_R',
'Postcentral_R',
'Fusiform_R',
'Precentral_R',
'Temporal_Inf_R',
'Temporal_Mid_R',

]
  with open('/Users/samiraabnar/Codes/Bridge/selected_prz_0.csv') as file:
    lines = file.readlines()
    prz = {}
    for i, line in enumerate(lines):
      line = line.replace(',','.')
      prz[i] = list(map(float,line.split()))


    print(len(prz.keys()))
    region_region_sims = {}
    for k in np.arange(len(labels)):
      region_region_sims[labels[k]] = []
      for sub in np.arange(8):
        region_region_sims[labels[k]].append(prz[sub*16+k ])

    for key in region_region_sims:
      print(key, np.mean(region_region_sims[key]))