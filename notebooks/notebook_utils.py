import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["axes.grid"] = False

models = ['bert_0','bert_6', 'bert_11',
          'google_lm_lstm_0', 'google_lm_lstm_1',
          'elmo_lstm_outputs1', 'elmo_lstm_outputs2',
          'universal_large_none',
          'glove_none',
          'tf_token_none_',
          'brain_1', 'brain_2','brain_3','brain_4','brain_5','brain_6','brain_7','brain_8',
          'uniform_random_300', 'normal_ransom_300',
           'subject_4Temporal_Pole_Mid_L',
           'subject_4Cerebelum_Crus1_L',
           'subject_4Temporal_Pole_Sup_L',
           'subject_4Parietal_Sup_L',
           'subject_4Postcentral_L',
           'subject_4ParaHippocampal_L',
           'subject_4Amygdala_L',
           'subject_4Pallidum_L',
           'subject_4Thalamus_R',
           'subject_4Hippocampus_R',
           'subject_4Putamen_R',
           'subject_4Postcentral_R',
           'subject_4Fusiform_R',
           'subject_4Precentral_R',
           'subject_4Temporal_Inf_R',
           'subject_4Temporal_Mid_R',
         ]
model_labels = ['L0', 'L6', 'L11',
                'L0', 'L1',
                'L0', 'L1',
                '',
                '',
                '',
                'brain_1', 'brain_2','brain_3','brain_4','brain_5','brain_6','brain_7','brain_8',
                'Rand_U', 'Rand_N',
                'subject_6Temporal_Pole_Sup_L',]

model_names = ['BERT', 'BERT', 'BERT',
                'GoogleLM', 'GoogleLM',
                'ELMO', 'ELMO',
                'UniSentEnc',
                'GloVe',
                'Token',
                'Brain_1', 'Brain_2','Brain_3','Brain_4','Brain_5','Brain_6','Brain_7','Brain_8',
                'Rand_U', 'Rand_N',
                'subject_6Temporal_Pole_Sup_L']

context_size= ['none', '0', '1', '2', '3', '4', '5', '6','','']
context_labels = ['0','1','2', '3', '4', '5', '6', '7','full','none']


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
  """
  Create a heatmap from a numpy array and two lists of labels.

  Arguments:
      data       : A 2D numpy array of shape (N,M)
      row_labels : A list or array of length N with the labels
                   for the rows
      col_labels : A list or array of length M with the labels
                   for the columns
  Optional arguments:
      ax         : A matplotlib.axes.Axes instance to which the heatmap
                   is plotted. If not provided, use current axes or
                   create a new one.
      cbar_kw    : A dictionary with arguments to
                   :meth:`matplotlib.Figure.colorbar`.
      cbarlabel  : The label for the colorbar
  All other arguments are directly passed on to the imshow call.
  """

  if not ax:
    ax = plt.gca()

  # Plot the heatmap
  im = ax.imshow(data, **kwargs)

  # create an axes on the right side of ax. The width of cax will be 5%
  # of ax and the padding between cax and ax will be fixed at 0.05 inch.
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cbar = ax.figure.colorbar(im, ax=ax, cax=cax, **cbar_kw)
  cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

  # We want to show all ticks...
  ax.set_xticks(np.arange(data.shape[1]))
  ax.set_yticks(np.arange(data.shape[0]))
  # ... and label them with the respective list entries.
  ax.set_xticklabels(col_labels)
  ax.set_yticklabels(row_labels)

  # Let the horizontal axes labeling appear on top.
  ax.tick_params(top=True, bottom=False,
                 labeltop=True, labelbottom=False)
  ax.tick_params(axis='both', labelsize=12)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
           rotation_mode="anchor")

  # Turn spines off and create white grid.
  # for edge, spine in ax.spines.items():
  #    spine.set_visible(False)

  ax.set_xticks(np.arange(0, data.shape[1] + 1) - 0.5, minor=True)
  ax.set_yticks(np.arange(0, data.shape[0] + 1) - 0.5, minor=True)
  # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
  # ax.tick_params(which="minor", bottom=False, left=False)

  return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
  """
  A function to annotate a heatmap.

  Arguments:
      im         : The AxesImage to be labeled.
  Optional arguments:
      data       : Data used to annotate. If None, the image's data is used.
      valfmt     : The format of the annotations inside the heatmap.
                   This should either use the string format method, e.g.
                   "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
      textcolors : A list or array of two color specifications. The first is
                   used for values below a threshold, the second for those
                   above.
      threshold  : Value in data units according to which the colors from
                   textcolors are applied. If None (the default) uses the
                   middle of the colormap as separation.

  Further arguments are passed on to the created text labels.
  """

  if not isinstance(data, (list, np.ndarray)):
    data = im.get_array()

  # Normalize the threshold to the images color range.
  if threshold is not None:
    threshold = im.norm(threshold)
  else:
    threshold = im.norm(data.max()) / 2.

  # Set default alignment to center, but allow it to be
  # overwritten by textkw.
  kw = dict(horizontalalignment="center",
            verticalalignment="center")
  kw.update(textkw)

  # Get the formatter in case a string is supplied
  if isinstance(valfmt, str):
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

  # Loop over the data and create a `Text` for each "pixel".
  # Change the text's color depending on the data.
  texts = []
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
      text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
      texts.append(text)

  return texts


def get_data_mat(selected_models, c_sizes, item_key, delay=0, block=1, summary_source=None):
  size = len(selected_models) * len(c_sizes)
  data_mat = np.zeros(shape=(size, size))
  keys = []
  labels = []

  for i in selected_models:
    for k in c_sizes:
      key = '_'.join([models[i], 'context', str(context_size[k])])
      l = '_'.join([model_labels[i], context_labels[k]])
      keys.append(key)
      labels.append(l)

  # keys.extend(['uniform_random_300','normal_ransom_300'])
  # labels.extend(['RAN_U','RAN_N'])

  for i in np.arange(len(keys)):
    index_i = indexed_labels(keys[i], delay, block, summary_source)
    for j in np.arange(len(keys)):
      index_j = indexed_labels(keys[j], delay, block, summary_source)
      data_mat[i][j] = summary_source[delay][block][item_key][index_i][index_j]

  return data_mat


def plot_heatmap(selected_models, c_sizes, plot_name, delay=0, block=1, summary_source=None):
  size = len(selected_models) * len(c_sizes)
  data_mat = np.zeros(shape=(size, size))
  p_vals = np.zeros(shape=(size, size))
  keys = []
  labels = []

  for i in selected_models:
    for k in c_sizes:
      key = '_'.join([models[i], 'context', str(context_size[k])])
      l = '_'.join([model_labels[i], context_labels[k]])
      keys.append(key)
      labels.append(l)

  # keys.extend(['uniform_random_300','normal_ransom_300'])
  # labels.extend(['RAN_U','RAN_N'])

  for i in np.arange(len(keys)):
    index_i = indexed_labels(keys[i], delay, block, summary_source)
    for j in np.arange(len(keys)):
      index_j = indexed_labels(keys[j], delay, block, summary_source)
      data_mat[i][j] = summary_source[delay][block]['prz'][index_i][index_j]
      p_vals[i][j] = summary_source[delay][block]['p_vals'][index_i][index_j]

  fig, ax = plt.subplots(figsize=(8, 8))

  im, cbar = heatmap(np.transpose(data_mat), labels, labels, ax=ax,
                     cmap="YlGn", cbarlabel="")
  texts = annotate_heatmap(im, valfmt="{x:.3f}", size=8)

  print("pvalue range:", np.min(p_vals), np.max(p_vals), np.mean(p_vals))
  fig.tight_layout()
  plt.savefig(plot_name, dpi=360)
  plt.show()


def print_table(selected_models, target_models,
                selected_sizes, target_sizes,
                plot_name, delay=0, block=1, summary_source=None):
  import pandas
  size_1 = len(selected_sizes)
  size_2 = len(selected_models)
  data_mat = np.zeros(shape=(size_1, size_2))
  p_vals = np.zeros(shape=(size_1, size_2))
  selected_keys = []
  selected_labels = []
  target_keys = []
  target_labels = []

  for i, (k, p) in enumerate(zip(selected_sizes, target_sizes)):
    for j, (sel, tar) in enumerate(zip(selected_models, target_models)):
      if 'brain' not in models[sel] and 'random' not in models[sel] \
          and 'ransom' not in models[sel] and 'subject' not in models[sel]:
        sel_key = '_'.join([models[sel], 'context', str(context_size[k])])
      else:
        sel_key = '_'.join([models[sel]])

      if 'brain' not in models[tar] and 'random' not in models[tar] \
          and 'ransom' not in models[tar] and 'subject' not in models[tar]:
        tar_key = '_'.join([models[tar], 'context', str(context_size[p])])
      else:
        tar_key = '_'.join([models[tar]])

      index_i = indexed_labels(sel_key, delay, block, summary_source)
      index_j = indexed_labels(tar_key, delay, block, summary_source)

      data_mat[i][j] = summary_source[delay][block]['prz'][index_i][index_j]
      p_vals[i][j] = summary_source[delay][block]['p_vals'][index_i][index_j]

  pandas.set_option('display.expand_frame_repr', False)

  # print("pvalue range:", np.min(p_vals), np.max(p_vals), np.mean(p_vals))
  print(pandas.DataFrame(data_mat, selected_sizes))  # ,[model_labels[k] for k in selected_models]))

  return data_mat, selected_sizes, [model_labels[k] for k in selected_models]


def indexed_labels(label, delay, block, source_summary):
  return source_summary[delay][block]['labels_'].index(label)


def compute_std(selected_models, c_sizes, blocks, delay, mean_block_id=-1, key='prz', rsa_summary=None):
  means = []
  mean_data = get_data_mat(selected_models=selected_models,
                        c_sizes=c_sizes,
                        delay=delay, block=mean_block_id,
                        item_key=key, summary_source=rsa_summary)
  for block in blocks:
    data = get_data_mat(selected_models=selected_models,
                        c_sizes=c_sizes,
                        delay=delay, block=block,
                        item_key=key, summary_source=rsa_summary)
    means.append(pow(mean_data - data, 2))

  means = np.mean(means, axis=0)
  means - np.sqrt(means)
  return means
