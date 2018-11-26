#SBATCH -S /bin/bash

#SBATCH -p gpu
#SBATCH -lnodes=1
#SBATCH -t 5:00:00


module load eb
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA-9.0.176/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/home/samigpu/Codes/GoogleLM1b/
cd ~/Codes/Bridge

text_encoder='google_lm' #universal_large elmo glove tf_token

past_window=0
embedding_type='lstm0'
python encode_stimuli_in_context.py --root /home/samigpu/Codes/ --text_encoder=$text_encoder --embedding_type=$embedding_type --past_window=$past_window --context_mode=sentence

past_window=0
embedding_type='lstm1'
python encode_stimuli_in_context.py --root /home/samigpu/Codes/ --text_encoder=$text_encoder --embedding_type=$embedding_type --past_window=$past_window --context_mode=sentence

past_window=1
embedding_type='lstm0'
python encode_stimuli_in_context.py --root /home/samigpu/Codes/ --text_encoder=$text_encoder --embedding_type=$embedding_type --past_window=$past_window --context_mode=sentence

past_window=1
embedding_type='lstm1'
python encode_stimuli_in_context.py --root /home/samigpu/Codes/ --text_encoder=$text_encoder --embedding_type=$embedding_type --past_window=$past_window --context_mode=sentence