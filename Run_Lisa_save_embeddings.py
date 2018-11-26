#PBS -S /bin/bash

#PBS -qgpu
#PBS -lnodes=1
#PBS -lwalltime=10:00:00

module load eb
module load Python/3.6.3-foss-2017b
module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Devian9/cuDNN/7.0.5-CUDA-9.0.176/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/Users/samigpu/Codes/GoogleLM1b/
cd ~/Codes/Bridge

text_encoder='google_lm' #universal_large elmo glove tf_token
embedding_type='lstm0'
past_window=0

python encode_stimuli_in_context.py --root /Users/samigpu/Codes/ --text_encoder=$text_encoder --embedding_type=$embedding_type --past_window=$past_window --context_mode=sentence