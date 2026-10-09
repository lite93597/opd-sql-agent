# Source this before using the project; the base AutoDL environment stays intact.
export OPD_ROOT=/root/autodl-tmp/opd-sql-agent
export OPD_STORAGE=/root/autodl-tmp
export HF_HOME="$OPD_STORAGE/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME="$OPD_STORAGE/cache/torch"
export TRITON_CACHE_DIR="$OPD_STORAGE/cache/triton"
export VLLM_CACHE_ROOT="$OPD_STORAGE/cache/vllm"
export UV_CACHE_DIR="$OPD_STORAGE/cache/uv"
export TMPDIR="$OPD_STORAGE/tmp"
export TOKENIZERS_PARALLELISM=false
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="12.0"
source "$OPD_STORAGE/envs/opd/bin/activate"
export PYTHONPATH="$OPD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# CUDA_HOME is filled with the CUDA 13 development toolkit during setup.
if [ -d "$OPD_STORAGE/cuda-13" ]; then
    export CUDA_HOME="$OPD_STORAGE/cuda-13"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
