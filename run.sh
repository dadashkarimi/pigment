#!/bin/bash
#SBATCH --mem-per-gpu=160G
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node=a40:1
#SBATCH --time=2-00:00:00         # Set expected wall time
#SBATCH --job-name="e5"
#SBATCH --output="e5.out"

# --- Env ---
module purge
module load cuda/12.2            # matches torch 2.5.1+cu121 logs
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate pushkar

export HF_HOME="$PWD/.hf_cache"
# export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
EPOCHS=1000   # example
export EPOCHS
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch, sys
print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__, "| CUDA avail:", torch.cuda.is_available(), "| CUDA:", torch.version.cuda)
print("GPU   :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

# ---------------- Common training args ----------------
RUNPY="train_segformer.py"

export TARGET_EPOCHS="$EPOCHS"

NUM_LABELS=2
IMAGE_SIZE=512
LR=3e-5
TRAIN_BS=16
EVAL_BS=16
SEED=42
SAVE_STEPS=200
EVAL_STEPS=200
MAXSTEPS=999518670
# Keep this fixed forever; outputs go under OUTROOT/<EXP_NAME>/<MODEL_TAG>/
OUTROOT="segformer-rerun-equal-epochs"

# ---------------- Models ----------------
MODELS=(
  "PushkarA07/segformer-b0-finetuned-net-15Oct"
  # "nvidia/mit-b0"
  # "nvidia/segformer-b0-finetuned-ade-512-512"
  # "restor/tcd-segformer-mit-b0"
)

# ---------------- Experiments ----------------
# Uncomment only the experiment(s) you want to run.
# Each experiment is ONE run that may concatenate multiple datasets via --dataset_ids.
EXPS=(
    # "E1Round1|PushkarA07/2017-1-A3-dataset-new"
  # "E1Round2|PushkarA07/2017-1-A3-dataset-new PushkarA07/2016-5-A2-dataset-new PushkarA07/2016-5-P2-dataset-new"
  # "E1Round3|PushkarA07/2017-1-A3-dataset-new PushkarA07/2016-5-A2-dataset-new PushkarA07/2016-5-P2-dataset-new PushkarA07/2016-6-A1-dataset-new"
  # "Exp3_norm|PushkarA07/2017-1-A3-dataset-new PushkarA07/2016-6-A1-dataset-new"
  "Exp3_unnorm|PushkarA07/2017-1-A3-dataset-new PushkarA07/2016-6-A1-unnormalized-new"
)

# ---------------- Helpers ----------------
# Match your existing folder naming style like "nvidia_mit-b0"
tagify(){ echo "$1" | tr '/' '_' | tr ':' '_'; }

# Returns YES if trainer_state.json indicates epoch >= TARGET_EPOCHS
# Returns YES if trainer_state.json indicates global_step >= TARGET_STEPS
finished () {
  local out="$1"
  python - "$out" <<'PY'
import json, os, sys
out = sys.argv[1]
p = os.path.join(out, "trainer_state.json")
if not os.path.exists(p):
    print("NO"); sys.exit(0)
j = json.load(open(p))
epoch = float(j.get("log_history", [])[-1].get("epoch", 0.0)) if j.get("log_history") else 0.0
target = float(os.environ.get("TARGET_EPOCHS","0"))
print("YES" if epoch >= target else "NO")
PY
}



# ---------------- Run sequentially (no job array needed) ----------------
mkdir -p "$OUTROOT"

for exp in "${EXPS[@]}"; do
  EXP_NAME="${exp%%|*}"
  DATASETS="${exp#*|}"

  for MODEL in "${MODELS[@]}"; do
    MODEL_TAG="$(tagify "$MODEL")"
    OUTDIR="${OUTROOT}/${EXP_NAME}/${MODEL_TAG}"
    mkdir -p "$OUTDIR"

    echo "=================================================="
    echo "Experiment : $EXP_NAME"
    echo "Datasets   : $DATASETS"
    echo "Model      : $MODEL"
    echo "Out        : $OUTDIR"
    echo "=================================================="

    # Skip if already finished target epochs
    if [[ "$(finished "$OUTDIR" | tail -n1)" == "YES" ]]; then
      echo "[skip] already reached target epochs: $OUTDIR"
      continue
    fi

    # Train (your train_segformer.py auto-resumes from latest checkpoint in OUTDIR)
    python -u "$RUNPY" \
      --model_id "$MODEL" \
      --dataset_ids $DATASETS \
      --output_dir "$OUTDIR" \
      --train_batch_size "$TRAIN_BS" \
      --eval_batch_size "$EVAL_BS" \
      --learning_rate "$LR" \
      --num_epochs "$EPOCHS" \
      --save_steps "$SAVE_STEPS" \
      --eval_steps "$EVAL_STEPS" \
      --logging_steps 50 \
      --seed "$SEED"


    # Post-check
    if [[ "$(finished "$OUTDIR" | tail -n1)" == "YES" ]]; then
      echo "[done] finished epochs: $OUTDIR"
    else
      echo "[not done] did not reach target epochs yet (sbatch again to continue): $OUTDIR"
    fi

    echo
    sleep 5
  done
done

echo "[all] complete."