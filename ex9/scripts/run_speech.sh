#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-auto}
CONDITION=${CONDITION:-hubert_soft,world_aux}
CONDITIONS=${CONDITIONS:-${CONDITION}}
TRAIN_BASE_DIR=${TRAIN_BASE_DIR:-runs/speech_world_flow}
SAMPLE_BASE_DIR=${SAMPLE_BASE_DIR:-outputs/speech_world_flow}
PLOT_MAX_PLOTS=${PLOT_MAX_PLOTS:-2}

train_args=()
sample_args=()

if [[ -n "${FEATURE_MANIFEST:-}" ]]; then
  train_args+=(--feature-manifest "${FEATURE_MANIFEST}")
fi
if [[ -n "${SPEAKERS:-}" ]]; then
  train_args+=(--speakers "${SPEAKERS}")
fi
if [[ -n "${FEATURE_STATISTICS:-}" ]]; then
  sample_args+=(--statistics-path "${FEATURE_STATISTICS}")
elif [[ -n "${STATISTICS_PATH:-}" ]]; then
  sample_args+=(--statistics-path "${STATISTICS_PATH}")
elif [[ -n "${FEATURE_MANIFEST:-}" ]]; then
  sample_args+=(--statistics-path "$(dirname "${FEATURE_MANIFEST}")/feature_statistics.json")
fi
if [[ -n "${NUM_STEPS:-}" ]]; then
  train_args+=(--num-steps "${NUM_STEPS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  train_args+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${SEGMENT_FRAMES:-}" ]]; then
  train_args+=(--segment-frames "${SEGMENT_FRAMES}")
fi
if [[ -n "${NUM_UTTERANCES:-}" ]]; then
  sample_args+=(--num-utterances "${NUM_UTTERANCES}")
fi
if [[ -n "${SPLIT:-}" ]]; then
  sample_args+=(--split "${SPLIT}")
fi
if [[ -n "${SEED:-}" ]]; then
  sample_args+=(--seed "${SEED}")
fi
if [[ -n "${LATENT_SCALE:-}" ]]; then
  sample_args+=(--latent-scale "${LATENT_SCALE}")
fi
if [[ "${NO_SYNTHESIS:-0}" == "1" ]]; then
  sample_args+=(--no-synthesis)
fi

for condition in ${CONDITIONS}; do
  condition_id="${condition//,/+}"
  train_output_dir="${TRAIN_BASE_DIR}/${condition_id}"
  sample_output_dir="${SAMPLE_BASE_DIR}/${condition_id}"

  "${PYTHON}" scripts/train_speech.py \
    --condition "${condition}" \
    --device "${DEVICE}" \
    --output-dir "${train_output_dir}" \
    "${train_args[@]}"

  "${PYTHON}" scripts/sample_speech.py \
    --condition "${condition}" \
    --device "${DEVICE}" \
    --checkpoint "${train_output_dir}/checkpoint.pt" \
    --output-dir "${sample_output_dir}" \
    "${sample_args[@]}"

  "${PYTHON}" scripts/plot_speech.py \
    --train-output-dir "${train_output_dir}" \
    --sample-output-dir "${sample_output_dir}" \
    --output-dir "${sample_output_dir}" \
    --max-plots "${PLOT_MAX_PLOTS}"
done
