#!/bin/bash

set -euo pipefail

########################################
# 1. Environment (Singularity)
########################################
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_NUM_GPUS="$GPUS_PER_NODE"
export RAY_NUM_CPUS="$CPUS_PER_NODE"
export VLLM_USE_V1=0
export VLLM_USE_RAY_SPANNABLE_POOL=0
export VLLM_USE_RAY_COMPILED_DAG=0
export RAY_CGRAPH_get_timeout=1800
NB_NODES="$NODES"

export SINGULARITY_BINDS
export SINGULARITY_IMAGE

sing() {
  singularity exec --nv -B "$SINGULARITY_BINDS" "$SINGULARITY_IMAGE" "$@"
}
export -f sing

RAY_SRUN_PID=""
VLLM_PID=""
GPU_MON_PID=""

cleanup() {
  set +e
  if [[ -n "${GPU_MON_PID:-}" ]]; then
    kill "$GPU_MON_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
  fi
  sing ray stop --force >/dev/null 2>&1 || true
  if [[ -n "${RAY_SRUN_PID:-}" ]]; then
    kill "$RAY_SRUN_PID" >/dev/null 2>&1 || true
    wait "$RAY_SRUN_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

########################################
# 2. Network detection without iproute2
########################################
# This uses only Python's standard library, so it works even when the host
# and the container do not provide the `ip` command.
NET_INFO_PY=$(cat <<'PY'
import fcntl
import os
import re
import socket
import struct
import sys

pattern = re.compile(r"^(ib|hsn|sl)")
interfaces = sorted(os.listdir("/sys/class/net"))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for iface in interfaces:
    if not pattern.match(iface):
        continue
    try:
        request = struct.pack("256s", iface[:15].encode())
        response = fcntl.ioctl(sock.fileno(), 0x8915, request)  # SIOCGIFADDR
        ipv4 = socket.inet_ntoa(response[20:24])
    except OSError:
        continue
    print(iface, ipv4)
    break
else:
    print(
        "No ib*/hsn*/sl* interface with an IPv4 address. Available: "
        + ", ".join(interfaces),
        file=sys.stderr,
    )
    sys.exit(1)
PY
)
export NET_INFO_PY

########################################
# 3. Node discovery and head-node address
########################################
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
readarray -t nodes_array <<< "$nodes"
head_node=${nodes_array[0]}

head_info=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
  singularity exec --nv \
  -B "$SINGULARITY_BINDS" \
  "$SINGULARITY_IMAGE" \
  python -c "$NET_INFO_PY")

read -r IB_IFACE head_node_ip <<< "$head_info"
if [[ -z "${IB_IFACE:-}" || -z "${head_node_ip:-}" ]]; then
  echo "ERROR: Could not determine the high-speed interface and IPv4 address" >&2
  exit 1
fi

echo "Ray head node: $head_node"
echo "High-speed network: $IB_IFACE ($head_node_ip)"

export RAY_HEAD_IP="$head_node_ip"
export RAY_ADDRESS="$RAY_HEAD_IP:6379"

########################################
# 4. NCCL
########################################
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=10
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME="$IB_IFACE"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0

########################################
# 5. Launch Ray cluster
########################################
sing ray stop --force || true
sleep 5

srun --nodes="$NB_NODES" \
     --ntasks="$NB_NODES" \
     --ntasks-per-node=1 \
     singularity exec --nv \
     -B "$SINGULARITY_BINDS" \
     "$SINGULARITY_IMAGE" \
     bash -c '
set -euo pipefail

read -r local_iface local_ip <<< "$(python -c "$NET_INFO_PY")"
if [[ -z "$local_iface" || -z "$local_ip" ]]; then
  echo "[$(hostname)] Failed to determine local network information" >&2
  exit 1
fi

export RAY_NODE_IP_ADDRESS="$local_ip"
export VLLM_HOST_IP="$local_ip"
export HOST_IP="$local_ip"
export NCCL_SOCKET_IFNAME="$local_iface"

echo "[$(hostname)] Starting Ray on $local_iface ($local_ip)"

if [[ "$SLURM_PROCID" -eq 0 ]]; then
  exec ray start --head \
    --node-ip-address="$RAY_NODE_IP_ADDRESS" \
    --port=6379 \
    --num-cpus="$CPUS_PER_NODE" \
    --num-gpus="$GPUS_PER_NODE" \
    --disable-usage-stats \
    --block
else
  until python -c "import socket; socket.create_connection((\"$RAY_HEAD_IP\", 6379), timeout=2).close()" \
      >/dev/null 2>&1; do
    echo "[$(hostname)] Waiting for Ray head at $RAY_HEAD_IP:6379..."
    sleep 2
  done

  exec ray start \
    --address="$RAY_ADDRESS" \
    --node-ip-address="$RAY_NODE_IP_ADDRESS" \
    --num-cpus="$CPUS_PER_NODE" \
    --num-gpus="$GPUS_PER_NODE" \
    --disable-usage-stats \
    --block
fi
' &
RAY_SRUN_PID=$!

########################################
# 6. Wait for Ray
########################################
sleep 60
if ! kill -0 "$RAY_SRUN_PID" 2>/dev/null; then
  echo "Ray srun step exited before the cluster became ready" >&2
  wait "$RAY_SRUN_PID" || true
  exit 1
fi
sing ray status || { echo "Ray failed to start" >&2; exit 1; }

########################################
# 7. Launch vLLM
########################################
export VLLM_RAY_USE_EXISTING_CLUSTER=1
export VLLM_HOST_IP="$RAY_HEAD_IP"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PORT=8000
echo "Launching vLLM on $head_node ($RAY_HEAD_IP:$PORT)"

sing python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL" \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --host "$RAY_HEAD_IP" \
  --port "$PORT" \
  --trust-remote-code &
VLLM_PID=$!

########################################
# 8. Wait for vLLM and test it
########################################
echo "Waiting for vLLM to initialize weights (Model: $MODEL_PATH)..."
deadline=$((SECONDS + 1800))

while true; do
  http_code=$(sing curl -s -o /dev/null -w '%{http_code}' \
    "http://$RAY_HEAD_IP:$PORT/v1/models" || true)

  if [[ "$http_code" == "200" ]]; then
    break
  fi

  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready" >&2
    wait "$VLLM_PID" || true
    exit 1
  fi

  if (( SECONDS >= deadline )); then
    echo "vLLM server failed to become ready within 30 minutes" >&2
    exit 1
  fi

  echo "Still loading weights... HTTP status: ${http_code:-none}"
  sleep 20
done

echo "Server is UP. Sending inference request..."
sing curl -X POST "http://$RAY_HEAD_IP:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_PATH\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Explain the concept of GPU tensor parallelism in one sentence.\"}
    ],
    \"max_tokens\": 50
  }"

echo -e "\nInference test complete."

########################################
# 9. Benchmarking loop
########################################
concurrencies=(50) # 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)

for conc in "${concurrencies[@]}"; do
  echo "======================================="
  echo "Running concurrency level $conc"
  echo "Results folder: $LAUNCH_FOLDER"
  echo "======================================="

  METRICS_FILE="$LAUNCH_FOLDER/gpu_metrics_${conc}.csv"
  LOG_FILE="$LAUNCH_FOLDER/logs_benchmarking_${conc}_concurrency.log"
  RESULT_FILE="$LAUNCH_FOLDER/Concurrency_${conc}.json"

  sing nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,power.draw,utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits -l 1 > "$METRICS_FILE" &
  GPU_MON_PID=$!

  set +e
  sing python "$BENCHMARK_FILE" \
    --backend vllm \
    --host "$RAY_HEAD_IP" \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --dataset-name "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --max-concurrency "$conc" \
    --num-prompts 1000 \
    --save-result \
    --result-filename "$RESULT_FILE" \
    > "$LOG_FILE" 2>&1
  RC=$?
  set -e

  kill "$GPU_MON_PID" >/dev/null 2>&1 || true
  wait "$GPU_MON_PID" >/dev/null 2>&1 || true
  GPU_MON_PID=""
  sleep 2

  if [[ "$RC" -ne 0 ]]; then
    echo "Benchmark failed at concurrency=$conc (exit code $RC). See: $LOG_FILE" >&2
    exit "$RC"
  fi

  echo "Done concurrency=$conc"
done

echo "All concurrency runs completed successfully."
echo "Shutting down..."
