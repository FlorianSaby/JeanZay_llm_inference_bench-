#!/bin/bash
set -a  # Automatically export all variables
source a100env
set +a  # Stop automatically exporting

# Load utility functions
source scripts/utils.sh

#######################################################
# ENVIRONMENT VARIABLES TO CHANGE
#######################################################
FRAMEWORKS=( "vllm" )
DATASETS=(  "sharegpt" "sonnet")  # "sharegpt"
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-70B-Instruct" "meta-llama/Llama-3.1-405B-Instruct" ) # "meta-llama/Llama-3.1-405B-Instruct" "meta-llama/Llama-3.1-70B-Instruct"
NUMBER_OF_NODES=(4) #You can let 4 Since DP is not working code will regulate number of nodes to use only the right number
REPEATS=3               # Number of runs per configuration
#######################################################
echo $ACCOUNT
GPUS_PER_NODE=$GPUS_PER_NODE
CPUS_PER_NODE=$CPUS_PER_NODE
VRAM_PER_NODE=$VRAM_PER_NODE
TIME_LIMIT=$TIME_LIMIT
JOB_IDS=()
CONFIG_INDEX=0
CURRENT_DIR=$(pwd)
TOTAL_CONFIGS=$(( ${#DATASETS[@]} * ${#FRAMEWORKS[@]} * ${#NUMBER_OF_NODES[@]} * ${#MODELS[@]} * REPEATS ))

for framework in "${FRAMEWORKS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for NODES in "${NUMBER_OF_NODES[@]}"; do
        # GENERAL PART (Common for all Frameworks).
        TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
        TOTAL_CPUS=$((NODES * CPUS_PER_NODE))
        TOTAL_VRAM=$((NODES *VRAM_PER_NODE))
        VRAM_PER_GPU=$((VRAM_PER_NODE / GPUS_PER_NODE))

        BASE_FOLDER="results/${framework}/${dataset}/$(basename "$model")"


        # Define a unique MODEL_PATH per configuration
        MODEL_TYPE=$(get_model_type "$model" "configs/model_type_map.json")
        MODEL_DIRECTORY=$(get_model_directory "$MODEL_TYPE" "configs/model_type_directories_map.json")
        MODEL_PATH="${MODEL_DIRECTORY}/${model}"
        RAY_PATH="/tmp"
        mkdir -p $RAY_PATH

        if [ -z "$MODEL_DIRECTORY" ]; then
          echo "Unknown model type '$MODEL_TYPE' or missing directory mapping. Exiting."
          exit 1
        fi


        DATASET_PATH=$(get_dataset_path "$dataset" "configs/config_datasets_paths_map.json")

        # Extract param count in billions (e.g., 13B → 13)
        param_b=$(basename "$model" | grep -Po '[0-9]+(?=[Bb])')
        # VRAM required in GB
        VRAM_FACTOR=3.5   # FP16 x2 + safety margin x1.5
        required_vram=$(echo "$VRAM_FACTOR * $param_b" | bc -l)  # Use -l for floating point

        # Skip if not enough total VRAM
        if [ "$(echo "$TOTAL_VRAM < $required_vram" | bc -l)" -eq 1 ]; then
            echo "Skipping $(basename "$model"): needs ${required_vram}GB but only ${TOTAL_VRAM}GB available."
            continue
        fi

        # Compute number of GPUs needed (ceil(required_vram / VRAM_PER_GPU))
        nb_gpu_actif=$(echo "($required_vram / $VRAM_PER_GPU + 0.9999)/1" | bc -l)
        nb_gpu_actif=${nb_gpu_actif%.*}  # Strip decimal part to get integer

        # Total GPU VRAM
        total_gpu_vram=$(echo "$nb_gpu_actif * $VRAM_PER_GPU" | bc -l)
        total_gpu_vram=${total_gpu_vram%.*}  # Integer

        # Number of nodes needed = ceil(total_gpu_vram / VRAM_PER_NODE)
        nb_noeud_actif=$(echo "($total_gpu_vram / $VRAM_PER_NODE + 0.9999)/1" | bc -l)
        nb_noeud_actif=${nb_noeud_actif%.*}  # Integer

        # Compute TP/PP
        if [ "$nb_gpu_actif" -lt "$GPUS_PER_NODE" ]; then
            TENSOR_PARALLEL=$nb_gpu_actif
        else
            TENSOR_PARALLEL=$GPUS_PER_NODE
        fi
        # If > 1 and odd → round up to make it divisible
        # (required because it crashes if the number of attention heads
        # is not divisible by TP)
        if [ "$TENSOR_PARALLEL" -gt 1 ] && [ $((TENSOR_PARALLEL % 2)) -ne 0 ]; then
            TENSOR_PARALLEL=$((TENSOR_PARALLEL + 1))
        fi

        PIPELINE_PARALLEL=$nb_noeud_actif


        # Compute DATA_PARALLEL = ceil(TOTAL_GPUS / (TP * PP))
        DATA_PARALLEL=$(echo "($TOTAL_GPUS / ($TENSOR_PARALLEL * $PIPELINE_PARALLEL) + 0.9999)/1" | bc) #WIP

        NODES=$PIPELINE_PARALLEL #wip remove when dp will be implemented
        RUN_FOLDER="Nodes_${NODES}-GPUs_${GPUS_PER_NODE}-TP_${TENSOR_PARALLEL}-PP_${PIPELINE_PARALLEL}" #_DP_${DATA_PARALLEL} WIP
        FULL_FOLDER="${BASE_FOLDER}/${RUN_FOLDER}"

        # vLLM
        if [[ "$framework" == "vllm" ]]; then
          # vLLM
          echo "FrameWork vLLM"

          for (( run_id=1; run_id<=REPEATS; run_id++ )); do
            LAUNCH_FOLDER="${CURRENT_DIR}/${FULL_FOLDER}/launch-${run_id}"
            echo "Setting up $LAUNCH_FOLDER"
            mkdir -p "$LAUNCH_FOLDER"
             
            cp $SCRIPT_VLLM "$LAUNCH_FOLDER"
            FILE_NAME="${SCRIPT_VLLM##*/}"
            cd "$LAUNCH_FOLDER" || exit 1
            export NODES CPUS_PER_NODE GPUS_PER_NODE TENSOR_PARALLEL PIPELINE_PARALLEL DATA_PARALLEL
            export FRAMEWORK="$framework" DATASET="$dataset" MODEL="$model" REPEAT_ID="$run_id" LAUNCH_FOLDER BENCHMARK_FILE DATASET_PATH
            export MODEL_PATH RAY_PATH
            export ADDITIONAL_ARGS
            export MODULES
	
            REMAINING=$((TOTAL_CONFIGS - CONFIG_INDEX))
            if [ "$REMAINING" -le 5 ] && [ "${#JOB_IDS[@]}" -gt 0 ]; then
              DEPENDENCY="--dependency=afterany:${JOB_IDS[-1]}"
            else
              DEPENDENCY=""
            fi
	    export TMP_DIR=$SCRATCH/TMP
	    echo "Submit job..."
            JOB_ID=$(sbatch --parsable \
            --chdir=$(pwd) \
            --nodes=$NODES \
            --cpus-per-task=$CPUS_PER_NODE \
            --gres=gpu:$GPUS_PER_NODE \
            --partition=$PARTITION_NAME \
            --constraint=$CONSTRAINT \
            $DEPENDENCY \
            --output=run-%j.out \
            --error=run-%j.out \
            -A $ACCOUNT \
            -q $QOS \
            --time=$TIME_LIMIT \
            --exclusive \
            $FILE_NAME)

            echo "Submitted job $JOB_ID for $LAUNCH_FOLDER"
            JOB_IDS+=("$JOB_ID")
            ((CONFIG_INDEX++))

            cd - > /dev/null
            sleep 5
          done
        fi

      done
    done
  done
done
