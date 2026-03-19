# Benchmark Instructions

## Notes
If you are using the code directly with the datasets and models in the `DSDIR` folder, steps 1 can be skipped.

## Step 1: Download Datasets and Models

**Datasets (into `benchmarks/datasets`):**  
- [ShareGPT_V3_unfiltered_cleaned_split](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json)  
- [Sonnet](https://huggingface.co/datasets/zhyncs/sonnet)  

**Models (into `benchmarks/models`):**  
- [LLaMA 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  
- [LLaMA 3.1 70B Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)  
- [LLaMA 3.1 405B Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)  

## Step 2: Configuration
- Update the paths in the files inside the `config` folder:  
  - `config_datasets_paths_map`  
  - `model_type_directories_map`  
  - `model_type_map.json`  
- Also update the information in `A100env` and `H100env` to match the user account number.

## Step 3: Run the Benchmark
- Execute the `run_benchmark.sh` script.  
- Select the GPU type (`A100env` or `H100env`) where the benchmark will run.

## Step 4: Generate Results Table
- After the benchmark finishes, run `generateSummaryTable-checkpoint.py` to convert the raw outputs into an Excel table.

## Step 5 (Optional): Analyze Results
- Use `bench_analysis.ipynb` to interpret the generated Excel file.

## Benchmark Results
Benchmark results are shown in the image below:

![Benchmark Results](results_bench.png)

