import pandas as pd
import ivy
import jax

def get_perf_analysis(mode="equal", metric="Avg", batch_size="small"):

    file = "frequency_sorted_benchmarking_results" if mode == "frequency" else "equal_sorted_benchmarking_results"

    df = pd.read_csv(f"/workspaces/graph-compiler/hf_models_testing/{file}.txt")
    size_grouping = df.groupby('batch_size')

    df_small = size_grouping.get_group(1)
    df_large = size_grouping.get_group(50)
    df = df_small if batch_size == "small" else df_large

    if metric == "Std":
        orig_metric, comp_metric, torch_metric, jax_metric =  df["orig_std_time"], df["comp_std_time"], df["torch_std_time"], df["jax_std_time"]
    elif metric == "Nbr":
        comp_metric, torch_metric, jax_metric =  df["comp_num_functions"], df["torch_num_functions"], df["jax_num_functions"]
        orig_metric = comp_metric
    else:
        orig_metric, comp_metric, torch_metric, jax_metric =  df["orig_mean_time"], df["comp_mean_time"], df["torch_mean_time"], df["jax_mean_time"]

    perf_df = pd.DataFrame({f"Torch {metric} Improvement": (1- torch_metric/orig_metric)*100, f"Jax {metric} Improvement": (1 - jax_metric/comp_metric)*100})

    return perf_df

if __name__ == "__main__":
    mode = "equal"
    metric = "Avg"
    batch_size = "large"

    print(get_perf_analysis(mode=mode, metric=metric, batch_size=batch_size))