import os
import sys
import pickle
import argparse
import numpy as np
import warnings
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from copy import deepcopy as copy
from dadapy.metric_comparisons import MetricComparisons

from imbalance_gain_windows import *

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    two_pi = 2 * np.pi

    ## read the initial starting time for IG estimation
    ## this is to recycle the same trajectory for independent estimates
    parser = argparse.ArgumentParser()
    ## initial time for tau estimation
    parser.add_argument("--ic", dest="ic", default=None, type=int)
    parser.add_argument("--dci", dest="dci", default=None, type=int)
    parser.add_argument(
        "--scores",
        dest="scores",
        default="pc_data/830to1060us_sklearn_standard_scores.npy",
    )
    args = parser.parse_args()
    ic = args.ic
    dci = args.dci  # this sets the discarded window size [t-discard_close_ind, t+discard_close_ind]

    #!# number of jobs to parallelize over
    Njobs = 56

    #!# how many chunks to break the trajectory into
    #!# total size = 230 microsec in the 830to1060us window, so chunk_size=230 microsec/total_chunks
    total_chunks = 10

    if ic < 1 or ic > total_chunks:
        sys.exit(f"Error: IC must be between 1 and {total_chunks}")

    #!# Load and process the full data
    all_data = np.load(args.scores, mmap_mode="r")

    #!# Extract PCs
    pc1 = all_data[:, 0]
    pc2 = all_data[:, 1]
    pc3 = all_data[:, 2]
    pc4 = all_data[:, 3]
    pc5 = all_data[:, 4]
    pc6 = all_data[:, 5]
    pc7 = all_data[:, 6]
    pc8 = all_data[:, 7]
    pc9 = all_data[:, 8]
    pc10 = all_data[:, 9]

    #!# standardized PCs
    pc1_new = pc1 / pc1.std(axis=0)
    pc2_new = pc2 / pc2.std(axis=0)
    pc3_new = pc3 / pc3.std(axis=0)
    pc4_new = pc4 / pc4.std(axis=0)
    pc5_new = pc5 / pc5.std(axis=0)
    pc6_new = pc6 / pc6.std(axis=0)
    pc7_new = pc7 / pc7.std(axis=0)
    pc8_new = pc8 / pc8.std(axis=0)
    pc9_new = pc9 / pc9.std(axis=0)
    pc10_new = pc10 / pc10.std(axis=0)

    # print(f"Data shapes: all_data={all_data.shape}, PC1={pc1.shape}, PC2={pc2.shape}")

    #!# Concatenate all data together
    data = np.column_stack([pc1_new, pc2_new, pc3_new, pc4_new, pc5_new, pc6_new, pc7_new, pc8_new, pc9_new, pc10_new])
    # print(f"Verify concatenated data shape: {data.shape}")

    ## Number of features in our concatenated array
    Nfeatures = data.shape[1]
    print(f"Number of features: {Nfeatures}")

    ## Calculate chunk size
    total_rows = data.shape[0]
    chunk_size = total_rows // total_chunks

    ## Calculate start and end indices for the specific chunk
    start_idx = (ic - 1) * chunk_size
    if ic == total_chunks:  # Last chunk gets any remaining rows
        end_idx = total_rows
    else:
        end_idx = ic * chunk_size

    #!# the embedding dimension
    E = 1

    #!# embedding time (in frames)
    #!# 1 frame = 200 ps
    tau_e = 1

    #!# how often we check for causality (in frames) == spacing between tau's
    ## every 1 frame = 1 frames * 200 ps = 200 ps
    tau_gap = 25  #!# 5 ns

    #!# the 'tau' to start from in the range of values in (taus)
    ## useful if a job has to be restarted and we want to skip the already processed 'tau' values
    tau_start = 0

    #!# max duration of time lag; 1000 frames = 200 ns
    tau_end = 5000  #!# 1.0 us

    #!# 'tau' --> how often to check for causality from initial starting frame
    taus = np.arange(tau_start, tau_end, tau_gap)
    Nlags = len(taus)
    ## how many time lags are being processed (in frames)
    print(f"Number of time lags: {Nlags}")

    #!# maximum number of neighbours to consider (2-5% of total points are recommended)
    k = 50

    #!# range of alphas, for the driving variable, to find alpha which gives the maximum IG (imblanace gain)
    # alphas = np.linspace(0.0, 5.0, 500)
    beta = np.linspace(0.0, 1.0, 500, endpoint=False)
    alphas = beta / (1 - beta)
    print(f"Alphas range: {alphas.min(), alphas.max()}")

    #!# sample independent starting frames (different t=0) every tau_sample number of points
    #!# then use dci to remove the nearest 'dci' number of neighbors in time
    tau_sample = 100  #!# 100 frames * 200 ps = 20 ns
    sample_times = np.arange(start_idx, end_idx - tau_end - 1, tau_sample, dtype=int)
    #!# these are "independent" and consecutive chunks that are being considered
    Ntrajs = len(sample_times)
    print(sample_times)
    print(f"Total no. of sampling points/Ntrajs: {Ntrajs}")

    ## Extract the specific chunk
    # data_chunk = data[start_idx:end_idx]

    print(f"Loading chunk {ic}/{total_chunks}")
    print(f"Chunk indices: {start_idx} to {end_idx}")
    # print(f"Chunk shape: {data_chunk.shape}")

    data_array = data.copy()
    print(f"Final Data array shape:{data_array.shape}")

    #!# all values must be made +ve to work with DADApy distance function
    #!# so translate the values that can be negative
    #!# be careful with periodic variables, in that case don't do this treatment
    min_values = np.min(data_array, axis=(0, 1))
    print(f"Negative/Min values to be removed: {min_values}")

    ## subtract either 0 (if minimum is > 0), or the most -ve value
    # print(f"Most -ve values before removal: {np.min(data_array, axis=(0))}")
    # data_array = data_array[:, :] - np.minimum(0, min_values)
    # print(f"Verify removal of -ve values worked: {np.min(data_array, axis=(0))}")

    # find the largest value in the data-set (ignoring time column)
    # ensure that the period for non-periodic variables is > this maximum value
    large_period = 1000 * np.max(data_array[:, 1:])
    print(f"The large period is: {large_period}")

    # print(f"All colvar data is preloaded, and memory usage is: {data_array.nbytes/(1024**3)} GB")

    ## get current working directory
    current_directory = os.getcwd()

    ## Create a new directory called 'iib_data/830to1060us_sklearn_standard_top5'
    save_data_directory = os.path.join(current_directory, "iib_data", "830to1060us_sklearn_standard_top5")
    os.makedirs(save_data_directory, exist_ok=True)
    print("Saving IIB/IG data in directory:", save_data_directory)

    #!# construct X and Y at time=t0 (starting time)
    (
        pc1_t0,
        pc2_t0,
        pc3_t0,
        pc4_t0,
        pc5_t0,
    ) = construct_Xt_Yt(
        data_array=data_array,
        Ntrajs=Ntrajs,
        t=sample_times,
        E=E,
        tau_e=tau_e,
        Njobs=Njobs,
    )

    for tau in taus:
        print("Tau = ", tau)

        #!# construct X and Y at time=tau (the time we are checking causality at)
        (
            pc1_tau,
            pc2_tau,
            pc3_tau,
            pc4_tau,
            pc5_tau,
        ) = construct_Xt_Yt(
            data_array=data_array,
            Ntrajs=Ntrajs,
            t=sample_times + tau,
            E=E,
            tau_e=tau_e,
            Njobs=Njobs,
        )

        #!# PC1 <-> PC2
        iib__pc1_to_pc2 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc2_t0,
            effect_future=pc2_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc2_to_pc1 = scan_alphas(
            cause_present=pc2_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC3
        iib__pc1_to_pc3 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc3_t0,
            effect_future=pc3_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc3_to_pc1 = scan_alphas(
            cause_present=pc3_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC4
        iib__pc1_to_pc4 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc4_t0,
            effect_future=pc4_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc4_to_pc1 = scan_alphas(
            cause_present=pc4_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC5
        iib__pc1_to_pc5 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc5_t0,
            effect_future=pc5_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc5_to_pc1 = scan_alphas(
            cause_present=pc5_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC2 <-> PC3
        iib__pc2_to_pc3 = scan_alphas(
            cause_present=pc2_t0,
            effect_present=pc3_t0,
            effect_future=pc3_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc3_to_pc2 = scan_alphas(
            cause_present=pc3_t0,
            effect_present=pc2_t0,
            effect_future=pc2_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC2 <-> PC4
        iib__pc2_to_pc4 = scan_alphas(
            cause_present=pc2_t0,
            effect_present=pc4_t0,
            effect_future=pc4_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc4_to_pc2 = scan_alphas(
            cause_present=pc4_t0,
            effect_present=pc2_t0,
            effect_future=pc2_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC2 <-> PC5
        iib__pc2_to_pc5 = scan_alphas(
            cause_present=pc2_t0,
            effect_present=pc5_t0,
            effect_future=pc5_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc5_to_pc2 = scan_alphas(
            cause_present=pc5_t0,
            effect_present=pc2_t0,
            effect_future=pc2_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC3 <-> PC4
        iib__pc3_to_pc4 = scan_alphas(
            cause_present=pc3_t0,
            effect_present=pc4_t0,
            effect_future=pc4_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc4_to_pc3 = scan_alphas(
            cause_present=pc4_t0,
            effect_present=pc3_t0,
            effect_future=pc3_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC3 <-> PC5
        iib__pc3_to_pc5 = scan_alphas(
            cause_present=pc3_t0,
            effect_present=pc5_t0,
            effect_future=pc5_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc5_to_pc3 = scan_alphas(
            cause_present=pc5_t0,
            effect_present=pc3_t0,
            effect_future=pc3_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC4 <-> PC5
        iib__pc4_to_pc5 = scan_alphas(
            cause_present=pc4_t0,
            effect_present=pc5_t0,
            effect_future=pc5_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc5_to_pc4 = scan_alphas(
            cause_present=pc5_t0,
            effect_present=pc4_t0,
            effect_future=pc4_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        # create file to save data in
        save_data_file = os.path.join(
            save_data_directory,
            f"iib_IC-{ic}_tau-{tau}_k-{k}_E-{E}_dci-{dci}.p",
        )

        # save IIB calculation
        pickle.dump(
            [
                iib__pc1_to_pc2,
                iib__pc2_to_pc1,
                iib__pc1_to_pc3,
                iib__pc3_to_pc1,
                iib__pc1_to_pc4,
                iib__pc4_to_pc1,
                iib__pc1_to_pc5,
                iib__pc5_to_pc1,
                iib__pc2_to_pc3,
                iib__pc3_to_pc2,
                iib__pc2_to_pc4,
                iib__pc4_to_pc2,
                iib__pc2_to_pc5,
                iib__pc5_to_pc2,
                iib__pc3_to_pc4,
                iib__pc4_to_pc3,
                iib__pc3_to_pc5,
                iib__pc5_to_pc3,
                iib__pc4_to_pc5,
                iib__pc5_to_pc4,
            ],
            open(save_data_file, "wb"),
        )


##############################################################################################################
#!#!!!!!!!!!!!!!!!!!!!!!!! Read colvar.txt files and create time-delayed embeddings !!!!!!!!!!!!!!!!!!!!!!!#!#
##############################################################################################################
def return_time_delayed_embeddings(data, t, E, tau_e):
    """
    Extract time-delayed embeddings for a single starting time t.

    Args:
        data: Full dataset of shape (total_timesteps, Nfeatures)
        t: Starting time index (scalar)
        E: Embedding dimension
        tau_e: Embedding time lag

    Returns:
        embedded_array: Shape (Nfeatures, E) for the PCs with E time-delay embeddings
    """
    embed_times = np.arange(t, t + (E * tau_e), tau_e)

    #!# PC(1-5) - extract embeddings across time for each PC
    pc1 = data[embed_times, 0]  # Shape: (E,)
    pc2 = data[embed_times, 1]
    pc3 = data[embed_times, 2]
    pc4 = data[embed_times, 3]
    pc5 = data[embed_times, 4]

    embedded_array = np.array(
        [
            pc1,
            pc2,
            pc3,
            pc4,
            pc5,
        ]
    )  # Shape: (Nfeatures, E)

    return embedded_array


##############################################################################################################
##############################################################################################################


##############################################################################################################
#!#!!!!!!!!!!!!!!!!!!!!!!! Construct driving and driven variable matrices at time=t !!!!!!!!!!!!!!!!!!!!!!!#!#
##############################################################################################################
def construct_Xt_Yt(data_array, Ntrajs, t, E, tau_e, Njobs):
    """
    Construct time-delayed embeddings for all trajectories (starting times).

    Args:
        data_array: Full dataset of shape (total_timesteps, Nfeatures)
        Ntrajs: Number of independent starting times
        t: Array of starting time indices, shape (Ntrajs,)
        E: Embedding dimension
        tau_e: Embedding time lag
        Njobs: Number of parallel jobs

    Returns:
        Tuple of (pc1, pc2, pc3, pc4, pc5), each with shape (Ntrajs, E)
    """
    #!# Process each starting time in parallel
    embedded_results = Parallel(n_jobs=Njobs)(
        delayed(return_time_delayed_embeddings)(
            data=data_array,  # Pass the full data array
            t=t[itraj],  # Pass single starting time
            E=E,
            tau_e=tau_e,
        )
        for itraj in range(Ntrajs)
    )

    # embedded_results is a list of length Ntrajs, each element shape (Nfeatures, E)
    # Stack them to get shape (Ntrajs, Nfeatures, E)
    stacked = np.array(embedded_results)  # Shape: (Ntrajs, Nfeatures, E)

    # Extract each PC separately
    pc1 = stacked[:, 0, :]  # Shape: (Ntrajs, E)
    pc2 = stacked[:, 1, :]
    pc3 = stacked[:, 2, :]
    pc4 = stacked[:, 3, :]
    pc5 = stacked[:, 4, :]

    return (pc1, pc2, pc3, pc4, pc5)


##############################################################################################################
##############################################################################################################

if __name__ == "__main__":
    main()
