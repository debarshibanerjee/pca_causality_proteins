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
        default="pc_data/r2-70_650us_sklearn_standard_scores.npy",
    )
    args = parser.parse_args()
    ic = args.ic
    dci = args.dci  # this sets the discarded window size [t-discard_close_ind, t+discard_close_ind]

    #!# number of jobs to parallelize over
    Njobs = 56

    #!# how many chunks to break the trajectory into
    #!# total size = 650 microsec, so chunk_size=650 microsec/total_chunks
    total_chunks = 20

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
    pc11 = all_data[:, 10]
    pc12 = all_data[:, 11]
    pc13 = all_data[:, 12]
    pc14 = all_data[:, 13]
    pc15 = all_data[:, 14]
    pc16 = all_data[:, 15]
    pc17 = all_data[:, 16]
    pc18 = all_data[:, 17]
    pc19 = all_data[:, 18]
    pc20 = all_data[:, 19]

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
    pc11_new = pc11 / pc11.std(axis=0)
    pc12_new = pc12 / pc12.std(axis=0)
    pc13_new = pc13 / pc13.std(axis=0)
    pc14_new = pc14 / pc14.std(axis=0)
    pc15_new = pc15 / pc15.std(axis=0)
    pc16_new = pc16 / pc16.std(axis=0)
    pc17_new = pc17 / pc17.std(axis=0)
    pc18_new = pc18 / pc18.std(axis=0)
    pc19_new = pc19 / pc19.std(axis=0)
    pc20_new = pc20 / pc20.std(axis=0)

    # print(f"Data shapes: all_data={all_data.shape}, PC1={pc1.shape}, PC2={pc2.shape}")

    #!# Concatenate all data together
    data = np.column_stack(
        [
            pc1_new,
            pc2_new,
            pc3_new,
            pc4_new,
            pc5_new,
            pc6_new,
            pc7_new,
            pc8_new,
            pc9_new,
            pc10_new,
            pc11_new,
            pc12_new,
            pc13_new,
            pc14_new,
            pc15_new,
            pc16_new,
            pc17_new,
            pc18_new,
            pc19_new,
            pc20_new,
        ]
    )
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
    #!# 1 frame = 200 ps (in NVT sims)
    tau_e = 1

    #!# how often we check for causality (in frames) == spacing between tau's
    ## every 1 frame = 1 frames * 200 ps = 200 ps
    tau_gap = 25  #!# 5 ns

    #!# the 'tau' to start from in the range of values in (taus)
    ## useful if a job has to be restarted and we want to skip the already processed 'tau' values
    tau_start = 0

    #!# max duration of time lag; 1000 frames = 200 ns
    tau_end = 2500  #!# 0.5 us

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

    ## Create a new directory called 'iib_data'
    save_data_directory = os.path.join(current_directory, "iib_data_r2-70_PC1")
    os.makedirs(save_data_directory, exist_ok=True)
    print("Saving IIB/IG data in directory:", save_data_directory)

    #!# construct X and Y at time=t0 (starting time)
    (
        pc1_t0,
        pc2_t0,
        pc3_t0,
        pc4_t0,
        pc5_t0,
        pc6_t0,
        pc7_t0,
        pc8_t0,
        pc9_t0,
        pc10_t0,
        pc11_t0,
        pc12_t0,
        pc13_t0,
        pc14_t0,
        pc15_t0,
        pc16_t0,
        pc17_t0,
        pc18_t0,
        pc19_t0,
        pc20_t0,
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
            pc6_tau,
            pc7_tau,
            pc8_tau,
            pc9_tau,
            pc10_tau,
            pc11_tau,
            pc12_tau,
            pc13_tau,
            pc14_tau,
            pc15_tau,
            pc16_tau,
            pc17_tau,
            pc18_tau,
            pc19_tau,
            pc20_tau,
        ) = construct_Xt_Yt(
            data_array=data_array,
            Ntrajs=Ntrajs,
            t=sample_times + tau,
            E=E,
            tau_e=tau_e,
            Njobs=Njobs,
        )

        #!# PC1 <-> PC6
        iib__pc1_to_pc6 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc6_t0,
            effect_future=pc6_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc6_to_pc1 = scan_alphas(
            cause_present=pc6_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC7
        iib__pc1_to_pc7 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc7_t0,
            effect_future=pc7_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc7_to_pc1 = scan_alphas(
            cause_present=pc7_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC8
        iib__pc1_to_pc8 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc8_t0,
            effect_future=pc8_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc8_to_pc1 = scan_alphas(
            cause_present=pc8_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC9
        iib__pc1_to_pc9 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc9_t0,
            effect_future=pc9_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc9_to_pc1 = scan_alphas(
            cause_present=pc9_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC10
        iib__pc1_to_pc10 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc10_t0,
            effect_future=pc10_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc10_to_pc1 = scan_alphas(
            cause_present=pc10_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC11
        iib__pc1_to_pc11 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc11_t0,
            effect_future=pc11_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc11_to_pc1 = scan_alphas(
            cause_present=pc11_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC12
        iib__pc1_to_pc12 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc12_t0,
            effect_future=pc12_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc12_to_pc1 = scan_alphas(
            cause_present=pc12_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC13
        iib__pc1_to_pc13 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc13_t0,
            effect_future=pc13_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc13_to_pc1 = scan_alphas(
            cause_present=pc13_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC14
        iib__pc1_to_pc14 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc14_t0,
            effect_future=pc14_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc14_to_pc1 = scan_alphas(
            cause_present=pc14_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC15
        iib__pc1_to_pc15 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc15_t0,
            effect_future=pc15_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc15_to_pc1 = scan_alphas(
            cause_present=pc15_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC16
        iib__pc1_to_pc16 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc16_t0,
            effect_future=pc16_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc16_to_pc1 = scan_alphas(
            cause_present=pc16_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC17
        iib__pc1_to_pc17 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc17_t0,
            effect_future=pc17_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc17_to_pc1 = scan_alphas(
            cause_present=pc17_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC18
        iib__pc1_to_pc18 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc18_t0,
            effect_future=pc18_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc18_to_pc1 = scan_alphas(
            cause_present=pc18_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC19
        iib__pc1_to_pc19 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc19_t0,
            effect_future=pc19_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc19_to_pc1 = scan_alphas(
            cause_present=pc19_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )

        #!# PC1 <-> PC20
        iib__pc1_to_pc20 = scan_alphas(
            cause_present=pc1_t0,
            effect_present=pc20_t0,
            effect_future=pc20_tau,
            alphas=alphas,
            discard_close_ind=dci,
            k=k,
            n_jobs=Njobs,
        )
        iib__pc20_to_pc1 = scan_alphas(
            cause_present=pc20_t0,
            effect_present=pc1_t0,
            effect_future=pc1_tau,
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
                iib__pc1_to_pc6,
                iib__pc6_to_pc1,
                iib__pc1_to_pc7,
                iib__pc7_to_pc1,
                iib__pc1_to_pc8,
                iib__pc8_to_pc1,
                iib__pc1_to_pc9,
                iib__pc9_to_pc1,
                iib__pc1_to_pc10,
                iib__pc10_to_pc1,
                iib__pc1_to_pc11,
                iib__pc11_to_pc1,
                iib__pc1_to_pc12,
                iib__pc12_to_pc1,
                iib__pc1_to_pc13,
                iib__pc13_to_pc1,
                iib__pc1_to_pc14,
                iib__pc14_to_pc1,
                iib__pc1_to_pc15,
                iib__pc15_to_pc1,
                iib__pc1_to_pc16,
                iib__pc16_to_pc1,
                iib__pc1_to_pc17,
                iib__pc17_to_pc1,
                iib__pc1_to_pc18,
                iib__pc18_to_pc1,
                iib__pc1_to_pc19,
                iib__pc19_to_pc1,
                iib__pc1_to_pc20,
                iib__pc20_to_pc1,
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

    #!# PC(1-20) - extract embeddings across time for each PC
    pc1 = data[embed_times, 0]  # Shape: (E,)
    pc2 = data[embed_times, 1]
    pc3 = data[embed_times, 2]
    pc4 = data[embed_times, 3]
    pc5 = data[embed_times, 4]
    pc6 = data[embed_times, 5]
    pc7 = data[embed_times, 6]
    pc8 = data[embed_times, 7]
    pc9 = data[embed_times, 8]
    pc10 = data[embed_times, 9]
    pc11 = data[embed_times, 10]
    pc12 = data[embed_times, 11]
    pc13 = data[embed_times, 12]
    pc14 = data[embed_times, 13]
    pc15 = data[embed_times, 14]
    pc16 = data[embed_times, 15]
    pc17 = data[embed_times, 16]
    pc18 = data[embed_times, 17]
    pc19 = data[embed_times, 18]
    pc20 = data[embed_times, 19]

    embedded_array = np.array(
        [
            pc1,
            pc2,
            pc3,
            pc4,
            pc5,
            pc6,
            pc7,
            pc8,
            pc9,
            pc10,
            pc11,
            pc12,
            pc13,
            pc14,
            pc15,
            pc16,
            pc17,
            pc18,
            pc19,
            pc20,
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
        Tuple of PC embeddings, each with shape (Ntrajs, E)
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
    pc6 = stacked[:, 5, :]
    pc7 = stacked[:, 6, :]
    pc8 = stacked[:, 7, :]
    pc9 = stacked[:, 8, :]
    pc10 = stacked[:, 9, :]
    pc11 = stacked[:, 10, :]
    pc12 = stacked[:, 11, :]
    pc13 = stacked[:, 12, :]
    pc14 = stacked[:, 13, :]
    pc15 = stacked[:, 14, :]
    pc16 = stacked[:, 15, :]
    pc17 = stacked[:, 16, :]
    pc18 = stacked[:, 17, :]
    pc19 = stacked[:, 18, :]
    pc20 = stacked[:, 19, :]

    return (
        pc1,
        pc2,
        pc3,
        pc4,
        pc5,
        pc6,
        pc7,
        pc8,
        pc9,
        pc10,
        pc11,
        pc12,
        pc13,
        pc14,
        pc15,
        pc16,
        pc17,
        pc18,
        pc19,
        pc20,
    )


##############################################################################################################
##############################################################################################################

if __name__ == "__main__":
    main()
