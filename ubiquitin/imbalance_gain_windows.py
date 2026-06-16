import numpy as np
from joblib import Parallel, delayed


def _compute_dist2_matrix(data, mask):
    """Computes the matrix of squared pair-wise Euclidean distances.

    Args:
        data (np.array(float)): matrix of shape (n_points, n_dim)
        mask (np.array(bool)): mask to discard distances around the diagonal,
            produced by function '_return_mask'

    Returns:
        dist2_matrix (np.array(float)): matrix of squared Euclidean distances,
            with shape (n_points, n_points - 2*discard_close_ind - 1) (where
            'discard_close_ind' is an argument of '_return_mask')
    """
    diffs = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    dist2_matrix = np.sum(diffs * diffs, axis=-1)
    dist2_matrix = dist2_matrix[mask].reshape((dist2_matrix.shape[0], -1))

    return dist2_matrix


def _compute_rank_matrix(dist2_matrix):
    """Computes the matrix of ranks.

    Args:
        dist2_matrix (np.array(float)): matrix of shape (n_rows, n_columns),
            where n_columns <= n_rows - 1 (diagonal should be always excluded)

    Returns:
        rank_matrix (np.array(float)): matrix of ranks, defined from 1 to
            n_columns
    """
    rank_matrix = dist2_matrix.argsort(axis=1).argsort(axis=1) + 1

    return rank_matrix


def compute_info_imbalance_causality(dists2_cause_present, dists2_effect_present, ranks_effect_future, alpha, k):
    """
    Computes for a single scaling parameter alpha the Information Imbalance
        Delta( (alpha*X(0), Y(0)) -> Y(tau)).

    Args:
        dists2_cause_present (np.array(float)): matrix of distances in space X(0)
        dists2_effect_present (np.array(float)): matrix of distances in space Y(0)
        ranks_effect_future (np.ndarray): matrix of ranks in space Y(tau)
        alpha (float): scaling parameter
        k (int): number of nearest neighbors considered in the Information Imbalance calculation

    Returns:
        info_imbalance (float): value of the Information Imbalance
    """
    N_rows = dists2_cause_present.shape[0]
    max_rank = dists2_cause_present.shape[1]
    ranks_present = _compute_rank_matrix(alpha * alpha * dists2_cause_present + dists2_effect_present)
    mask = np.where(ranks_present <= k, 1.0, 0.0)
    info_imbalance = 2.0 / (N_rows * (max_rank + 1) * k) * np.sum(ranks_effect_future * mask)

    return info_imbalance


def _return_mask(npoints, discard_close_ind=0):
    """Returns a square boolean mask with False on the diagonals, and True elsewhere.

    Args:
        npoints (int): number of rows and columns of the mask matrix.
        discard_close_ind (int): defines the diagonals filled with False, with offset between
            -discard_close_ind (below the main diagonal) and +discard_close_ind (above the main
            diagonal). If discard_close_ind==0 (default), only the main diagonal is filled with
            False and discarded.

    Returns:
        mask (np.array(float)): square boolean matrix of shape (npoints, npoints).
    """
    mask = np.abs(np.arange(npoints)[:, np.newaxis] - np.arange(npoints)[np.newaxis, :])
    mask = mask > discard_close_ind

    if discard_close_ind > 0:
        # more columns than necessary discarded for starting and final rows, for shape compatibility
        first_rows = np.concatenate(
            (
                np.zeros(2 * discard_close_ind + 1),
                np.ones(npoints - 2 * discard_close_ind - 1),
            )
        )
        last_rows = np.concatenate(
            (
                np.ones(npoints - 2 * discard_close_ind - 1),
                np.zeros(2 * discard_close_ind + 1),
            )
        )
        mask[:discard_close_ind] = first_rows
        mask[-discard_close_ind:] = last_rows
    return mask


def _return_mask_new(npoints, discard_close_ind=0):
    """Returns a square boolean mask with False on the diagonals, and True elsewhere.
    This differs from the other '_return_mask' function by filling the extra zeros
    'periodically' in the starting/ending rows, by adding the extra zeros partly at
    the beginning of the rows, and partly at the end. In the other function, the extra zeros
    are added consecutively.

    Args:
        npoints (int): number of rows and columns of the mask matrix.
        discard_close_ind (int): defines the diagonals filled with False, with offset between
            -discard_close_ind (below the main diagonal) and +discard_close_ind (above the main
            diagonal). If discard_close_ind==0 (default), only the main diagonal is filled with
            False and discarded.

    Returns:
        mask (np.array(float)): square boolean matrix of shape (npoints, npoints).
    """
    mask = np.ones((npoints, npoints), dtype=bool)
    for i in range(0, npoints):
        for j in range(i - discard_close_ind, i + discard_close_ind + 1):
            mask[i, j % npoints] = False
    return mask


def scan_alphas(cause_present, effect_present, effect_future, alphas, k=1, discard_close_ind=0, n_jobs=1):
    """
    Computes the Information Imbalance
        Delta( (alpha*X(0), Y(0)) -> Y(tau))
    by parallelizing the loop over different values of the scaling parameter.

    Args:
        cause_present (np.array(float)): array of shape (n_points, n_dims) defining X(0)
            in the feature space
        effect_present (np.array(float)): array of shape (n_points, n_dims) defining Y(0)
            in the feature space
        effect_future (np.array(float)): array of shape (n_points, n_dims) defining Y(tau)
            in the feature space
        alphas (np.array(float)): scaling parameters scanned in the loop
        k (int): number of nearest neighbors considered in the Information Imbalance calculation
        discard_close_ind (int): defines the of the window inside which distances are not
            computed. In particular, distances are only computed among samples t1, t2 such that
            |t1 - t2| > 2*discard_close_ind + 1.
        n_jobs (int): the number of jobs to run in parallel

    Returns:
        info_imbalance (float): value of the Information Imbalance
    """
    npoints = cause_present.shape[0]
    mask = _return_mask(npoints=npoints, discard_close_ind=discard_close_ind)

    dists2_cause_present = _compute_dist2_matrix(cause_present, mask=mask)
    dists2_effect_present = _compute_dist2_matrix(effect_present, mask=mask)
    ranks_effect_future = _compute_rank_matrix(_compute_dist2_matrix(effect_future, mask=mask))

    info_imbalances = Parallel(n_jobs=n_jobs)(
        delayed(compute_info_imbalance_causality)(
            dists2_cause_present=dists2_cause_present,
            dists2_effect_present=dists2_effect_present,
            ranks_effect_future=ranks_effect_future,
            alpha=alpha,
            k=k,
        )
        for alpha in alphas
    )

    return np.array(info_imbalances)


def return_nn_indices(data, discard_close_ind=0):
    npoints = data.shape[0]
    mask = _return_mask(npoints=npoints, discard_close_ind=discard_close_ind)

    dists2_matrix = _compute_dist2_matrix(data, mask=mask)
    rank_matrix = _compute_rank_matrix(dists2_matrix)

    nn_indices = np.argmin(rank_matrix, axis=1)
    return nn_indices


def compute_imbalance_gain(info_imbalances):
    """Computes the Imbalance Gain
        ( Delta(alpha=0) - min_alpha Delta(alpha) ) / Delta(alpha=0)

    Args:
        info_imbalances (np.ndarray): Delta(alpha) for the values of, output of 'scan_alphas'
                                              of the putative driver system at time 0
    Returns:
        imbalance_gain (float): value of the Imbalance Gain

        optimal_alpha_index (int): index of the scaling parameter minimizing Delta(alpha)
    """

    imbalance_gain = (info_imbalances[0] - np.min(info_imbalances)) / info_imbalances[0]
    optimal_alpha_index = np.argmin(info_imbalances)

    return imbalance_gain, optimal_alpha_index
