"""Objective related functions used in ablation studies. (author: jaeeun-n)"""

import tensorflow as tf
import tensorflow_probability as tfp


def euclidean_dist(A, B=None):
    """
    Computes pairwise squared euclidean distances between each element of A and each element of B
    and converts to similarities.
    The Code for computing the pairwise distances is taken from https://fairyonice.github.io/mahalanobis-tf2.html.

    Args:
      A:    [n,d] matrix.
      B:    [n,d] matrix.

    Returns:
      D:    [n,n] matrix of pairwise similarities.
    """
    if B is None:
        B = A

    v = tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1)
    p1 = tf.reshape(tf.reduce_sum(v, axis=1), (-1, 1))
    v = tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1])
    p2 = tf.transpose(tf.reshape(tf.reduce_sum(v, axis=1), (-1, 1)))
    distance_matrix = tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True)
    similarity_matrix = tf.map_fn(convert_to_similarity, distance_matrix)
    return similarity_matrix


def mahalanobis_dist(A, B=None):
    """
        Computes pairwise squared mahalanobis distances between each element of A and each element of B
        and converts to similarities.
        The Code for computing the pairwise distances is taken from https://fairyonice.github.io/mahalanobis-tf2.html.

        Args:
          A:    [n,d] matrix.
          B:    [n,d] matrix.

        Returns:
          D:    [n,n] matrix of pairwise similarities.
    """
    if B is None:
        B = A
        S = tfp.stats.covariance(A)
    else:  # if two matrices are given, assume that columns in A and B are from the same distribution
        AB = tf.concat([A, B], 0)
        S = tfp.stats.covariance(AB)

    invS = tf.linalg.inv(S)
    S_half = tf.linalg.cholesky(invS)
    A_star = tf.matmul(A, S_half)
    B_star = tf.matmul(B, S_half)
    similarity_matrix = euclidean_dist(A_star, B_star)
    return similarity_matrix


def convert_to_similarity(x):
    return 1 / (x + 1)
