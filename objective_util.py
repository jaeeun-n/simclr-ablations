"""Objective related functions used in experiments."""

import tensorflow as tf
import tensorflow_probability as tfp



def euclidean_dist(A, B=None):
    """
    Computes pairwise euclidean distances between each elements of A and each elements of B
    and converts to similarities.
    Args:
      A,    [n,d] matrix
      B,    [n,d] matrix
    Returns:
      D,    [n,n] matrix of pairwise similarities
    """

    if B is None:
        B = A

    v = tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1)
    p1 = tf.reshape(tf.reduce_sum(v, axis=1), (-1, 1))
    v = tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1])
    p2 = tf.transpose(tf.reshape(tf.reduce_sum(v, axis=1), (-1, 1)))
    distance_matrix = tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True)
    similarity_matrix = tf.map_fn(convert_to_similarity, distance_matrix)
    return (similarity_matrix)


def mahalanobis_dist(A, B=None):
    """
        Computes pairwise mahalanobis distances between each elements of A and each elements of B
        and converts to similarities.
        Args:
          A,    [n,d] matrix
          B,    [n,d] matrix
        Returns:
          D,    [n,n] matrix of pairwise similarities
    """

    if B is None:
        B = A
        S = tfp.stats.covariance(A)
    else:
        AB = tf.concat([A, B], 0)
        S = tfp.stats.covariance(AB)

    invS = tf.linalg.inv(S)
    S_half = tf.linalg.cholesky(invS)
    A_star = tf.matmul(A, S_half)
    B_star = tf.matmul(B, S_half)
    similarity_matrix = euclidean_dist(A_star, B_star)
    return (similarity_matrix)


def convert_to_similarity(x):
    return 1 / (x + 1)