import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files
import trimesh
import numpy as np


def plot_3d_points(X, Y, Z, cmap="viridis"):
    """
    Plots a point cloud in 3d space.
    """
    _ = plt.figure(figsize=(5, 4))
    ax = plt.axes(projection="3d")
    ax.scatter(X, Y, Z, cmap=cmap)
    plt.show()


def plot_3d_shape(shape):
    """
    Unpacks points in from a shape to call plot_3d_points.
    """
    plot_3d_points(shape[:, 0], shape[:, 1], shape[:, 2])


def plot_surface(X, Y, Z, surface, cmap="viridis"):
    """
    Plots a 3d trimesh surface. Assumes the surface is from the trimesh library
    """
    _ = plt.figure(figsize=(5, 4))
    ax = plt.axes(projection="3d")
    ax.plot_trisurf(X, Y, Z, triangles=surface.faces, cmap=cmap)
    plt.show()


def get_genus_g_surface(g):
    """
    Produces a surface of genus g for testing. Returns a surface that can be plotted (this was mainly because it's fun to look at these surfaces with fancy cmaps than anything really practical) and a set of points sampled from that surface.
    """
    file_names = list_repo_files("appliedgeometry/EuLearn", repo_type="dataset")
    path = None

    for fn in file_names:
        maybe_g = fn[1]
        if maybe_g.isdigit() and int(maybe_g) == g:
            path = hf_hub_download("appliedgeometry/EuLearn", fn, repo_type="dataset")
            break

    surface = trimesh.load(path)
    return surface, surface.sample(1000)


def sample_points_from_sphere(center, r=1, num_points=1000):
    """
    Approach taken from https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    Very simple and clever idea, to get points on a sphere, generate vectors randomly and then normalize them.
    """
    vectors = np.random.randn(num_points, 3)
    vectors = r * vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors += center

    return vectors


def get_n_spheres(n):
    """
    Returns a point cloud samples from n-spheres, any pair of which intersect at at most point.
    """
    # generate n points at least distance 2 from each other
    # sample from spheres
    points = sample_points_from_sphere([0, 0, 0])
    for i in range(1, n):
        points = np.concatenate(
            (points, sample_points_from_sphere([2 * i, 0, 0])), axis=0
        )
    return points
