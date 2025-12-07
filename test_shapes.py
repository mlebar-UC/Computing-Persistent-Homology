import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files
import trimesh
import numpy as np
from datasets import load_dataset


def plot_3d_shape(X, Y, Z, cmap="viridis"):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes(projection="3d")
    ax.scatter(X, Y, Z, cmap=cmap)
    plt.show()


def plot_surface(X, Y, Z, surface, cmap="viridis"):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes(projection="3d")
    ax.plot_trisurf(X, Y, Z, triangles=surface.faces, cmap=cmap)
    plt.show()


def get_genus_g_surface(g):
    file_names = list_repo_files("appliedgeometry/EuLearn", repo_type="dataset")
    path = None

    for fn in file_names:
        maybe_g = fn[1]
        if maybe_g.isdigit() and int(maybe_g) == g:
            path = hf_hub_download("appliedgeometry/EuLearn", fn, repo_type="dataset")
            break

    surface = trimesh.load(path)
    return surface, surface.sample(1000)
