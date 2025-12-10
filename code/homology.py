import math
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def find_youngest(boundary_chain, simplex_dictionary):
    """
    Passed a boundary chain and a dictionary containing the index by filtration
    """
    max_idx = -1
    for face in boundary_chain:
        idx = simplex_dictionary[face][1]
        if idx > max_idx:
            max_idx = idx
    return max_idx


class Simplex:
    """
    A class implementing simplices for use in the
    """

    def __init__(self, vertices, degree):
        self.vertices = vertices
        self.dim = len(vertices) - 1
        self.deg = degree

    def __str__(self):
        return f"Vertices: {self.vertices}, degree: {self.deg}"

    def compute_reduced_boundary_chain(self, simplex_dictionary):
        """
        Computes the boundary chain for the simplex, given a simplex dictionary. The dictionary helps us by allowing us to check if the faces of the boundary are marked, which is crucial for telling if they birthed a cycle or not. See final report section 3 for justification
        """
        boundary_chain = set()

        if self.dim == 0:
            return boundary_chain

        for i, _ in enumerate(self.vertices):
            # look at all k-1
            face = tuple(self.vertices[:i] + self.vertices[i + 1 :])
            # only add marked faces
            if simplex_dictionary[face][0]:
                boundary_chain.add(face)

        return boundary_chain

    def mark(self, simplex_dictionary):
        """
        Marks that the simplex birthed a cycle in the simplex dictionary
        """
        simplex_dictionary[tuple(self.vertices)][0] = True

    def check_if_marked(self, simplex_dictionary):
        """
        Checks if the simplex is marked in the simplex dictionary
        """
        return simplex_dictionary[tuple(self.vertices)][0]


def create_persistence_complex(dataset, complex_type="alpha", max_dim=3):
    """
    Given a dataset, constructs a persistence complex out of that dataset. Must be either a Vietoris-rips or Alpha complex. Returns the complex itself, a dictionary with information about the simplices needed for computing PH, and the Gudhi simplex tree (Gudhi's representaiton of a persistence complex) for the complex. Gudhi tree is returned for testing
    """
    if complex_type == "alpha":
        complex = gd.AlphaComplex(points=dataset)
    elif complex_type == "rips":
        complex = gd.RipsComplex(points=dataset)
    else:
        raise TypeError("Complex type must be 'rips' or 'alpha'")

    # persistence complex itself
    pc = []
    # simplex dictionary - this allows for a very simple implementation of the boundary function, see Simplex class for details
    sd = {}

    # Create simplex tree in Gudhi
    stree = complex.create_simplex_tree()
    stree.prune_above_dimension(max_dim)

    # take the Gudhi simplex and process it into a list of simplices, each represented in the Simplex class. Note they are passed in by filtration order, so the index (which is stored in the simplex dictionary) tells us the relative age of simplices.
    for i, (vertices, squared_circumradii) in enumerate(stree.get_filtration()):
        deg = np.sqrt(squared_circumradii)
        s = Simplex(vertices, deg)
        pc.append(s)
        sd[tuple(vertices)] = [False, i]

    # return stree for testing
    return pc, sd, stree


def remove_dead_boundaries(simplex, simplex_dictionary, T):
    """
    Given a simplex, simplex dictionary, and array containing the "killers" of cycles, removes any elements of the boundary that have already been killed, and returns the remaining boundary. See final report section 3 for justification
    """
    # this removes all simplices that did not birth a cycle
    d = simplex.compute_reduced_boundary_chain(simplex_dictionary)

    while len(d) > 0:
        i = find_youngest(d, simplex_dictionary)
        if T[i] is None:
            # this means the simplex kills something, and we can exit and record that
            break
        # otherwise, that element has already been killed, so we remove anything that would have been killed along with it
        d = d ^ T[i]

    return d


def compute_intervals(
    persistence_complex, simplex_dictionary, max_dim=3, remove_small_intervals=True
):
    """
    Computes persistence homology intervals given a persistence complex, assumed to be in filtration order. We also assume the maximum dimension of the persistence complex is one more than the max dimension of the algorithm. Also requires a dictionary that tells if simplices birthed cycles, and also what their filtration index was. Returns the set of intervals in the persistence complex. See section 3 of final report for explanation.
    """

    # initialize data structures
    interval_sets = [[] for _ in range(max_dim)]
    T = [None for _ in range(len(persistence_complex))]

    # the simplex is in filtration order, which is important for the use of find_youngest
    for j, simplex_j in enumerate(persistence_complex):
        d = remove_dead_boundaries(simplex_j, simplex_dictionary, T)
        if len(d) == 0:
            # this means that the k-simplex did not "kill off" a k-1 simplex, so we mark it as a creator of a k-cycle
            simplex_j.mark(simplex_dictionary)
        else:
            # otherwise we want to record that it killed a cycle
            i = find_youngest(d, simplex_dictionary)
            simplex_i = persistence_complex[i]
            k = simplex_i.dim
            T[i] = d
            if simplex_j.deg > simplex_i.deg or not remove_small_intervals:
                interval_sets[k].append((simplex_i.deg, simplex_j.deg))

    for j, simplex in enumerate(persistence_complex):
        if simplex.check_if_marked(simplex_dictionary) and T[j] is None:
            # this means the simplex was marked as creating a cycle, but never marked as having that cycle killed. So we add an interval to infinity here
            k = simplex.dim
            interval_sets[k].append((simplex.deg, float("inf")))

    return interval_sets


def display_intervals(persistence_interval_set, max_intervals=20, min_distance=0.01):
    """
    Creates barcode display for given persistence homology intervals for a single dimension
    """
    dims = len(persistence_interval_set)

    # associate intervals with their dimensions and length
    interval_lengths = {}
    interval_dims = {}
    intervals = []
    end = 0
    for i, interval_set in enumerate(persistence_interval_set):
        for a, b in interval_set:
            if b > end and not math.isinf(b):
                end = b
            if b - a > min_distance:
                interval_lengths[(a, b)] = b - a
                # save information about dimension of interval. Honestly, this is kind of sloppy, but I think the odds of getting two intervals in different dimensions with the exact same persistence values is pretty low
                interval_dims[(a, b)] = i
                intervals.append((a, b))
    # get only max intervals
    intervals.sort(key=lambda interval: interval_lengths[interval], reverse=True)
    intervals = intervals[:max_intervals]

    num_intervals = len(intervals)

    fig, ax = plt.subplots(figsize=(6, num_intervals * 0.2))

    # color by deimension
    color_map = {0: "red", 1: "blue", 2: "green"}
    legend_elements = [
        Patch(facecolor="red", edgecolor="red", label="Dimension 0"),
        Patch(facecolor="blue", edgecolor="blue", label="Dimension 1"),
        Patch(facecolor="green", edgecolor="green", label="Dimension 2"),
    ]

    ax.barh(
        np.arange(num_intervals),
        [b - a if b < end else end - a for (a, b) in intervals],
        left=[a for (a, _) in intervals],
        height=0.7,
        color=[color_map[interval_dims[(a, b)]] for (a, b) in intervals],
    )
    ax.legend(handles=legend_elements)

    ax.get_yaxis().set_visible(False)

    plt.show()
