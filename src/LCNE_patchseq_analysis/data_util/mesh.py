"""
Load and plot LC meshgrid
"""

import numpy as np


def plot_mesh(
    ax,
    allmeshes,
    direction: str = "coronal",
    meshcol: str = "lightgray",
    both_sides: bool = True,
    midline: float = 5700.0,
    alpha: float = 0.4,
) -> None:
    """Plot mesh(es) on the given Matplotlib axis.

    Args:
        ax: Matplotlib Axes to draw on.
        allmeshes: dict[str, trimesh.Trimesh] or a single trimesh.Trimesh instance.
        direction: 'coronal' (uses vertex index 2 for x) or 'sagittal' (index 0).
        meshcol: Base color for mesh wireframe.
        both_sides: When True and direction=='coronal', also plot a reflected copy
            across the provided midline (default 5700) to show contralateral side.
        midline: Midline x-coordinate used for reflection when both_sides is True.
    """
    import trimesh

    ax.set_aspect("equal")
    i = 2 if direction == "coronal" else 0

    def _plot_single(mesh, label=None):
        ax.triplot(
            mesh.vertices.T[i],
            mesh.vertices.T[1],
            mesh.faces,
            alpha=alpha,
            label=label,
            color=meshcol,
        )

    def _reflect_and_plot(mesh, label_suffix="reflected"):
        # Reflect along axis index i around midline: x_reflected = 2*midline - x
        verts = mesh.vertices.copy()
        verts[:, i] = 2 * midline - verts[:, i]

        class _Tmp:
            vertices = verts
            faces = mesh.faces

        _plot_single(_Tmp(), label=None)

    if isinstance(allmeshes, dict):
        for k, mesh in allmeshes.items():
            _plot_single(mesh, label=k)
            if both_sides and direction == "coronal":
                _reflect_and_plot(mesh)
    elif isinstance(allmeshes, trimesh.Trimesh):
        _plot_single(allmeshes, label=None)
        if both_sides and direction == "coronal":
            _reflect_and_plot(allmeshes)
    else:
        print("wrong mesh input")

    ax.invert_yaxis()


def trimesh_to_bokeh_data(mesh, direction: str = "coronal", both_sides: bool = True) -> dict:
    """
    Project mesh to 2D and prepare Bokeh plotting data
    parameter direction: select index to choose coordinate ('c' uses index 2, otherwise index 0)
    parameter both_sides: if True, returns data for both sides of the mesh
    """
    i = 2 if direction == "coronal" else 0
    x = mesh.vertices[:, i]
    y = mesh.vertices[:, 1]  # Y always in axis 1
    faces = mesh.faces

    # Each triangle becomes a patch
    xs = [x[face].tolist() for face in faces]
    ys = [y[face].tolist() for face in faces]

    # If direction=="coronal" and both_sides=True, add the patches for the other side
    # (flip horizontal axis relative to the center of the brain, i.e., 5700)
    if direction == "coronal" and both_sides:
        xs = xs + (5700 * 2 - np.array(xs)).tolist()
        ys = ys + ys

    return dict(xs=xs, ys=ys)


def add_mesh_to_k3d(
    plot,
    mesh,
    color: int = 0xAAAAAA,
    opacity: float = 0.3,
    both_sides: bool = True,
    midline: float = 5700.0,
):
    """Add LC mesh to k3d plot.

    Args:
        plot: k3d.Plot object
        mesh: trimesh.Trimesh object
        color: hex color for mesh
        opacity: transparency (0-1)
        both_sides: if True, also add reflected mesh across midline
        midline: Z coordinate for reflection (default 5700 for L/R hemisphere boundary)
    """
    import k3d

    vertices = mesh.vertices.astype("float32")
    indices = mesh.faces.astype("uint32")

    # Add original mesh
    plot += k3d.mesh(vertices, indices, color=color, opacity=opacity, name="LC_mesh")

    # Add reflected mesh if both_sides
    if both_sides:
        vertices_reflected = vertices.copy()
        vertices_reflected[:, 2] = 2 * midline - vertices_reflected[:, 2]  # flip Z axis
        plot += k3d.mesh(
            vertices_reflected, indices, color=color, opacity=opacity, name="LC_mesh_reflected"
        )

    return plot


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3

    # Example usage
    mesh = load_mesh_from_s3()
    fig, ax = plt.subplots()
    plot_mesh(ax, mesh, direction="coronal", meshcol="lightgray")
    plt.show()
