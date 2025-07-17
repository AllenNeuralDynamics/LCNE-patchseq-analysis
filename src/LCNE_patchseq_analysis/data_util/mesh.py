"""
Load and plot LC meshgrid
"""


def plot_mesh(ax, allmeshes, direction: str = "coronal", meshcol="lightgray") -> None:
    """
    Plot the three meshes on the given axis.
    parameter direction: select index to choose coordinate ('c' uses index 2, otherwise index 0)
    allmeshes is a dictionary now
    """
    import trimesh

    ax.set_aspect("equal")
    i = 2 if direction == "coronal" else 0
    if isinstance(allmeshes, dict):  # , trimesh.Trimesh
        for k, mesh in allmeshes.items():
            ax.triplot(
                mesh.vertices.T[i],
                mesh.vertices.T[1],
                mesh.faces,
                alpha=0.4,
                label=k,
                color=meshcol,
            )
    elif isinstance(allmeshes, trimesh.Trimesh):
        ax.triplot(
            allmeshes.vertices.T[i],
            allmeshes.vertices.T[1],
            allmeshes.faces,
            alpha=0.4,
            color=meshcol,
        )
    else:
        print("wrong mesh input")

    ax.invert_yaxis()


def trimesh_to_bokeh_data(mesh, direction: str = "coronal"):
    """
    Project mesh to 2D and prepare Bokeh plotting data
    """
    i = 2 if direction == "coronal" else 0
    x = mesh.vertices[:, i]
    y = mesh.vertices[:, 1]  # Y always in axis 1
    faces = mesh.faces

    # Each triangle becomes a patch
    xs = [x[face].tolist() for face in faces]
    ys = [y[face].tolist() for face in faces]

    return dict(xs=xs, ys=ys)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3

    # Example usage
    mesh = load_mesh_from_s3()
    fig, ax = plt.subplots()
    plot_mesh(ax, mesh, direction="coronal", meshcol="lightgray")
    plt.show()
