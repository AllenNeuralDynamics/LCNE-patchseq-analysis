from typing import List, Optional, Tuple

import k3d
import matplotlib.colors as mcolors
import navis as nv
import numpy as np


def _split_paths_by_backtrack(df) -> List[np.ndarray]:
    """Split into polylines: if current row's parent_id != previous row's node_id,
    we 'backtrack' → start a new polyline. Roots (parent_id < 0) also start a new polyline.
    Returns list of (Ni,3) float32 arrays."""
    df = df[["node_id", "x", "y", "z", "parent_id"]].copy()
    df["node_id"] = df["node_id"].astype(int)
    df["parent_id"] = df["parent_id"].astype(int)
    df = df.sort_values("node_id").reset_index(drop=True)

    coords = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    id2row = {int(nid): i for i, nid in enumerate(df["node_id"].to_numpy())}

    paths: List[np.ndarray] = []
    current: List[np.ndarray] = []
    last_node_id: Optional[int] = None

    for _, r in df.iterrows():
        nid = int(r["node_id"])
        pid = int(r["parent_id"])
        if pid < 0:
            if len(current) >= 2:
                paths.append(np.array(current, dtype=np.float32))
            current = [coords[id2row[nid]]]
        else:
            p_xyz = coords[id2row[pid]]
            c_xyz = coords[id2row[nid]]
            if last_node_id is None or pid == last_node_id:
                if not current:
                    current = [p_xyz]
                if not np.allclose(current[-1], p_xyz):
                    current.append(p_xyz)
                current.append(c_xyz)
            else:
                if len(current) >= 2:
                    paths.append(np.array(current, dtype=np.float32))
                current = [p_xyz, c_xyz]
        last_node_id = nid

    if len(current) >= 2:
        paths.append(np.array(current, dtype=np.float32))
    return [p for p in paths if len(p) >= 2]


def plot_swc_k3d(  # NoQA: C901
    swc_path: str,
    color: int = 0x00AAFF,
    width: float = 0.3,
    soma_color: int = 0xFF0000,
    background_color: Optional[int] = None,
    save_html: Optional[str] = None,
    plot: Optional[k3d.Plot] = None,
    offset: Optional[np.ndarray] = None,
) -> Tuple[k3d.Plot, List[np.ndarray]]:
    """Render SWC in k3d with a single color; soma is defined ONLY as rows with parent_id==-1,
    drawn using the real SWC radius.

    Args:
        offset: Optional (x, y, z) offset to apply to all coordinates
        plot: Optional existing k3d.Plot to add to (creates new if None)
    """
    # Load (first neuron if list)
    n = nv.read_swc(swc_path)
    if isinstance(n, nv.core.NeuronList):
        n = n[0]
    df = n.nodes.copy()

    # Required columns
    needed = {"node_id", "x", "y", "z", "parent_id", "radius"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"SWC missing columns: {missing}")

    # Apply offset if provided
    if offset is not None:
        df["x"] += offset[0]
        df["y"] += offset[1]
        df["z"] += offset[2]

    # Build polylines by the backtrack rule
    paths = _split_paths_by_backtrack(df)

    # Prepare plot
    if plot is None:
        plot = k3d.plot()
        if background_color is not None:
            plot.background_color = background_color

    # Draw neuron (single color)
    # for P in paths:
    #     plot += k3d.line(P.astype(np.float32), shader='thick', width=width, color=color)

    #  Concatenate all paths with NaN separators to create disconnected segments
    combined_path = np.vstack(
        [np.vstack([p[0], p, p[-1], [[np.nan, np.nan, np.nan]]]) for p in paths[:-1]] + [paths[-1]]
    )
    plot += k3d.line(
        combined_path.astype(np.float32),
        shader="thick",
        width=width,
        color=color,
        name=f"{n.name}_neurites",
    )

    # ---- Soma: ONLY parent_id == -1, using true radius ----
    soma_rows = df.loc[df["parent_id"] < 0, ["x", "y", "z", "radius"]]
    if len(soma_rows) > 0:
        centers = soma_rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
        radii = soma_rows["radius"].to_numpy(dtype=np.float32)

        # Prefer spheres with real-world radii; fall back to points if spheres are unavailable
        try:
            # Some k3d versions provide k3d.spheres(centers=..., radii=..., colors=...)
            plot += k3d.spheres(
                centers=centers,
                radii=radii,
                colors=[soma_color] * len(radii),
                name=f"{n.name}_soma",
            )
        except AttributeError:
            # Fallback: render as 3D points; point_size approximates radius
            # (Note: depending on k3d version, point_size may behave like pixels;
            #  this is a best-effort fallback.)
            for c, r in zip(centers, radii):
                plot += k3d.points(
                    positions=c.reshape(1, 3),
                    point_size=float(max(r, 1e-3)),
                    color=soma_color,
                    name=f"{n.name}_soma",
                )

    if save_html:
        with open(save_html, "w", encoding="utf-8") as f:
            f.write(plot.get_snapshot())

    return plot, paths


def plot_all_morphology_cells(
    save_html: str = "all_morphology_cells.html",
    width: float = 0.3,
    filter_query: Optional[str] = None,
    add_lc_mesh: bool = True,
    mesh_color: int = 0xAAAAAA,
    mesh_opacity: float = 0.1,
) -> k3d.Plot:
    """Plot all cells with morphology data in a single k3d visualization.
    Uses coordinate system: X (A->P), Y (D->V), Z (L->R).
    Flips left hemisphere cells (z > 5700) to right hemisphere.
    Colors neurons by injection region."""
    from LCNE_patchseq_analysis import MORPHOLOGY_DIRECTORY, REGION_COLOR_MAPPER
    from LCNE_patchseq_analysis.data_util.mesh import add_mesh_to_k3d
    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
    from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3

    def color_to_hex(color_name: str) -> int:
        """Convert matplotlib color name to hex integer for k3d."""
        rgb = mcolors.to_rgb(color_name)
        return int(rgb[0] * 255) << 16 | int(rgb[1] * 255) << 8 | int(rgb[2] * 255)

    # Load metadata
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True, if_with_morphology=True)

    # Apply filter if provided
    if filter_query:
        df_meta = df_meta.query(filter_query).copy()

    # Filter cells with morphology
    df_with_morph = df_meta[df_meta["morphology_soma_surface_area"].notna()].copy()

    # Create k3d plot with axis labels
    plot = k3d.plot()
    plot.axes = ["X (A→P)", "Y (D→V)", "Z (L→R)"]

    # Add LC mesh if requested
    if add_lc_mesh:
        mesh = load_mesh_from_s3()
        plot = add_mesh_to_k3d(plot, mesh, color=mesh_color, opacity=mesh_opacity, both_sides=False)

    # Plot each cell
    for idx, row in df_with_morph.iterrows():
        specimen_id = int(row["cell_specimen_id"])
        swc_path = f"{MORPHOLOGY_DIRECTORY}/swc_upright_mdp/{specimen_id}.swc"

        # Get coordinates from metadata: X (A->P), Y (D->V), Z (L->R)
        x, y, z = row["x"], row["y"], row["z"]

        # Flip left hemisphere cells to right hemisphere
        if z > 5700:
            z = 5700 * 2 - z

        offset = np.array([x, y, z], dtype=np.float32)

        # Get color based on injection region
        injection_region = row.get("injection region", "Non-Retro")
        region_key = (
            injection_region
            if injection_region in REGION_COLOR_MAPPER
            else injection_region.lower()
        )
        color_name = REGION_COLOR_MAPPER.get(region_key, "gray")
        neuron_color = color_to_hex(color_name)

        # Add neuron to plot with offset
        plot, _ = plot_swc_k3d(
            swc_path,
            color=neuron_color,
            width=width,
            soma_color=neuron_color,
            plot=plot,
            offset=offset,
        )

    # Save to HTML
    with open(save_html, "w", encoding="utf-8") as f:
        f.write(plot.get_snapshot())

    print(f"Saved visualization of {len(df_with_morph)} cells to {save_html}")
    return plot


if __name__ == "__main__":
    from LCNE_patchseq_analysis.figures import GLOBAL_FILTER

    plot_all_morphology_cells(filter_query=None, save_html="all_morphology_cells.html")
    plot_all_morphology_cells(
        filter_query=GLOBAL_FILTER, save_html="all_morphology_cells_filtered.html"
    )
