import navis as nv
import numpy as np
import k3d
from typing import Tuple, List, Optional

def _split_paths_by_backtrack(df) -> List[np.ndarray]:
    """Split into polylines: if current row's parent_id != previous row's node_id,
    we 'backtrack' → start a new polyline. Roots (parent_id < 0) also start a new polyline.
    Returns list of (Ni,3) float32 arrays."""
    df = df[['node_id','x','y','z','parent_id']].copy()
    df['node_id']   = df['node_id'].astype(int)
    df['parent_id'] = df['parent_id'].astype(int)
    df = df.sort_values('node_id').reset_index(drop=True)

    coords = df[['x','y','z']].to_numpy(dtype=np.float32)
    id2row = {int(nid): i for i, nid in enumerate(df['node_id'].to_numpy())}

    paths: List[np.ndarray] = []
    current: List[np.ndarray] = []
    last_node_id: Optional[int] = None

    for _, r in df.iterrows():
        nid = int(r['node_id']); pid = int(r['parent_id'])
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


def plot_swc_k3d(
    swc_path: str,
    color: int = 0x00aaff,
    width: float = 0.3,
    soma_color: int = 0xff0000,
    background_color: Optional[int] = None,
    save_html: Optional[str] = None,
) -> Tuple[k3d.Plot, List[np.ndarray]]:
    """Render SWC in k3d with a single color; soma is defined ONLY as rows with parent_id==-1,
    drawn using the real SWC radius."""
    # Load (first neuron if list)
    n = nv.read_swc(swc_path)
    if isinstance(n, nv.core.NeuronList):
        n = n[0]
    df = n.nodes.copy()

    # Required columns
    needed = {'node_id','x','y','z','parent_id','radius'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"SWC missing columns: {missing}")

    # Build polylines by the backtrack rule
    paths = _split_paths_by_backtrack(df)

    # Prepare plot
    plot = k3d.plot()
    if background_color is not None:
        plot.background_color = background_color

    # Draw neuron (single color)
    # for P in paths:
    #     plot += k3d.line(P.astype(np.float32), shader='thick', width=width, color=color)

    #  Concatenate all paths with NaN separators to create disconnected segments
    combined_path = np.vstack([np.vstack([p[0], p, p[-1], [[np.nan, np.nan, np.nan]]]) 
                            for p in paths[:-1]] + [paths[-1]])
    plot += k3d.line(combined_path.astype(np.float32), shader='thick', width=width, color=color, name=f"{n.name}_neurites")

    # ---- Soma: ONLY parent_id == -1, using true radius ----
    soma_rows = df.loc[df['parent_id'] < 0, ['x','y','z','radius']]
    if len(soma_rows) > 0:
        centers = soma_rows[['x','y','z']].to_numpy(dtype=np.float32)
        radii   = soma_rows['radius'].to_numpy(dtype=np.float32)

        # Prefer spheres with real-world radii; fall back to points if spheres are unavailable
        try:
            # Some k3d versions provide k3d.spheres(centers=..., radii=..., colors=...)
            plot += k3d.spheres(centers=centers, radii=radii, colors=[soma_color]*len(radii))
        except AttributeError:
            # Fallback: render as 3D points; point_size approximates radius
            # (Note: depending on k3d version, point_size may behave like pixels; this is a best-effort fallback.)
            for c, r in zip(centers, radii):
                plot += k3d.points(positions=c.reshape(1,3),
                                   point_size=float(max(r, 1e-3)),
                                   color=soma_color,
                                   name=f"{n.name}_soma")

    if save_html:
        with open(save_html, "w", encoding="utf-8") as f:
            f.write(plot.get_snapshot())

    return plot, paths