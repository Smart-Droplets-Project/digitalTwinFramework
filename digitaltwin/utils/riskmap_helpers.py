import os
import datetime

import onnx
import onnxruntime as rt

from typing import Union, Sequence

from torch._dynamo.backends import onnxrt

from ascab.env.env import AScabEnv
from ascab.train import RLAgent

import pickle

import statistics, pandas as pd
from pathlib import Path

import geopandas as gpd
import numpy as np

import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import shapely
from shapely.geometry import Polygon, Point, MultiPolygon, mapping
from shapely.ops import voronoi_diagram, unary_union

import json
from shapely.geometry import shape

from sd_data_adapter.models.agri_food import AgriParcel
from gymnasium.wrappers import FlattenObservation, FilterObservation

AI_ASCAB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "ascabmodel", "AI_pesticide_agent"
)

# ------------------------------------------------------------
#  User-tunable parameters
# ------------------------------------------------------------
BUFFER_MAX_M = 30.0            #      0–30 m fade zone
BUFFER_STEP_M = 1.0            #      1-metre slices → smooth gradient
CRS_METRIC   = "EPSG:3857"     #  Web-Mercator = metres       (any metric CRS is fine)
CRS_GEO      = "EPSG:4326"     #  Output prescription in WGS-84 lon/lat
RATE_MAX     = 500.0           #  kg ha-¹ at risk = 0.05
RISK_MAX     = 0.05


def get_polygon_bounds(parcel: AgriParcel):
    return parcel.location['features'][-1]['geometry']['coordinates'][0]

def generate_points_in_polygon(
        polygon: Union[Polygon, list],
        action_dict: dict[datetime.date, float],
        step: float = 0.000001,
        seed: int = 10,
        include_boundary: bool = True,
        max_points: int | None = None,
        points_per_day: int | None = None,
        scale_by_action: bool = False,
        min_points_when_action: int = 4,
) -> dict[datetime.date, list[tuple[float, float]]]:
    """
    For each day in ``action_dict`` create a *repeatable* random subset of points inside ``polygon``.

    Workflow:
      1) Build a reproducible pool of candidate points by jittering one point per grid cell
         (grid spacing = ``step``) and keeping those inside the polygon (boundary allowed if
         ``include_boundary`` is True).
      2) For each date, derive a day-specific RNG seed from (``seed``, date) and pick a subset
         of the candidate pool. The subset size can be:
         - fixed via ``points_per_day``; or
         - scaled by the action value for that date (``scale_by_action=True``) up to ``max_points``.

    Args:
        polygon: Shapely Polygon or a list of (x, y) vertices.
        action_dict: {date -> action_value}. Only keys are required; values can scale count.
        step: Grid cell size in coordinate units.
        seed: Base RNG seed for reproducibility across runs.
        include_boundary: If True, include points on the boundary (uses .covers instead of .contains).
        max_points: Upper bound of *total* candidate points (cap after pool creation). If None, use all.
        points_per_day: If given, choose exactly this many points per day (unless fewer candidates exist).
        scale_by_action: If True and ``points_per_day`` is None, scale daily count by action value relative
                         to the max action in ``action_dict``.
        min_points_when_action: Minimum points to select on days with nonzero action when scaling.

    Returns:
        dict: {date -> [(x, y), ...]}
    """
    # --- Normalize polygon ---
    if not isinstance(polygon, Polygon):
        polygon = Polygon(polygon)

    # --- Build candidate pool once (deterministic wrt `seed`) ---
    minx, miny, maxx, maxy = polygon.bounds
    x_vals = np.arange(minx, maxx, step)
    y_vals = np.arange(miny, maxy, step)

    base_rng = np.random.default_rng(seed)
    candidates: list[tuple[float, float]] = []

    for x0 in x_vals:
        for y0 in y_vals:
            # uniform point within the current cell [x0, x0+step) × [y0, y0+step)
            x = x0 + base_rng.uniform(0.0, step)
            y = y0 + base_rng.uniform(0.0, step)
            pt = Point(x, y)
            inside = polygon.covers(pt) if include_boundary else polygon.contains(pt)
            if inside:
                candidates.append((x, y))

    # Cap the candidate pool size if requested (stable, but randomly subsampled deterministically)
    if max_points is not None and len(candidates) > max_points:
        idx = base_rng.permutation(len(candidates))[:max_points]
        candidates = [candidates[i] for i in idx]

    # Edge case: empty polygon or too coarse step
    if not candidates:
        return {d: [] for d in action_dict.keys()}

    # --- Decide per-day sample sizes ---
    # If fixed points_per_day is given, that overrides scaling by action.
    fixed_k = None
    if points_per_day is not None:
        fixed_k = max(0, min(points_per_day, len(candidates)))

    # Prepare scaling if needed
    act_values = list(action_dict.values())
    max_act = float(max(act_values)) if act_values else 0.0

    def _derive_day_seed(base: int, day: datetime.date) -> int:
        """Mix base seed with date; wrap to 64 bits to keep NumPy happy."""
        MASK64 = (1 << 64) - 1
        mixed = (int(base) ^ ((day.toordinal() * 0x9E3779B97F4A7C15) & MASK64)) & MASK64
        return mixed  # plain Python int in [0, 2**64-1]

    per_day: dict[datetime.date, list[tuple[float, float]]] = {}

    for day, act in action_dict.items():
        # Day-specific RNG -> different but repeatable subset for each date
        day_seed = _derive_day_seed(seed, day)
        day_rng = np.random.default_rng(day_seed)

        if fixed_k is not None:
            k = fixed_k
        else:
            # If scaling by action and we have a max, map action to [0,1] and scale up to len(candidates)
            if scale_by_action and max_act > 0:
                frac = float(max(0.0, act)) / max_act
                est = int(np.ceil(frac * len(candidates)))
                k = max(min_points_when_action if act > 0 else 0, est)
            else:
                # Default: use all candidates
                k = len(candidates)
        k = max(0, min(k, len(candidates)))

        if k == len(candidates):
            # Full set but shuffled deterministically per day for spatial variation
            order = day_rng.permutation(len(candidates))
            chosen = [candidates[i] for i in order]
        else:
            idx = day_rng.permutation(len(candidates))[:k]
            chosen = [candidates[i] for i in idx]
        per_day[day] = chosen

    return per_day


def get_risks(points_list: list, end_date: datetime.date = None) -> dict:
    year = datetime.date.today().year
    end_date = end_date or datetime.date(year-1, 10, 1)
    year = end_date.year

    start_date = datetime.date(year, 1, 1)
    dates = (start_date.isoformat(), end_date.isoformat())

    risk_dict = {}
    print(points_list)
    for point in points_list:
        ascab = AScabEnv(
            location=point,
            dates=dates,
            biofix_date="March 10",
            budbreak_date="March 10",
        )
        ascab.reset()
        terminated = False
        while not terminated:
            _, _, terminated, _, infos = ascab.step(0)

        print(f"Got point {point}!")

        risk_dict[point] = {
            infos['Date'][i]: infos['Risk'][i]
            for i in range(len(infos['Date']))
            if infos["InfectionWindow"][i] == 1
        }

    return risk_dict


def get_daily_actions(coor: tuple, start_date: datetime.date = None, end_date: datetime.date = None) -> dict:
    year = datetime.date.today().year
    end_date = end_date or datetime.date(year-1, 10, 1)
    year = end_date.year

    if start_date is None:
        start_date = datetime.date(year, 1, 1)
    dates = (start_date.isoformat(), end_date.isoformat())

    risk_dict = {}

    onnx_model_file = os.path.join(AI_ASCAB_DIR, "ascab_agent.onnx")
    agent = onnx.load(onnx_model_file)
    onnx.checker.check_model(agent)

    ort_session = rt.InferenceSession(onnx_model_file)

    ascab = AScabEnv(
        location=coor,
        dates=dates,
        biofix_date="March 10",
        budbreak_date="March 10",
    )
    ascab = FilterObservation(
        ascab,
        filter_keys=[
            'ActionHistory', 'AppliedPesticide', 'Beta', 'Discharge', 'Forecast_day1_HasRain',
            'Forecast_day1_HumidDuration', 'Forecast_day1_LeafWetness', 'Forecast_day1_Precipitation',
            'Forecast_day1_Temperature', 'Forecast_day2_HasRain', 'Forecast_day2_HumidDuration',
            'Forecast_day2_LeafWetness', 'Forecast_day2_Precipitation', 'Forecast_day2_Temperature',
            'HasRain', 'HumidDuration', 'InfectionWindow', 'LAI', 'LeafWetness', 'Precipitation',
            'RemainingSprays', 'Temperature'
        ]
    )
    ascab = FlattenObservation(ascab)
    infos = None
    obs, _ = ascab.reset()
    terminated = False

    h_pi = c_pi = h_vf = c_vf = np.zeros((1, 1, 256), dtype=np.float32)
    episode_starts = np.zeros(1, dtype=np.bool_)

    while not terminated:
        action, _, _, h_pi, c_pi, h_vf, c_vf = ort_session.run(
            None,
            input_feed={
                "input": np.array(obs, dtype=np.float32)[None, :],
                "h0_pi": np.array(h_pi),
                "c0_pi": np.array(c_pi),
                "h0_vf": np.array(h_vf),
                "c0_vf": np.array(c_vf),
                "episode_starts": episode_starts,
            }
        )
        obs, _, terminated, _, infos = ascab.step(action.item())
        # episode_starts = np.array(terminated, dtype=np.bool_)

    for i in range(len(infos['Date'])):
        risk_dict[infos['Date'][i]] = infos['Action'][i]

    return risk_dict

def split_polygon_by_points(
    big_poly: Polygon,
    seed_pts: list[Point],
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Make one Voronoi cell per point and clip to `big_poly`.

    Parameters
    ----------
    big_poly : shapely Polygon (lon/lat)
    seed_pts : list of shapely Points inside big_poly
    crs      : CRS string for the output GeoDataFrame (defaults to WGS-84)

    Returns
    -------
    GeoDataFrame with columns:
        ├─ 'seed_id'   – index of the input point
        └─ 'geometry'  – the clipped Voronoi cell
    """
    if not all(big_poly.contains(pt) for pt in seed_pts):
        raise ValueError("All seed points must lie inside big_poly.")

    # 1 ▸ Voronoi diagram in lon/lat degrees
    vd = voronoi_diagram(unary_union(seed_pts), envelope=None)

    # 2 ▸ find & clip the cell for each point
    records = []
    for idx, pt in enumerate(seed_pts):
        cell = next(poly for poly in vd.geoms if poly.contains(pt))
        clipped = cell.intersection(big_poly)
        records.append(dict(seed_id=idx, geometry=clipped))

    return gpd.GeoDataFrame(records, crs=crs)


def _build_seed_gdf(
        seed_points: Union[
            dict[Union[str, int], list[tuple[float, float]]],
            tuple[Sequence[tuple[float, float]], Sequence[int]],
        ],
) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame with columns [parcel_id, seed_id, geometry]  (metric CRS).

    Accepts either:
      • a dict  {parcel_id: [(lon, lat), …], …}
      • or a pair (list_of_points, list_of_parcel_ids) of equal length.

    seed_id is just an incremental integer.
    """
    if isinstance(seed_points, dict):
        rows = []
        for pid, pts in seed_points.items():
            for pt in pts:
                rows.append((pid, pt))
    else:  # two parallel lists
        pts, pids = seed_points
        if len(pts) != len(pids):
            raise ValueError("seed_points and seed_parcel_ids must match length")
        rows = list(zip(pids, pts))

    gdf = gpd.GeoDataFrame(
        {
            "parcel_id": [pid for pid, _ in rows],
            "geometry": [Point(pt) for _, pt in rows],
        },
        crs=CRS_GEO,
    ).to_crs(CRS_METRIC)

    gdf["seed_id"] = np.arange(len(gdf))  # unique ID per point
    return gdf


def prescription_from_points(
        risk_dict_path: Union[str, Path, dict],
        seed_points: Union[
            dict[Union[int, str], list[tuple[float, float]]],
            tuple[Sequence[tuple[float, float]], Sequence[int]],
        ],
        output_dir: Union[str, Path] = "../risk_maps",
):
    # ---------- 1 · load risk_dict -----------------------------------------
    if isinstance(risk_dict_path, (str, Path)):
        with open(risk_dict_path, "rb") as f:
            risk_dict = pickle.load(f)
    else:
        risk_dict = risk_dict_path

    # ---------- 2 · seed-point GDF -----------------------------------------
    gdf_pts = _build_seed_gdf(seed_points)

    # ---------- 3 · Voronoi cells clipped to convex hull -------------------
    vd = voronoi_diagram(unary_union(gdf_pts.geometry), envelope=None)
    hull = unary_union(gdf_pts.geometry).convex_hull.buffer(5 * BUFFER_MAX_M)

    cells = []
    for idx, pt in gdf_pts.iterrows():
        seed_geom = pt.geometry
        cell = next(poly for poly in vd.geoms if poly.contains(seed_geom))
        cells.append(dict(seed_id=pt.seed_id,
                          geometry=cell.intersection(hull)))
    gdf_cells = gpd.GeoDataFrame(cells, crs=CRS_METRIC)

    # ---------- 4 · build buffer rings per seed point ----------------------
    rings = []
    for _, row in gdf_pts.iterrows():
        sid, seed_geom = row.seed_id, row.geometry
        cell_geom = gdf_cells.loc[gdf_cells.seed_id == sid, "geometry"].values[0]

        for d_in in np.arange(0, BUFFER_MAX_M, BUFFER_STEP_M):
            d_out = d_in + BUFFER_STEP_M
            ring_raw = seed_geom.buffer(d_out).difference(seed_geom.buffer(d_in))
            ring = ring_raw.intersection(cell_geom)
            if ring.is_empty:
                continue
            centre_d = d_in + 0.5 * BUFFER_STEP_M
            factor = max(0.0, 1.0 - centre_d / BUFFER_MAX_M)
            rings.append(dict(
                seed_id=sid,
                dist_m=centre_d,
                factor=factor,
                geometry=ring,
            ))

        outer = cell_geom.difference(seed_geom.buffer(BUFFER_MAX_M))
        if not outer.is_empty:
            rings.append(dict(
                seed_id=sid,
                dist_m=BUFFER_MAX_M + 1,
                factor=0.0,
                geometry=outer,
            ))

    gdf_rings = gpd.GeoDataFrame(rings, crs=CRS_METRIC)

    # ---------- 5 · pull the risk series for EACH SEED POINT ---------------
    recs = []
    for _, pt in gdf_pts.iterrows():
        pid = pt.parcel_id
        # lonlat = seed_points[pid][i]  # read the stored WGS-84 coords

        for lonlat in seed_points[pid]:
            try:
                date_map = risk_dict[pid][lonlat]
            except KeyError:
                raise KeyError(
                    f"risk_dict missing entry for parcel {pid} at {lonlat}"
                )

            for dt, r in date_map.items():
                base_rate = min(RATE_MAX, (r / RISK_MAX) * RATE_MAX)
                recs.append(
                    dict(seed_id=pt.seed_id,
                         parcel_id=pid,
                         date=dt,
                         base_rate=base_rate)
                )

    df_rates = pd.DataFrame(recs)

    # ---------- 6 · join geometry + rates, scale by factor -----------------
    gdf_join = gdf_rings.merge(df_rates, on="seed_id", how="left")
    gdf_join["Target_Rat"] = (gdf_join["base_rate"] * gdf_join["factor"]).round(2)

    # ---------- 7 · write out one file per date ----------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf_join = gdf_join.to_crs(CRS_GEO)

    gdf_join.to_file(os.path.join(output_dir, f"parcel.geojson"), driver="GeoJSON")

    print(f"✓ wrote {gdf_join['date'].nunique()} files → {out_dir.resolve()}")

import numpy as np

def gaussian_field_from_points(X, Y, pts, vals, sigma, scale=False):
    """
    X, Y : 2D meshgrid arrays (same shape)
    pts  : array-like of shape (N, 2) with point coords [[x1,y1], [x2,y2], ...]
    vals : array-like of shape (N,) with weights/heights at those points
    sigma: float or array-like of shape (N,) – Gaussian std dev (same units as X/Y)
    """
    pts = np.asarray(pts)
    vals = np.asarray(vals)
    sigma = np.asarray(sigma)
    if sigma.ndim == 0:
        sigma = np.full(len(pts), float(sigma))

    # Broadcast: for each point i, compute Gaussian on the whole grid
    # (N, H, W) after broadcasting
    dx = X[None, :, :] - pts[:, 0, None, None]
    dy = Y[None, :, :] - pts[:, 1, None, None]
    s2 = (2.0 * sigma[:, None, None] ** 2)
    kernels = np.exp(-(dx**2 + dy**2) / s2)

    # Weight by vals and sum across points
    Z = np.sum(vals[:, None, None] * kernels, axis=0)
    if scale and np.any(Z):
        Z = (Z - Z.min()) / (Z.max() - Z.min())
    Z[Z < 2.0] = 0.0
    return Z


# --- helper: convert contourf on lon/lat grid to GeoJSON polygons, clipped to clip_poly ---
def _contours_to_geojson_lonlat(X, Y, Z, levels, clip_poly: Polygon, parcel_id=None, date_str=None, scale_factor=500.0):
    """
    Turn a contourf result on lon/lat grids into GeoJSON Polygon features, clipped to clip_poly.

    Returns: {"type":"FeatureCollection","features":[...]}
    Each feature has properties:
      - parcel_id (if provided)
      - date (if provided)
      - Risk_min, Risk_max
      - Target_Rat = scale_factor * 0.5*(Risk_min + Risk_max)
    """
    features = []

    # Normalize inputs
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    Zm = np.array(Z, dtype=float)

    # Mask NaNs (outside parcel already NaN in caller)
    mask_valid = ~np.isnan(Zm)
    if not np.any(mask_valid):
        return {"type": "FeatureCollection", "features": features}

    # Derive approximate grid spacing from the meshgrid
    # X, Y are (H, W) with rows ~ constant Y and cols ~ constant X
    def _safe_step(arr, axis):
        try:
            d = np.nanmedian(np.diff(arr, axis=axis))
            if not np.isfinite(d) or d == 0:
                return None
            return float(abs(d))
        except Exception:
            return None

    dx = _safe_step(X, axis=1)
    dy = _safe_step(Y, axis=0)
    # Fallbacks: try the other axis if one is degenerate
    if dx is None:
        dx = _safe_step(X, axis=0)
    if dy is None:
        dy = _safe_step(Y, axis=1)
    # Absolute fallback to small cell if still None
    if dx is None or dx == 0:
        dx = 1e-6
    if dy is None or dy == 0:
        dy = 1e-6

    half_dx, half_dy = 0.5 * dx, 0.5 * dy

    H, W = Zm.shape
    # Pre-create center coords (same shape as Z)
    Xc, Yc = X, Y

    # Iterate contour bands and polygonize by merging cell-rectangles that fall into each band
    for li in range(len(levels) - 1):
        zmin = float(levels[li])
        zmax = float(levels[li + 1])
        target = float(scale_factor * 0.5 * (zmin + zmax))

        cell_polys = []

        # Select cells whose Z lies inside this band
        # Include the top edge on the last band to capture the max.
        if li == len(levels) - 2:
            in_band = (Zm >= zmin) & (Zm <= zmax) & mask_valid
        else:
            in_band = (Zm >= zmin) & (Zm < zmax) & mask_valid

        # Build rectangles for each cell center and clip to input polygon
        idxs = np.argwhere(in_band)
        for iy, ix in idxs:
            cx = float(Xc[iy, ix])
            cy = float(Yc[iy, ix])
            rect = Polygon([
                (cx - half_dx, cy - half_dy),
                (cx + half_dx, cy - half_dy),
                (cx + half_dx, cy + half_dy),
                (cx - half_dx, cy + half_dy),
            ])
            inter = rect.intersection(clip_poly)
            if not inter.is_empty:
                cell_polys.append(inter)

        if not cell_polys:
            continue

        merged = unary_union(cell_polys)
        if merged.is_empty:
            continue

        def _emit_geom(geom):
            if isinstance(geom, Polygon):
                rings = [np.asarray(geom.exterior.coords)]
                rings += [np.asarray(r.coords) for r in geom.interiors]
                exterior_ll = rings[0].tolist()
                holes_ll = [r.tolist() for r in rings[1:]]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [exterior_ll] + holes_ll},
                    "properties": {
                        **({"parcel_id": str(parcel_id)} if parcel_id is not None else {}),
                        **({"date": str(date_str)} if date_str is not None else {}),
                        "Risk_min": zmin,
                        "Risk_max": zmax,
                        "Target_Rat": target,
                    },
                })
            elif isinstance(geom, MultiPolygon):
                for g in geom.geoms:
                    _emit_geom(g)

        _emit_geom(merged)

        background = clip_poly.difference(unary_union([shape(f["geometry"]) for f in features]))
        if not background.is_empty:
            features.append({
                "type": "Feature",
                "geometry": mapping(background),
                "properties": {
                    "parcel_id": parcel_id,
                    **({"date": str(date_str)} if date_str is not None else {}),
                    "Risk_min": 0.0,
                    "Risk_max": levels[0],
                    "Target_Rat": 0.0
                }
            })

    return {"type": "FeatureCollection", "features": features}

def create_prescription_maps(
        risk_dict,
        polygon_bounds,
        points_dict,
        output_dir: Union[str, Path] = "../risk_maps",
):
    for parcel_id, polygon_bounds in polygon_bounds.items():
        poly = Polygon(polygon_bounds)

        # Get bounding box limits
        minx, miny, maxx, maxy = poly.bounds

        # Define resolution (smaller step = denser grid)
        step = 0.000005  # about 10 m spacing (roughly)

        # Define the spread of gaussian
        sigma=0.0003,  # don't change this value

        # Create meshgrid
        x = np.arange(minx, maxx, step)
        y = np.arange(miny, maxy, step)
        xx, yy = np.meshgrid(x, y)

        max_z_value = 0.0

        val_per_day = {}
        for day, action in risk_dict[parcel_id].items():
            zz = gaussian_field_from_points(
                xx,
                yy,
                points_dict[parcel_id][day],
                [action] * len(points_dict[parcel_id][day]),
                sigma
            )
            if not np.any(zz):
                continue
            val_per_day[day] = zz

            max_z_value = max(max_z_value, zz.max())

            # --- build a polygon mask (True inside polygon) ---
            # vectorized containment check over the grid
            flat_pts = np.column_stack([xx.ravel(), yy.ravel()])
            inside = np.array([poly.covers(Point(p)) for p in flat_pts]).reshape(xx.shape)

            # apply mask to Z so contour ignores outside cells
            Zplot = np.where(inside, zz, np.nan)

            # --- build the filled-contour polygons and write GeoJSON ---
            levels = np.linspace(0.0, max_z_value, 16)  # adjust if you want a different scale/extent
            X, Y = xx, yy
            fc = _contours_to_geojson_lonlat(
                X,
                Y,
                Zplot,
                levels,
                poly,
                parcel_id=parcel_id,
                date_str=day,
                scale_factor=50.0
            )

            os.makedirs(output_dir, exist_ok=True)
            safe_day = str(day).replace(":", "-")
            out_file = os.path.join(output_dir, f"taskmap_ascab_{safe_day}.geojson")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(fc, f, ensure_ascii=False, allow_nan=False, indent=2)
                f.write("\n")

            # Read data
            gdf = gpd.read_file(out_file)

            # Basic info
            print(f"CRS: {gdf.crs}")
            print("Columns:", list(gdf.columns))
            print(f"Feature count: {len(gdf)}")

            # Separate polygons and points
            geom_type = gdf.geom_type.astype(str)
            poly_mask = geom_type.isin(["Polygon", "MultiPolygon"])
            point_mask = geom_type.isin(["Point"]) | (gdf.get("role", "") == "seed_point")

            gdf_poly = gdf[poly_mask].copy()
            gdf_pts = gdf[point_mask].copy()

            # --- compute background geometry: parcel minus union of contour polygons ---
            contour_union = unary_union(gdf_poly.geometry) if not gdf_poly.empty else shapely.geometry.GeometryCollection()
            bg_geom = poly.difference(contour_union)


            # Sort polygons by Target_Rat ascending so low-dose draw first
            if "Target_Rat" in gdf_poly.columns:
                gdf_poly = gdf_poly.sort_values("Target_Rat")

            # Figure and common colormap
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f"Task map {safe_day}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal", adjustable="box")

            cmap = plt.get_cmap("summer")

            # Compute normalization from quantiles across BOTH polygons and points
            all_vals = []
            if "Target_Rat" in gdf_poly.columns:
                all_vals.extend(gdf_poly["Target_Rat"].dropna().tolist())
            if "Target_Rat" in gdf_pts.columns:
                all_vals.extend(gdf_pts["Target_Rat"].dropna().tolist())
            if all_vals:
                vmin = np.quantile(all_vals, 0.0)
                vmax = np.quantile(all_vals, 1.0)
                if vmin == vmax:
                    vmin, vmax = min(all_vals), max(all_vals)
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = Normalize(vmin=0, vmax=1)

            # --- draw the rest of the parcel so the whole polygon is visible ---
            if bg_geom and not bg_geom.is_empty:
                gdf_bg = gpd.GeoDataFrame(
                    {"parcel_id": [parcel_id], "Target_Rat": [0.0], "geometry": [bg_geom]},
                    crs=gdf.crs if gdf.crs else "EPSG:4326",
                )
                # Color for "no-dose" area = lowest value on the scale
                bg_color = "white"
                gdf_bg.plot(ax=ax, color=[bg_color], edgecolor="black", linewidth=0.6)

            # Optional: draw parcel boundary on top for clarity
            try:
                gpd.GeoSeries([poly], crs=gdf.crs if gdf.crs else "EPSG:4326").boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=9)
            except Exception:
                pass

            # Plot polygons (no legend yet; we'll add a unified colorbar)
            if not gdf_poly.empty:
                # Geopandas doesn't accept a Matplotlib norm directly; map colors ourselves
                if "Target_Rat" in gdf_poly.columns:
                    colors = [cmap(norm(v)) for v in gdf_poly["Target_Rat"].fillna(0).tolist()]
                    gdf_poly.plot(ax=ax, color=colors, edgecolor="black", linewidth=0.6)
                else:
                    gdf_poly.plot(ax=ax, edgecolor="black", linewidth=0.6)

            # Overlay seed points with same colormap
            if not gdf_pts.empty:
                try:
                    xs = gdf_pts.geometry.x.values
                    ys = gdf_pts.geometry.y.values
                except Exception:
                    # Ensure points
                    pts = gdf_pts.geometry.representative_point()
                    xs = pts.x.values
                    ys = pts.y.values
                sizes = np.full_like(xs, 16.0, dtype=float)
                if "Target_Rat" in gdf_pts.columns:
                    pt_vals = gdf_pts["Target_Rat"].fillna(0).tolist()
                    pt_colors = [cmap(norm(v)) for v in pt_vals]
                else:
                    pt_colors = "k"
                ax.scatter(xs, ys, s=sizes, c=pt_colors, edgecolors="white", linewidths=0.6, zorder=10)

            # Unified colorbar
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Target_Rat")

            plt.tight_layout()
            out_fig = os.path.join(output_dir, f"taskmap_ascab_{safe_day}.png")
            plt.savefig(out_fig, dpi=200)
            print(f"Saved PNG to {out_fig}")
