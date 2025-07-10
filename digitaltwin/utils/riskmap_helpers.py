import os
import datetime

from typing import Union, Sequence

from ascab.env.env import AScabEnv

import pickle

import statistics, pandas as pd
from pathlib import Path

import geopandas as gpd
import numpy as np

import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import voronoi_diagram, unary_union

import json
import folium
import branca.colormap as cm
from shapely.geometry import shape

from models.agri_food import AgriParcel

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

def generate_points_in_polygon(polygon: Union[Polygon, list], step: float = 0.001):
    """
    Generate a list of (lon, lat) tuples spaced at `step` degrees that lie inside the given polygon.
    """
    if not isinstance(polygon, Polygon):
        polygon = Polygon(polygon)
    minx, miny, maxx, maxy = polygon.bounds
    x_vals = np.arange(minx, maxx + step, step)
    y_vals = np.arange(miny, maxy + step, step)

    points = []
    for x in x_vals:
        for y in y_vals:
            pt = Point(x, y)
            if polygon.contains(pt):
                points.append((x, y))
    return points

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


def _as_geodataframe(
    geom_input: Union[str, list, Path, gpd.GeoDataFrame, Polygon, Sequence[Polygon]],
    risk_dict: dict,
) -> gpd.GeoDataFrame:
    """
    Turn *anything* the user passes in into a GeoDataFrame that
    has a 'parcel_id' column matching the keys of risk_dict.
    """
    if isinstance(geom_input, dict):
        pols = {}
        for k, val in geom_input.items():
            pols[k] = Polygon(val)
        gdf = gpd.GeoDataFrame(
            {"parcel_id": list(risk_dict.keys()), "geometry": list(pols.values())}, crs="EPSG:4326"
        )

    else:
        raise TypeError(
            "parcel_geojson_path must be a file path, GeoDataFrame, "
            "Polygon or list/tuple of Polygons"
        )

    # final sanity-check
    if "parcel_id" not in gdf.columns:
        raise ValueError("Geometry layer must contain a 'parcel_id' column")

    return gdf


def create_prescription_geojson(
    risk_dict: Union[str, dict],
    parcel_geojson_path: Union[AgriParcel, list, dict, Polygon],
    output_path: Union[Path, str] = "../risk_maps",
    agg: str = "mean",          # "mean" or "max"
    risk_zero: float = 0.0,     # risk = 0   ⇒ 0 kg ha-1
    risk_full: float = 0.05,    # risk = 0.05 ⇒ 500 kg ha-1
    full_rate: float = 500.0,
):
    """
    Build ONE GeoJSON with one feature per (parcel, date).

    The parcel GeoJSON must contain a column called `parcel_id`
    that matches the numeric keys in risk_dict.
    """
    os.makedirs(output_path, exist_ok=True)

    if isinstance(risk_dict, str):
        with open(risk_dict, "rb") as f:
            risk_dict = pickle.load(f)

    parcels = _as_geodataframe(parcel_geojson_path, risk_dict)

    # ── build a look-up: parcel_id → [Point, …] ────────────────────────────────
    parcel_points = {
        pid: [Point(xy) for xy in coord_map.keys()]
        for pid, coord_map in risk_dict.items()
    }

    # ── make Voronoi cells *within each* parcel ────────────────────────────────
    cells = {}  # (pid, (lon, lat)) → Polygon
    for pid, points in parcel_points.items():
        parent_poly = parcels.loc[parcels["parcel_id"] == pid, "geometry"].iloc[0]
        if len(points) == 1:
            # only one seed ⇒ whole parcel
            cells[(pid, points[0].coords[0])] = parent_poly
            continue

        # Voronoi of this parcel’s points (lon/lat degrees is OK for sub-field size)
        vd = voronoi_diagram(unary_union(points), envelope=None)
        for pt in points:
            cell = next(poly for poly in vd.geoms if poly.contains(pt))
            clipped = cell.intersection(parent_poly)
            cells[(pid, pt.coords[0])] = clipped

    # ---------- 2 · Collapse risk_dict ----------
    # recs = []
    # for pid, coord_map in risk_dict.items():
    #     per_date = {}
    #     for _coord, date_r_map in coord_map.items():
    #         for dt, r in date_r_map.items():
    #             per_date.setdefault(dt, []).append(r)
    #
    #     for dt, lst in per_date.items():
    #         r = statistics.mean(lst) if agg == "mean" else max(lst)
    #         # Linear  ramp:  r=0  → 0   |  r=risk_full → full_rate
    #         rate = ((max(risk_zero, min(r, risk_full)) - risk_zero) /
    #                 (risk_full - risk_zero) * full_rate)
    #         recs.append(
    #             dict(
    #                 parcel_id=pid,
    #                 date=dt.isoformat(),
    #                 Target_Rat=round(rate, 2),
    #             )
    #         )

    recs = []
    for pid, coord_map in risk_dict.items():
        for xy, date_risk in coord_map.items():
            poly = cells[(pid, xy)]
            for dt, r in date_risk.items():
                rate = min(full_rate, (r / risk_full) * full_rate)
                recs.append(
                    dict(
                        parcel_id=pid,
                        lon=xy[0],
                        lat=xy[1],
                        date=dt.isoformat(),
                        Target_Rat=round(rate, 2),
                        geometry=poly,
                    )
                )

    # df = pd.DataFrame(recs)

    gdf = gpd.GeoDataFrame(recs, crs="EPSG:4326")

    gdf.sort_values("date", inplace=True)
    # # ---------- 3 · Join with parcel geometries ----------
    # gdf = parcels.merge(df, on="parcel_id", how="inner")

    # ---------- 4 · Write out ----------
    # for dt, sub in gdf.groupby("date"):
    #     out_file = os.path.join(output_path, f"{dt}_prescription.geojson")
    #     sub.to_file(out_file, driver="GeoJSON")
    gdf.to_file(os.path.join(output_path, f"parcel.geojson"), driver="GeoJSON")
    print(f"Written: {output_path}  ({len(gdf)} features)")


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


def visualize_riskmap(
        risk_dict_path: Union[str, Path, dict] = None,
):
    if risk_dict_path is not None:
        path = risk_dict_path
    else:
        # --- Load the GeoJSON risk‑map layer ---
        path = "../risk_maps/parcel.geojson"
    with open(path, "r") as f:
        gj = json.load(f)

    dates = sorted({feat["properties"]["date"] for feat in gj["features"]})
    first_date = dates[20]

    patches = []
    values = []
    for feat in gj["features"]:
        if feat["properties"]["date"] != first_date:
            continue
        geom = shape(feat["geometry"])
        rate = feat["properties"].get("Target_Rat", 0)
        if isinstance(geom, Polygon):
            geoms = [geom]
        elif isinstance(geom, MultiPolygon):
            geoms = list(geom.geoms)
        else:
            continue
        for g in geoms:
            coords = np.asarray(g.exterior.coords)
            patches.append(MplPolygon(coords, closed=True))
            values.append(rate)

    fig, ax = plt.subplots()
    pc = PatchCollection(patches, array=np.array(values), linewidths=0.5, edgecolor='black')
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f"Risk map – {first_date}")
    fig.savefig(f"risk_map_{first_date}.png")

    # # --- What dates exist? ---
    # dates = sorted({feat["properties"]["date"] for feat in gj["features"]})
    #
    # # Pick the first date as a demo (change this to any available date)
    # selected_date = dates[0]
    #
    # # Filter features to that date
    # sub_features = [feat for feat in gj["features"] if feat["properties"]["date"] == selected_date]
    # sub_gj = {"type": "FeatureCollection", "features": sub_features}
    #
    # # --- Map centre: centroid of all selected polygons ---
    # centroids = [shape(feat["geometry"]).centroid for feat in sub_features]
    # cent_lat = sum(pt.y for pt in centroids) / len(centroids)
    # cent_lon = sum(pt.x for pt in centroids) / len(centroids)
    #
    # # --- Colour scale for Target_Rat (0 kg → green, 500 kg → red) ---
    # colormap = cm.LinearColormap(
    #     colors=["green", "yellow", "red"],
    #     vmin=0,
    #     vmax=500,
    #     caption="Target_Rat (kg/ha⁻¹)"
    # )
    #
    # def style_function(feature):
    #     rate = feature["properties"]["Target_Rat"]
    #     return {
    #         "fillOpacity": 0.6,
    #         "weight": 0.5,
    #         "color": "black",
    #         "fillColor": colormap(rate),
    #     }
    #
    # # --- Build the interactive map ---
    # m = folium.Map(location=[cent_lat, cent_lon], zoom_start=17, tiles="OpenStreetMap")
    #
    # folium.GeoJson(
    #     sub_gj,
    #     name=f"Prescription {selected_date}",
    #     style_function=style_function,
    #     tooltip=folium.GeoJsonTooltip(fields=["Target_Rat"], aliases=["kg/ha⁻¹"]),
    # ).add_to(m)
    #
    # colormap.add_to(m)