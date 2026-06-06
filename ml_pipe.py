##########################################################
# ml_pipe.py # Locates all features using RF + XGBoost   #
# ------------------------------------------------------ #
# Copyright (c) Payton Jachec 2026. | Disclaimer:        #
# For research purposes only, not for clinical use.      #
# harrisonjachec@usf.edu                                 #
##########################################################

#  CONFIG — edit everything in this section


# ── Input spectral index GeoTIFFs (output of compute_indices.py)
INDEX_PATHS = {
    "NDVI":  "/PATH/TO/NDVI.tif",
    "NDWI":  "/PATH/TO/NDWI.tif",
    "MNDVI": "/PATH/TO/MNDVI.tif",
    "MNDWI": "/PATH/TO/MNDWI.tif",
    "SAVI":  "/PATH/TO/SAVI.tif",
    "SABI":  "/PATH/TO/SABI.tif",
}

DEM_PATH = "/PATH/TO/DEM.tif"

# ── Training data GeoJSONs
POSITIVE_GEOJSON = "/PATH/TO/known_positives.geojson"
NEGATIVE_GEOJSON = "/PATH/TO/known_negatives.geojson"

# ── Output directory
OUT_DIR = "/PATH/TO/OUTPUT/DIRECTORY"

# ── Positive site buffer radii (metres)
# Training samples are drawn from each of these buffer rings around known sites.
# Multiple scales capture both mine cores and disturbed margins.
BUFFER_RADII_M = [50, 100, 200, 400]

# ── Detection threshold
# Confidence score [0–1] above which a pixel is classified as a mine.
# Lower  → more detections, more false positives  (cast wide net)
# Higher → fewer detections, higher confidence only
CONFIDENCE_THRESHOLD = 0.45

# ── Feature engineering toggles
USE_RAW_INDICES    = True   # Raw NDVI, NDWI, MNDVI, MNDWI, SAVI, SABI values
USE_TERRAIN        = True   # DEM-derived: elevation, slope, TPI, TWI
USE_INDEX_RATIOS   = True   # SABI/NDVI, SAVI/NDVI, MNDVI/NDVI, MNDWI/NDWI etc.
USE_TEXTURE        = True   # Local variance in sliding window (mine surfaces are rough)
USE_CONTEXT_ZSCORE = True   # Per-pixel z-score vs local neighborhood (anomaly detection)

TEXTURE_WINDOW_PX  = 5      # Sliding window size for texture (pixels; must be odd)
CONTEXT_WINDOW_PX  = 21     # Neighborhood window for z-score context (pixels; must be odd)

# TPI (Topographic Position Index) neighborhood — how far to look for "local summit vs valley"
# Larger = coarser landform classification.  ~1km at 30m res = 33px
TPI_WINDOW_PX      = 33     # Must be odd

# ── Model hyperparameters
RF_N_ESTIMATORS    = 500
RF_MAX_DEPTH       = 20
RF_MIN_SAMPLES_LEAF= 5
RF_N_JOBS          = 4      # Parallel jobs for RF training/prediction

XGB_N_ESTIMATORS   = 500
XGB_MAX_DEPTH      = 8
XGB_LEARNING_RATE  = 0.05
XGB_SUBSAMPLE      = 0.8
XGB_N_JOBS         = 4

# Weight of RF vs XGBoost in ensemble (must sum to 1.0)
ENSEMBLE_WEIGHT_RF  = 0.5
ENSEMBLE_WEIGHT_XGB = 0.5

# ── Training / evaluation
TEST_SPLIT         = 0.2    # Fraction of samples held out for evaluation
RANDOM_SEED        = 42
CLASS_WEIGHT       = "balanced"   # Handle class imbalance automatically

# ── Prediction
PREDICT_CHUNK_ROWS = 2048   # Rows to predict at once — reduce if RAM is tight

# ── Post-processing
# Minimum connected area (m²) for a detection to be kept as a polygon.
# Filters out isolated noisy pixels.
MIN_DETECTION_AREA_M2 = 5000    # ~0.5 ha — smallest plausible site

# Polygon simplification tolerance (degrees @ eq.)
POLYGON_SIMPLIFY_TOL  = 0.00009

# ── Output toggles
SAVE_CONFIDENCE_RASTER = True
SAVE_POINT_GEOJSON     = True
SAVE_POLYGON_GEOJSON   = True
SAVE_MODEL             = True    # Pickle trained ensemble for reuse
OVERWRITE              = False


#  END CONFIG


import gc
import math
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import rasterio.mask
import geopandas as gpd
import pandas as pd
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter, generic_filter
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import xgboost as xgb

warnings.filterwarnings("ignore")

np.seterr(divide="ignore", invalid="ignore")



# Feature names (built dynamically from config)


INDEX_NAMES = list(INDEX_PATHS.keys())

def _feature_names() -> list[str]:
    names = []
    if USE_RAW_INDICES:
        names += INDEX_NAMES
    if USE_INDEX_RATIOS:
        names += [
            "SABI_over_NDVI",
            "SAVI_over_NDVI",
            "MNDVI_over_NDVI",
            "MNDWI_over_NDWI",
            "SABI_minus_NDVI",
            "NDWI_minus_NDVI",
            "MNDWI_minus_MNDVI",
        ]
    if USE_TEXTURE:
        names += [f"{n}_texture" for n in INDEX_NAMES]
    if USE_CONTEXT_ZSCORE:
        names += [f"{n}_zscore" for n in INDEX_NAMES]
    if USE_TERRAIN:
        names += ["elevation", "slope_deg", "tpi", "twi"]
    return names

FEATURE_NAMES = _feature_names()



# I/O helpers

def open_indices() -> tuple[dict[str, rasterio.DatasetReader], dict]:
    """Open all index GeoTIFFs and the DEM; return (dataset dict, reference profile)."""
    datasets = {}
    profile = None
    for name, path in INDEX_PATHS.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {p}")
        ds = rasterio.open(p)
        datasets[name] = ds
        if profile is None:
            profile = ds.profile.copy()
    if USE_TERRAIN:
        p = Path(DEM_PATH)
        if not p.exists():
            raise FileNotFoundError(f"DEM file not found: {p}\n  (set USE_TERRAIN=False to skip)")
        datasets["DEM"] = rasterio.open(p)
    return datasets, profile


def read_window(
    datasets: dict[str, rasterio.DatasetReader],
    window: rasterio.windows.Window,
) -> dict[str, np.ndarray]:
    # Read a window from all open index datasets into float32 arrays.
    return {
        name: ds.read(1, window=window).astype(np.float32)
        for name, ds in datasets.items()
    }


# Feature engineering

def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.where(np.abs(b) > 1e-6, a / b, np.nan)
    return out.astype(np.float32)


def _local_variance(arr: np.ndarray, window: int) -> np.ndarray:
    # Local variance via E[X²] - E[X]² in a sliding window.
    mean   = uniform_filter(arr.astype(np.float64), size=window)
    mean_sq= uniform_filter(arr.astype(np.float64) ** 2, size=window)
    var    = mean_sq - mean ** 2
    return np.clip(var, 0, None).astype(np.float32)


def _local_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    # Z-score of each pixel relative to its local neighborhood.
    mean = uniform_filter(arr.astype(np.float64), size=window)
    std  = np.sqrt(np.clip(
        uniform_filter(arr.astype(np.float64) ** 2, size=window) - mean ** 2,
        0, None
    ))
    z = np.where(std > 1e-6, (arr - mean) / std, 0.0)
    return z.astype(np.float32)



def _compute_slope(dem: np.ndarray, res_deg: float = 0.00017966) -> np.ndarray:
    """
    Slope in degrees from a DEM (EPSG:4326).
    Uses Sobel gradients; res_deg is pixel size in degrees (~20m Sentinel res).
    For the Copernicus 30m DEM at equator: 1° ≈ 111320m, so res_m ≈ res_deg * 111320.
    """
    res_m = res_deg * 111320.0
    from scipy.ndimage import sobel
    dx = sobel(dem.astype(np.float64), axis=1) / (8.0 * res_m)
    dy = sobel(dem.astype(np.float64), axis=0) / (8.0 * res_m)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope.astype(np.float32)


def _compute_tpi(dem: np.ndarray, window: int) -> np.ndarray:
    """
    Topographic Position Index: elevation minus local mean.
    Negative → valley/channel bottom.  Positive → ridge/hilltop.
    """
    local_mean = uniform_filter(dem.astype(np.float64), size=window)
    return (dem - local_mean).astype(np.float32)


def _compute_twi(dem: np.ndarray, res_deg: float = 0.00017966) -> np.ndarray:
    """
    Simplified Topographic Wetness Index: ln(1 / (slope_rad + 0.001)).
    Higher → flatter terrain with more potential flow accumulation.
    A full D8/D-inf flow accumulation is impractical per-chunk; this
    slope-only proxy is a well-validated approximation for coarse screening
    (Sorensen et al. 2006).
    """
    res_m = res_deg * 111320.0
    from scipy.ndimage import sobel
    dx = sobel(dem.astype(np.float64), axis=1) / (8.0 * res_m)
    dy = sobel(dem.astype(np.float64), axis=0) / (8.0 * res_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    twi = np.log(1.0 / (slope_rad + 0.001))
    return twi.astype(np.float32)

def build_feature_array(
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Given a dict of 2D index arrays (H×W), compute all enabled features
    and return a (H*W, n_features) array.

    NaN pixels are preserved — callers must handle masking.
    """
    H, W = next(iter(arrays.values())).shape
    features = []

    # a) Raw indices
    if USE_RAW_INDICES:
        for name in INDEX_NAMES:
            features.append(arrays[name].ravel())

    # b) Index ratios — mine signatures vs background forest
    if USE_INDEX_RATIOS:
        ndvi  = arrays["NDVI"]
        ndwi  = arrays["NDWI"]
        mndvi = arrays["MNDVI"]
        mndwi = arrays["MNDWI"]
        savi  = arrays["SAVI"]
        sabi  = arrays["SABI"]

        features.append(_safe_divide(sabi,  ndvi ).ravel())   # high → cleared area
        features.append(_safe_divide(savi,  ndvi ).ravel())   # high → bare soil relative to veg
        features.append(_safe_divide(mndvi, ndvi ).ravel())   # vegetation structure anomaly
        features.append(_safe_divide(mndwi, ndwi ).ravel())   # water/soil interaction
        features.append((sabi  - ndvi ).ravel())              # large positive → clearing vs forest
        features.append((ndwi  - ndvi ).ravel())              # water vs veg
        features.append((mndwi - mndvi).ravel())              # SWIR vs NIR water signal

    # c) Texture — positives have high local variance (bare soil, pit edges, water)
    if USE_TEXTURE:
        w = TEXTURE_WINDOW_PX
        for name in INDEX_NAMES:
            features.append(_local_variance(arrays[name], w).ravel())

    # d) Context z-score — features are anomalies in surrounding area
    if USE_CONTEXT_ZSCORE:
        w = CONTEXT_WINDOW_PX
        for name in INDEX_NAMES:
            features.append(_local_zscore(arrays[name], w).ravel())

    # e) Terrain derivatives — used if features are terrain-dependent
    if USE_TERRAIN and "DEM" in arrays:
        dem = arrays["DEM"].copy()
        dem = np.where((dem == -9999) | ~np.isfinite(dem), np.nan, dem)
        features.append(dem.ravel())                                    # raw elevation
        features.append(_compute_slope(dem).ravel())                    # slope (degrees)
        features.append(_compute_tpi(dem, TPI_WINDOW_PX).ravel())       # TPI (valley = negative)
        features.append(_compute_twi(dem).ravel())                      # TWI proxy

    return np.column_stack(features).astype(np.float32)  # (H*W, n_features)



# Training sample extraction


def metres_to_degrees(metres: float, latitude: float = -2.0) -> float:
    """Approximate conversion at a given latitude (default: central DRC)."""
    lat_deg = metres / 111320.0
    lon_deg = metres / (111320.0 * math.cos(math.radians(latitude)))
    return (lat_deg + lon_deg) / 2.0


def buffer_sites(
    gdf: gpd.GeoDataFrame,
    radii_m: list[float],
    crs_degrees: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Buffer each site geometry at multiple radii (in metres).
    Returns a GeoDataFrame of buffered polygons in the original CRS.
    Uses an equal-area projection for accurate buffering.
    """
    gdf_proj = gdf.to_crs("EPSG:3857")  # Web Mercator — good enough for ~1km buffers
    buffered_parts = []
    for r in radii_m:
        b = gdf_proj.copy()
        b["geometry"] = gdf_proj.geometry.buffer(r)
        b["buffer_radius_m"] = r
        buffered_parts.append(b)
    combined = pd.concat(buffered_parts, ignore_index=True)
    return combined.to_crs(crs_degrees)


def extract_samples_from_geojson(
    geojson_path: str,
    datasets: dict[str, rasterio.DatasetReader],
    label: int,
    buffer_radii_m: list[float] | None = None,
    max_pixels_per_site: int = 500,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature vectors from pixels overlapping each GeoJSON feature.

    For positives: buffer at multiple radii and sample from each buffer ring.
    For negatives: sample directly from the polygon/point geometry.

    Returns (X, y) arrays.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    gdf = gpd.read_file(geojson_path)
    ref_ds = next(iter(datasets.values()))
    ref_crs = ref_ds.crs.to_string()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(ref_crs)

    if label == 1 and buffer_radii_m:
        gdf = buffer_sites(gdf, buffer_radii_m, crs_degrees=ref_crs)

    all_X, all_y = [], []

    for _, row in tqdm(gdf.iterrows(), total=len(gdf),
                       desc=f"  Extracting {'positive' if label else 'negative'} samples"):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        try:
            # Read a window around this geometry
            win = rasterio.windows.from_bounds(
                *geom.bounds,
                transform=ref_ds.transform,
            ).round_lengths().round_offsets()

            # Clamp to raster extent
            win = win.intersection(rasterio.windows.Window(
                0, 0, ref_ds.width, ref_ds.height
            ))
            if win.width < 1 or win.height < 1:
                continue

            win_transform = rasterio.windows.transform(win, ref_ds.transform)
            arrays = read_window(datasets, win)

            # Build feature matrix for this window
            feat = build_feature_array(arrays)  # (H*W, n_features)
            H, W = next(iter(arrays.values())).shape

            # Build pixel mask for this geometry
            mask = rasterio.features.geometry_mask(
                [mapping(geom)],
                out_shape=(H, W),
                transform=win_transform,
                invert=True,  # True = inside geometry
            ).ravel()

            # Also mask out NaN pixels
            nan_mask = np.any(~np.isfinite(feat), axis=1)
            valid = mask & ~nan_mask

            if valid.sum() == 0:
                continue

            feat_valid = feat[valid]

            # Subsample if too many pixels
            if len(feat_valid) > max_pixels_per_site:
                idx = rng.choice(len(feat_valid), max_pixels_per_site, replace=False)
                feat_valid = feat_valid[idx]

            all_X.append(feat_valid)
            all_y.append(np.full(len(feat_valid), label, dtype=np.int8))

        except Exception as e:
            print(f"    [WARN] Skipping geometry: {e}", flush=True)
            continue

    if not all_X:
        raise RuntimeError(f"No valid samples extracted from {geojson_path}")

    return np.vstack(all_X), np.concatenate(all_y)



# Model training

def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[RandomForestClassifier, xgb.XGBClassifier, StandardScaler]:
    """
    Train RF + XGBoost on training data, evaluate on test set.
    Returns (rf_model, xgb_model, scaler).
    """
    print(f"\n  Training samples : {(y_train == 1).sum():,} positive, "
          f"{(y_train == 0).sum():,} negative", flush=True)
    print(f"  Test samples     : {(y_test == 1).sum():,} positive, "
          f"{(y_test == 0).sum():,} negative", flush=True)
    print(f"  Features         : {X_train.shape[1]} ({', '.join(FEATURE_NAMES)})\n",
          flush=True)

    # Scale features (XGBoost doesn't strictly need it but helps RF with NaN filling)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
    X_test_s  = scaler.transform(np.nan_to_num(X_test,  nan=0.0))

    # Random Forest
    print("  Training Random Forest...", flush=True)
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_SEED,
        oob_score=True,
    )
    rf.fit(X_train_s, y_train)
    print(f"    OOB accuracy: {rf.oob_score_:.4f}", flush=True)

    # XGBoost
    print("  Training XGBoost...", flush=True)
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        scale_pos_weight=scale_pos,
        n_jobs=XGB_N_JOBS,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb_model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )

    # Evaluate ensemble on test set
    rf_prob  = rf.predict_proba(X_test_s)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test_s)[:, 1]
    ens_prob = ENSEMBLE_WEIGHT_RF * rf_prob + ENSEMBLE_WEIGHT_XGB * xgb_prob
    ens_pred = (ens_prob >= CONFIDENCE_THRESHOLD).astype(int)

    print("\n  ── Ensemble Evaluation (test set) ──", flush=True)
    print(classification_report(y_test, ens_pred,
                                 target_names=["background", "mine"]), flush=True)
    print(f"  ROC-AUC            : {roc_auc_score(y_test, ens_prob):.4f}", flush=True)
    print(f"  Avg Precision (AP) : {average_precision_score(y_test, ens_prob):.4f}",
          flush=True)

    cm = confusion_matrix(y_test, ens_pred)
    print(f"\n  Confusion matrix (threshold={CONFIDENCE_THRESHOLD}):", flush=True)
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}", flush=True)
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}", flush=True)

    # Feature importances
    fi_rf  = rf.feature_importances_
    fi_xgb = xgb_model.feature_importances_
    fi_ens = ENSEMBLE_WEIGHT_RF * fi_rf + ENSEMBLE_WEIGHT_XGB * fi_xgb
    fi_sorted = sorted(zip(FEATURE_NAMES, fi_ens), key=lambda x: -x[1])
    print("\n  Top 10 feature importances (ensemble):", flush=True)
    for fname, fi in fi_sorted[:10]:
        bar = "█" * int(fi * 200)
        print(f"    {fname:<30s} {fi:.4f}  {bar}", flush=True)

    return rf, xgb_model, scaler


# Full-mosaic prediction

def predict_mosaic(
    datasets: dict[str, rasterio.DatasetReader],
    rf: RandomForestClassifier,
    xgb_model: xgb.XGBClassifier,
    scaler: StandardScaler,
    profile: dict,
    out_path: Path,
) -> None:
    """
    Predict confidence scores across the full mosaic in row-chunks.
    Writes a Float32 GeoTIFF of mine confidence [0, 1].
    """
    ref_ds   = next(iter(datasets.values()))
    height   = ref_ds.height
    width    = ref_ds.width
    n_chunks = math.ceil(height / PREDICT_CHUNK_ROWS)

    out_profile = profile.copy()
    out_profile.update(
        dtype="float32", count=1, nodata=-1.0,
        compress="lzw", tiled=True,
        blockxsize=512, blockysize=512, bigtiff="YES",
    )

    print(f"\n  Writing confidence raster → {out_path}", flush=True)

    with rasterio.open(out_path, "w", **out_profile) as dst:
        for i in tqdm(range(n_chunks), desc="  Predicting"):
            row_off = i * PREDICT_CHUNK_ROWS
            rows    = min(PREDICT_CHUNK_ROWS, height - row_off)
            window  = rasterio.windows.Window(0, row_off, width, rows)

            arrays = read_window(datasets, window)
            feat   = build_feature_array(arrays)           # (rows*width, n_feat)
            nan_px = np.any(~np.isfinite(feat), axis=1)   # pixels with any NaN

            feat_clean = scaler.transform(np.nan_to_num(feat, nan=0.0))

            rf_prob  = rf.predict_proba(feat_clean)[:, 1]
            xgb_prob = xgb_model.predict_proba(feat_clean)[:, 1]
            conf     = (ENSEMBLE_WEIGHT_RF * rf_prob +
                        ENSEMBLE_WEIGHT_XGB * xgb_prob).astype(np.float32)

            conf[nan_px] = -1.0   # nodata for missing input pixels

            dst.write(conf.reshape(rows, width), 1, window=window)
            gc.collect()


# Post-processing: raster → vectors

def raster_to_vectors(
    conf_path: Path,
    profile: dict,
    out_dir: Path,
) -> None:
    """
    Threshold confidence raster → binary mask → vectorise → filter by area.
    Writes point centroid GeoJSON and polygon GeoJSON.
    """
    print(f"\n  Vectorising detections (threshold={CONFIDENCE_THRESHOLD})...",
          flush=True)

    with rasterio.open(conf_path) as src:
        conf = src.read(1)
        transform = src.transform
        crs = src.crs

    binary = (conf >= CONFIDENCE_THRESHOLD).astype(np.uint8)
    total_px = int(binary.sum())
    print(f"  Pixels above threshold: {total_px:,}", flush=True)

    if total_px == 0:
        print("  [WARN] No detections above threshold — "
              "try lowering CONFIDENCE_THRESHOLD.", flush=True)
        return

    # Vectorise connected components
    shapes = list(rasterio.features.shapes(
        binary, mask=binary, transform=transform
    ))
    print(f"  Raw polygons: {len(shapes):,}", flush=True)

    # Build GeoDataFrame
    records = []
    px_area = abs(transform.a * transform.e)   # area of one pixel in CRS units²
    # Approximate m² per degree² at equator (~12,300 km²/degree² → 1.23e10 m²/deg²)
    # For EPSG:4326 the pixel area is in degrees² — convert to m²
    deg2_to_m2 = 111320.0 ** 2

    for geom_dict, val in shapes:
        if val != 1:
            continue
        geom = shape(geom_dict)
        area_m2 = geom.area * deg2_to_m2
        if area_m2 < MIN_DETECTION_AREA_M2:
            continue
        simplified = geom.simplify(POLYGON_SIMPLIFY_TOL, preserve_topology=True)
        centroid   = simplified.centroid
        records.append({
            "geometry_poly": simplified,
            "geometry_pt":   centroid,
            "area_m2":       round(area_m2, 1),
            "area_ha":       round(area_m2 / 10000, 2),
            "lon":           round(centroid.x, 6),
            "lat":           round(centroid.y, 6),
        })

    print(f"  Detections after area filter (≥{MIN_DETECTION_AREA_M2}m²): "
          f"{len(records):,}", flush=True)

    if not records:
        print("  [WARN] All detections filtered out — "
              "try lowering MIN_DETECTION_AREA_M2.", flush=True)
        return

    # Polygon GeoJSON
    if SAVE_POLYGON_GEOJSON:
        poly_gdf = gpd.GeoDataFrame(
            [{k: v for k, v in r.items() if k != "geometry_pt"
              and k != "geometry_poly"} | {"geometry": r["geometry_poly"]}
             for r in records],
            crs=crs,
        )
        poly_path = out_dir / "detections_polygons.geojson"
        poly_gdf.to_file(poly_path, driver="GeoJSON")
        print(f"  ✓ Polygons → {poly_path}  ({len(poly_gdf)} features)", flush=True)

    # Point GeoJSON (centroids)
    if SAVE_POINT_GEOJSON:
        pt_gdf = gpd.GeoDataFrame(
            [{k: v for k, v in r.items() if k != "geometry_pt"
              and k != "geometry_poly"} | {"geometry": r["geometry_pt"]}
             for r in records],
            crs=crs,
        )
        pt_path = out_dir / "detections_points.geojson"
        pt_gdf.to_file(pt_path, driver="GeoJSON")
        print(f"  ✓ Centroids → {pt_path}  ({len(pt_gdf)} features)", flush=True)


# Main pipeline

def run() -> None:
    print("\n=== Gold Mine Detection Pipeline ===\n", flush=True)
    print(f"  Confidence threshold : {CONFIDENCE_THRESHOLD}", flush=True)
    print(f"  Buffer radii (m)     : {BUFFER_RADII_M}", flush=True)
    print(f"  Features enabled     : raw={USE_RAW_INDICES}  ratios={USE_INDEX_RATIOS}  "
          f"texture={USE_TEXTURE}  zscore={USE_CONTEXT_ZSCORE}  terrain={USE_TERRAIN}", flush=True)
    print(f"  Ensemble weights     : RF={ENSEMBLE_WEIGHT_RF}  "
          f"XGB={ENSEMBLE_WEIGHT_XGB}\n", flush=True)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Open index rasters
    print("Step 1 — Opening index rasters...", flush=True)
    datasets, profile = open_indices()
    for name, ds in datasets.items():
        label = " (DEM)" if name == "DEM" else ""
        print(f"  {name:<6s}  {ds.width}×{ds.height}px{label}", flush=True)
    if USE_TERRAIN:
        print(f"  Terrain features   : elevation, slope, TPI (window={TPI_WINDOW_PX}px), TWI", flush=True)

    # ── Extract training samples
    print("\nStep 2 — Extracting training samples...", flush=True)

    print("  Positive sites (known mines):", flush=True)
    X_pos, y_pos = extract_samples_from_geojson(
        POSITIVE_GEOJSON, datasets, label=1,
        buffer_radii_m=BUFFER_RADII_M,
    )
    print(f"  → {len(X_pos):,} positive pixels extracted", flush=True)

    print("  Negative sites:", flush=True)
    X_neg, y_neg = extract_samples_from_geojson(
        NEGATIVE_GEOJSON, datasets, label=0,
        buffer_radii_m=None,
    )
    print(f"  → {len(X_neg):,} negative pixels extracted", flush=True)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    print(f"\n  Total samples: {len(X):,}  |  "
          f"positive={y.sum():,}  negative={(y==0).sum():,}", flush=True)

    # ── Train/test split and model training
    print("\nStep 3 — Training ensemble model...", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    rf, xgb_model, scaler = train_ensemble(X_train, y_train, X_test, y_test)

    # Save model
    if SAVE_MODEL:
        model_path = out_dir / "ensemble_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"rf": rf, "xgb": xgb_model, "scaler": scaler,
                         "feature_names": FEATURE_NAMES,
                         "threshold": CONFIDENCE_THRESHOLD}, f)
        print(f"\n  ✓ Model saved → {model_path}", flush=True)

    # ── Predict full mosaic
    conf_path = out_dir / "confidence.tif"

    if conf_path.exists() and not OVERWRITE:
        print(f"\nStep 4 — Confidence raster already exists ({conf_path}). "
              f"Set OVERWRITE=True to redo.", flush=True)
    else:
        print("\nStep 4 — Predicting full mosaic...", flush=True)
        if SAVE_CONFIDENCE_RASTER:
            predict_mosaic(datasets, rf, xgb_model, scaler, profile, conf_path)
        else:
            print("  SAVE_CONFIDENCE_RASTER=False — skipping.", flush=True)

    # ── Vectorise detections
    print("\nStep 5 — Vectorising detections...", flush=True)
    if conf_path.exists():
        raster_to_vectors(conf_path, profile, out_dir)
    else:
        print("  No confidence raster found — skipping vectorisation.", flush=True)

    # Cleanup
    for ds in datasets.values():
        ds.close()

    print("\n=== Pipeline complete ===", flush=True)
    print(f"Outputs in: {out_dir}", flush=True)
    for p in sorted(out_dir.iterdir()):
        size = p.stat().st_size / 1e6
        print(f"  {p.name:<40s}  {size:>8.1f} MB", flush=True)



# Reuse a saved model without retraining

def predict_from_saved_model(model_path: str) -> None:
    """
    Load a previously saved ensemble and run prediction + vector calculation only.
    Useful for tweaking CONFIDENCE_THRESHOLD without retraining.

    Usage:
        from detect_mines import predict_from_saved_model
        predict_from_saved_model("/path/to/ensemble_model.pkl")
    """
    print(f"\nLoading model from {model_path}...", flush=True)
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    rf        = bundle["rf"]
    xgb_model = bundle["xgb"]
    scaler    = bundle["scaler"]

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, profile = open_indices()
    conf_path = out_dir / "confidence.tif"
    predict_mosaic(datasets, rf, xgb_model, scaler, profile, conf_path)
    raster_to_vectors(conf_path, profile, out_dir)

    for ds in datasets.values():
        ds.close()


if __name__ == "__main__":
    run()
