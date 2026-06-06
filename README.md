> *For research use only. Not for clinical decisions. Copyright (c) 2026, Payton Jachec*

# S2_Feature_finder
An ML process for locating classes of features using RF and XGBoost across an extent using common geospatial analysis indices as inputs.

## S2_Feature_Finder Contents
- ml_pipe.py
- environment.yml

## Required Packages and Environment
### Known Working Configuration
| Package | Version |
| ----------- | ----------- |
| arrow | 1.4.0 |
| bokeh | 3.9.0 |
| dask | 2026.3.0 |
| distributed | 2026.3.0 |
| geopandas | 1.1.3 |
| mgrs | 1.5.4 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| planetary-computer | 1.0.0 |
| pystac-client | 0.9.0 |
| pystac | 1.14.3 |
| python | 3.11.15 |
| rasterio | 1.4.4 |
| scikit-learn | 1.9.0 |
| scipy | 1.17.1 |
| shapely | 2.1.2 |
| stackstac | 0.5.0 |
| tqdm | 4.68.0 |
| xarray | 2026.4.0 |
| xgboost | 3.2.0 |

### Required Channels
- conda-forge

### Environment Setup
It is recommended to use Anaconda or Mamba to create a venv for this program suite. 
If running on an HPC, default configurations will almost certainly break the code. 

**Suggested Environment**

> ~] $ micromamba create -n ENV_NAME python=3.11.15 -c conda-forge bokeh dask distributed geopandas mgrs numpy pandas>=2.2.3 planetary-computer pystac-client rasterio scikit-learn scipy shapely stackstac tqdm xarray xgboost

or using the **included environment.yml**
> ~] $ micromamba create -n ENV_NAME environment.yml

## Job Submission
### ml_pipe.py
This script takes geospatial analysis indices, positive locations, and negative locations as inputs and then outputs all positive locations over the extent.

| Field | Line | Use |
| -------- | -------- | -------- |
| INDEX_PATHS | 13-19 | Sets paths to analysis indices |
| DEM_PATH | 22 | Sets path to DEM file |
| POSITIVE_GEOJSON | 25 | Sets path to geojson of known positives |
| NEGATIVE_GEOJSON | 26 | Sets path to geojson of known negatives |
| OUT_DIR | 29 | Sets path to output directory |
| BUFFER_RADII_M | 34 | Sets buffer radii around known locations |
| CONFIDENCE_THRESHOLD | 40 | Sets threshold (0-1) for a location to be considered a positive |
| USE_RAW_INDICES | 43 | Toggles direct use of analysis indices |
| USE_TERRAIN | 44 | Toggles use of DEM-derived terrain variables |
| USE_INDEX_RATIOS | 45 | Toggles use of analysis index ratios |
| USE_TEXTURE | 46 | Toggles use of local variability in other predictors |
| USE_CONTEXT_ZSCORE | 47 | Toggles use of z-score comparred to surrounding area |
| TEXTURE_WINDOW_PX | 49 | Sets window for texture pixel groups (must be odd) |
| CONTEXT_WINDOW_PX | 50 | Sets neighborhood window for z-score context (must be odd) |
| TPI_WINDOW_PX | 54 | Sets context window for TRI (topographic position index) (must be odd) |
| RF_N_ESTIMATORS | 57 | Sets number of decision trees for random forest |
| RF_MAX_DEPTH | 58 | Sets max depth per tree in random forest |
| RF_MIN_SAMPLES_LEAF | 59 | Sets minimum number of pixels required for each tree to continue growing |
| RF_N_JOBS | 60 | Sets number of CPU cores to use for parallel tasks |
| XGB_N_ESTIMATORS | 62 | Sets number of boosting rounds for XGBoost |
| XGB_MAX_DEPTH | 63 | Sets depth of boosting rounds for XGBoost |
| XGB_LEARNING_RATE | 64 | Sets learning (error correction) rate for each subsequent boost round |
| XGB_SUBSAMPLE | 65 | Sets fraction of training pixels randomly sampled to grow each boost tree in XGBoost |
| XGB_N_JOBS | 66 | Sets number of parallel jobs for training/prediction |
| ENSEMBLE_WEIGHT_RF | 69 | Sets weight of RF in ensemble |
| ENSEMBLE_WEIGHT_XGB | 70 | Sets weight of XGBoost in ensemble |
| TEST_SPLIT | 73 | Sets fraction of samples held for evaluation |
| RANDOM_SEED | 74 | Sets random seed for entire ensemble |
| CLASS_WEIGHT | 75 | Toggles between methods of handling class imbalance |
| PREDICT_CHUNK_ROWS | 78 | Sets number of rows to predict at once |
| MIN_DETECTION_AREA_M2 | 83 | Sets plausible minimum area on square meters of a positive location |
| POLYGON_SIMPLIFY_TOL | 86 | Sets polygon simplification in degrees |
| SAVE_CONFIDENCE_RASTER | 89 | Toggles whether the confidence raster is saved to the output directory |
| SAVE_POINT_GEOJSON | 90 | Toggles whether the geojson of calculated positive location centroids is saved to the output directory |
| SAVE_POLYGON_GEOJSON | 91 | Toggles whether the geojson of calculated positive location polygons is saved to the output directory |
| SAVE_MODEL | 92 | Toggles whether the model parameters and weights are saved to reuse without training |
| OVERWRITE | 93 | Toggles whether existing outputs are overwritten |

Then, 
> ~] $ python3 ml_pipe.py

> *For research use only. Not for clinical decisions. Copyright (c) 2026, Payton Jachec*
