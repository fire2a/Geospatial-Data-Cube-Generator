# Geospatial-Data-Cube-Genrator

This repository contains two Google Colab notebooks designed for geospatial data processing and analysis using Google Earth Engine (GEE). The tools generate a comprehensive data cube of environmental and spectral variables for a specified region of interest (ROI). 

## Key Features

- **Topographic Variables**: Elevation, slope, aspect, landforms, and TP.
- **Climatic Variables**: Total precipitation and average temperature over defined periods.
- **Spectral Indices**: NDVI, NDBI, NDWI, EVI from Sentinel-2 imagery.
- **Sentinel-2 Variables**: Includes Red, Green, Blue, and additional spectral bands.
- **Radar Data**: Sentinel-1 VV and VH polarizations with slope correction.
- **EVI Statistics**: Vegetation dynamics over specified timeframes.

## Included Notebooks

### 1. **[CPU] GEE - Data Cube Generator**
This notebook is optimized for processing with CPU resources and is suitable for most typical ROIs. Start with this notebook to generate your data cube efficiently.

### 2. **[GPU + CPU] GEE - Data Cube Generator**
This notebook uses GPU resources during preprocessing (e.g., cloud correction or slope correction). It's designed for handling larger or more complex regions where GPU acceleration may improve performance. Once preprocessing is complete, it switches to CPU for data export to avoid unnecessary GPU usage.

### Recommended Workflow
1. Start with the **CPU version** to process your region of interest.
2. If you encounter memory or performance issues with larger regions, use the **GPU version** for preprocessing and switch to CPU for exporting.

## How to Use

### Prerequisites
- A [Google Earth Engine](https://earthengine.google.com/) account.
- A Google Drive account for data storage.
- A region file in `.shp` or `.gpkg` format.

### General Steps
1. Open the desired notebook in Google Colab.
2. Follow the instructions within the notebook:
   - Authenticate with GEE and Google Drive.
   - Specify your region file and desired parameters (e.g., season year, export CRS).
3. Run the preprocessing and export steps as outlined.

### Important Notes
- All variables are processed in EPSG:4326 and exported in the specified CRS.
- Be cautious about switching runtime environments (CPU/GPU) as directed in the GPU notebook to optimize resource usage.

## Output
- Geospatial variables exported as GeoTIFFs in Google Drive.
- Organized data for easy integration into GIS tools like QGIS or ArcGIS.
