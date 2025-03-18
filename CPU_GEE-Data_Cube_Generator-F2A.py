# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="Li_TRYN9YJtq"
# # GEE - Data Cube Generator
#
# ## Introduction
#
# This Colab notebook is designed to process and analyze geospatial data using Google Earth Engine (GEE). It focuses on deriving topographic, climatic, and spectral information for a specified region of interest (ROI). By leveraging GEE's powerful cloud computing capabilities, this workflow enables efficient data handling and export of key environmental variables for further analysis.
#
#
# ### What Does This Notebook Do?:
#
# 1) Loads a Region of Interest (ROI):
#
# - Accepts shapefiles or GeoPackages to define the area for analysis.
# - Applies spatial buffers to adapt the region for specific datasets.
#
# 2) Processes Geospatial Variables:
#
# - Topographic Variables: Elevation, slope, aspect, landforms, and topographic position index (TPI).
# - Climate Variables: Total precipitation and average temperature over a defined period.
# - Spectral Indices: NDVI, NDBI, NDWI, and EVI derived from Sentinel-2 imagery.
# - Sentinel-2 Variables: Red, Green, Blue, R1, R2, R3, NIR, R4, SWIR1, and SWIR2 bands with cloud correction included.
# - Radar Data: VV and VH polarizations from Sentinel-1 with slope correction applied.
# - EVI Statistics: Summarizes vegetation dynamics for selected time periods.
#
# 3) Defines and Analyzes Temporal Periods:
#
# - Period 1 (P1): 15 March (desired_year) - 30 May (desired_year).
# - Period 2 (P2): 01 June (desired_year) - 15 August (desired_year).
# - Period 3 (P3): 15 November (previous_year) - 15 February (desired_year).
#
# 4) Exports Results to Google Drive:
#
# - The variables are organized into batches and exported as GeoTIFF files to Google Drive for easy integration with GIS tools like QGIS or ArcGIS.
#
#
# ### Prerequisites:
#
# To use this notebook, you will need:
#
# - A Google Earth Engine account.
# - Basic familiarity with Python and geospatial concepts.
# - A GeoPackage (.gpkg) or Shapefile (.shp) defining your region of interest.
#
# ## Instructions and considerations
#
# This are the steps to follow for a correct function of the Google Colab:
#
# - Go to the [Google Earth Engine web site](https://code.earthengine.google.com/) and sing in to create a proyect.
# - Make sure that the CPU is selected, go to **"Runtime"** on the menu above and go to **"Change runtime type"** to check it.
# - Inside the cell **"Set-up and User Inputs"** make the next changes:
#   - In the cell **"/Initializing Google Earth Engine"** insert your GEE proyect id.
#   - Execute the cell **Mounting Google Drive**
#   - Select the paths of the region file (.shp or .gpkg) stored on your drive by using the panel on the right by selecting the folder icon and then right-click the desired file and copy the path and paste it on the variable called **uploaded_file**. (If the region is too complex is recommended to simplify the region before).
#   - Define the name of the folder where the output will be stored in the variable **"drive_folder"**.
#   - Define de CRS which the tiff files will be downloaded in the variable **"export_crs"** (by default is set to EPSG:32718).
#   - Insert the year that you want to extract the images from on the variable **season_year**.
#
#    (Ex: If I select the year "2021" the images obtained will be for the 2021-2022 Fire Season)
# - After doing the changes above you can execute all the cells by executing the cell **Processing**.
#
# ### **Other Considerations**
# - All the varaibles are processed in EPSG:4326 and then downloaded in the desired CRS (NASADEM variables can't be directly downloaded with some CRS so we donwload them in EPSG:4326 and then it's changed).
# - While a cell is executing you can press the execute button of other cells and they will be executed after the previous cell is done.

# %% [markdown] id="WnmF0VDniLPJ"
# # Processing

# %% [markdown] id="Zb7mMmCeUcdq"
# ## Installing and Importing Required Libraries
#

# %% [markdown] id="kXjQGmeGfLb6"
# ### Install Libraries

# %% [markdown] id="0CKId_FreLe_"
# To begin, we need to install some Python libraries that are not pre-installed in Google Colab but are necessary for this project:
#
# - `xarray`: For handling multi-dimensional arrays.
# - `rioxarray`: For geospatial raster operations.
# - `rasterio`: For reading and writing geospatial raster data.
# - `numpy`: For numerical operations.
# - `matplotlib`: For visualizing data.
# - `pyproj`: For handling cartographic projections and CRS transformations.
#

# %% id="E5SgTIT8dUYN"
# !pip install xarray rioxarray rasterio numpy matplotlib pyproj

# %% [markdown] id="VAnQN4azfPt5"
# ### Import Libraries

# %% [markdown] id="WOEzG8WWeBRT"
# This section imports the following libraries:
# - `ee`: The Google Earth Engine (GEE) API for managing geospatial data.
# - `geopandas`: For working with vector geospatial data.
# - `google.colab.drive`: To access files stored in your Google Drive.
# - `os`: For directory and file path management.
# - `time`: For time measuring when downloading files from GEE.
# - `pickle`: For storing variables
# - `rasterio`:For reading, writing, and transforming geospatial raster data.

# %% id="nL9J6owSeESs"
import ee
import geopandas as gpd
from google.colab import drive
import os
import time
import pickle
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
import nummpy as np
from milib import apply_region_mask

# %% [markdown] id="hB4oty-qU-wf"
# ## Setup and User Inputs
#
#

# %% [markdown] id="pLjQci1OeSVj"
# ### Initializing Google Earth Engine (GEE)

# %% [markdown] id="1Nm9logeS-7-"
# How to Set Up a GEE Project
# - Visit the Google Earth Engine website.
# - Click on "Sign In" and use your Google account to authenticate.
# - Go to the "Projects" section in the Console.
# - Create a new project and note down the Project ID.
# - Replace 'your_project_id' in the ee.Initialize() call with your actual Project ID.
#
#

# %% id="ALOU0_T27p5I"
# Authenticate GEE
ee.Authenticate()

# Initialize GEE with the user's project
# IMPORTANT: Replace 'your_project_id' with your actual project ID from GEE
ee.Initialize(project="ee-your_project_id")

# %% [markdown] id="yM6M0xq3ekSK"
# ### Mounting Google Drive

# %% id="qceIJKTBehM_"
# Mount Google Drive with your Google account to access files
drive.mount("/content/drive")

# %% [markdown] id="C9UlPJRtf_zR"
# ### User Input: Define Paths and desired year to export the files

# %% [markdown] id="PZKJdZ5UTSMH"
# How to Reference Your Shapefile/GeoPackage after you mount Google Drive
# - Open the left panel in Colab and click the "Files" tab.
# - Upload your .gpkg or .shp file to a folder in your Google Drive.
# - Copy the full path of your file (e.g., /content/drive/MyDrive/YourFolder/your_file.gpkg).
# - Enter this path in the uploaded_file variable at the start of the code.
# - (If you need to simplify the region do it before importing the file)
#

# %% id="-EgQJ5Rj7xCF"
# Prompt the user to provide the path of their shapefile (.shp) or GeoPackage (.gpkg)
# Example: "/content/drive/MyDrive/YourFolder/your_file.gpkg"
uploaded_file = "/content/drive/MyDrive/YourFolder/your_file.gpkg"

# Prompt the user to provide the folder in Google Drive where outputs will be saved (e.g., GEE_FuelM)
drive_folder = "Folder Name"

# Define de CRS which the tiff files will be downloaded
export_crs = "EPSG:32718"

# Prompt the user to define the year of the desired wildfire season (from november/december of the selected year to march of the next year)
season_year = 2021
print(f"The Images will be obtained for the {season_year}-{season_year+1} wildfire season")

# %% [markdown] id="Fvm9Biy6VMD3"
# ## Loading the Region of Interest (ROI) and Defining Buffers
#
#

# %% [markdown] id="tzKnAlD9kuES"
# ### Region of Interest

# %% [markdown] id="F645y3YZdJWZ"
# This section:
# - Loads the shapefile or GeoPackage into a GeoDataFrame (`gpd`).
# - Converts the GeoDataFrame into a Google Earth Engine `FeatureCollection`.
#
# This FeatureCollection defines the area of interest (AOI) for processing.

# %% id="LZ0Gzo9gVPVH"
# Load the vector file into a GeoDataFrame
gdf = gpd.read_file(uploaded_file)
print("Loaded Region of Interest (ROI):")
print(gdf)

# Ensure CRS is EPSG:4326
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)

# Convert GeoDataFrame to Earth Engine FeatureCollection
region = ee.FeatureCollection(gdf.__geo_interface__)

# %% [markdown] id="UC0IUVFDVria"
# ### Buffers
#
#

# %% [markdown] id="DxRuqIs-dF0Q"
# Buffers are applied to the region to extend the area of analysis based on specific requirements:
# - **100m Buffer:** General buffer for region processing.
# - **5000m Buffer:** For climate data variables.
# - **500m Buffer:** For TPI (Topographic Position Index).
# - **200m Buffer:** For landforms.
#
# Each buffer is created using the `buffer()` method in GEE.
#

# %% id="2qFNhDEbVkrb"
# Create a 100m buffer around the region
region_buffered = region.map(lambda f: f.buffer(100))

# Define specific buffers for different datasets
climate_buffer = region.map(lambda f: f.buffer(5000))
tpi_buffer = region.map(lambda f: f.buffer(500))
landform_buffer = region.map(lambda f: f.buffer(200))

# %% [markdown] id="MQq9lHhyV2Pp"
# ## Defining Analysis Periods and Helper Functions
#

# %% [markdown] id="2BWc62M-lAWg"
# ### Analysis Periods

# %% [markdown] id="UpHnYk-vWXa9"
# The script defines three periods for temporal analysis:
# - **Period 1 (P1):** 15 March (selected year) - 30 May (selected year).
# - **Period 2 (P2):** 01 June (selected year) - 15 August (selected year).
# - **Period 3 (P3):** 15 November (previous year) - 15 February (selected year).
#
#
#

# %% id="kF5Y0sGI-jbd"
periods = [
    (f"{season_year}-03-15", f"{season_year}-05-30"),  # Period 1 (P1)
    (f"{season_year}-06-01", f"{season_year}-08-15"),  # Period 2 (P2)
    (f"{season_year-1}-11-15", f"{season_year}-02-15"),  # Period 3 (P3)
]


# %% [markdown] id="zdm290_jlVDR"
# ### Helper Functions

# %% [markdown] id="rjq9eun5coI3"
# - `apply_region_mask`: Clips an image to the region of interest (ROI) and assigns a NoData value to pixels outside the ROI.


# %% id="mZbDmAKo76a1"


# %% [markdown] id="trD44a5TlPDk"
# A helper function (`get_period_name`) maps each period to its corresponding name (P1, P2, or P3).


# %% id="smgHtyTMlQyB"
def get_period_name(period):
    """
    Returns the name of the period based on the start and end dates.

    Args:
        period: Tuple with start_date and end_date.

    Returns:
        Name of the period (e.g., 'P1', 'P2', 'P3').
    """
    if period == periods[0]:
        return "P1"
    elif period == periods[1]:
        return "P2"
    elif period == periods[2]:
        return "P3"


# %% [markdown] id="Z-bmXjnbWA0T"
# ## Data Processing

# %% [markdown] id="CaGdEn4KV9Wr"
# ### Processing Topographic Variables
#
#

# %% [markdown] id="U8vdb8s9rIXY"
# This section calculates and combines the following topographic variables:
# - Elevation (from NASADEM).
# - Slope (derived from elevation).
# - Aspect (orientation of slope).
# - Landforms (specific topographic features).
# - Topographic Position Index (TPI).
#
# All these variables are combined into a single image for export.

# %% id="4Qds8xGW78Dx"
# Elevation, slope, aspect
elevation = apply_region_mask(ee.Image("NASA/NASADEM_HGT/001").select("elevation").toFloat(), region_buffered)
slope = apply_region_mask(ee.Terrain.slope(elevation).rename("slope").toFloat(), region_buffered)
aspect = apply_region_mask(ee.Terrain.aspect(elevation).rename("aspect").toFloat(), region_buffered)

# Landforms (200m buffer)
landform = apply_region_mask(
    ee.Image("CSP/ERGo/1_0/Global/SRTM_landforms").select("constant").rename("landform").toFloat(), landform_buffer
)

# TPI (500m buffer)
tpi = apply_region_mask(ee.Image("CSP/ERGo/1_0/Global/SRTM_mTPI").rename("TPI").toFloat(), tpi_buffer)

# Combine all NASADEM variables into a single image
nasadem_variables = apply_region_mask(ee.Image([elevation, aspect, slope]), region)

# Clip Landform and Tpi Variables to region size
landform = apply_region_mask(landform, region)
tpi = apply_region_mask(tpi, region)

# %% [markdown] id="LE5tIy-9WSDq"
# ### Processing Climate Variables
#
#

# %% [markdown] id="vlUtD9HnrSq2"
# This section calculates:
# - **Total Precipitation:** Summed over a one-year period.
# - **Average Temperature:** Mean temperature over the same period.
#
# These variables are combined into a single image for export.

# %% id="6A3H8x497-CH"
# Climate data: total precipitation and average temperature
climate = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filter(
    ee.Filter.date(f"{season_year-1}-07-01", f"{season_year}-07-01")
)

precipitation = apply_region_mask(
    climate.select("total_precipitation_sum").sum().rename("precipitation").toFloat(), climate_buffer
)
temperature = apply_region_mask(climate.select("temperature_2m").mean().rename("temperature").toFloat(), climate_buffer)

# Combine all climate variables into a single image
climate_variables = apply_region_mask(ee.Image([precipitation, temperature]), region)

# %% [markdown] id="EcDMs62JWgIb"
# ### Sentinel-2 Processing
#
#

# %% [markdown] id="JU1xDtNYrbJh"
# - Fetches Sentinel-2 surface reflectance data with cloud masking applied.
# - Calculates spectral indices:
#   - NDVI (Normalized Difference Vegetation Index).
#   - NDBI (Normalized Difference Built-Up Index).
#   - NDWI (Normalized Difference Water Index).
#   - EVI (Enhanced Vegetation Index).
#
#   These indices are added as additional bands for further analysis.
#
# - Get the main variables of Sentinel-2:
#   - RGB (Red-Green-Blue)
#   - R1, R2, R3 and R4 (Red Edges)
#   - NIR (Near InfraRed)
#   - SWIR 1 and 2 (Short Wave InfraRed)
#
# The cloud correction function retrieves Sentinel-2 images and their cloud probability data, filters them by area, date, and cloud thresholds, and masks pixels with high cloud probability. It joins the reflectance and cloud data, applies cloud masks, and computes spectral indices like NDVI, NDBI, NDWI, and EVI. The cleaned images are then sorted by date and aggregated for analysis, ensuring only cloud-free data is used.

# %% id="koMnpO8-BCHV"
# Define cloud threshold and mask probability
scene_cloud_threshold = 60
cloud_mask_probability = 30


def add_indices(image):
    """
    Adds spectral indices (NDVI, NDBI, NDWI, EVI) to a Sentinel-2 image.
    """
    ndbi = (
        image.expression(
            "(SWIR - NIR) / (SWIR + NIR)",
            {
                "SWIR": image.select("swir1"),
                "NIR": image.select("nir"),
            },
        )
        .multiply(100)
        .rename("ndbi")
    )

    ndvi = image.normalizedDifference(["nir", "red"]).multiply(100).rename("ndvi")
    ndwi = image.normalizedDifference(["green", "nir"]).multiply(100).rename("ndwi")
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": image.select("nir"),
            "RED": image.select("red"),
            "BLUE": image.select("blue"),
        },
    ).rename("evi")

    return image.addBands([ndvi, ndbi, ndwi, evi])


def mask_clouds(cloud_probability_threshold):
    """
    Masks clouds based on a given probability threshold.
    """

    def _mask_image(img):
        cloud_mask = img.select("probability").lt(cloud_probability_threshold)
        return img.updateMask(cloud_mask)

    return _mask_image


def get_s2_sr_cloud_probability(aoi, start_date, end_date, scene_cloud_threshold, cloud_mask_probability):
    """
    Fetches Sentinel-2 surface reflectance images with cloud masking.
    """
    primary = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", scene_cloud_threshold)
        .select(
            ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
            ["blue", "green", "red", "R1", "R2", "R3", "nir", "R4", "swir1", "swir2"],
        )
        .map(add_indices)
    )

    secondary = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(aoi).filterDate(start_date, end_date)

    joined = ee.Join.inner().apply(
        primary=primary,
        secondary=secondary,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def merge_image_bands(join_result):
        return ee.Image(join_result.get("primary")).addBands(join_result.get("secondary"))

    return (
        ee.ImageCollection(joined.map(merge_image_bands))
        .map(mask_clouds(cloud_mask_probability))
        .sort("system:time_start")
    )


def process_sentinel2_index_variables(period, region):
    start_date, end_date = period
    sentinel2 = get_s2_sr_cloud_probability(region, start_date, end_date, scene_cloud_threshold, cloud_mask_probability)
    vars = {
        "ndvi": apply_region_mask(
            sentinel2.select("ndvi").median().rename(f"ndvi_{get_period_name(period)}").toFloat(), region
        ),
        "ndbi": apply_region_mask(
            sentinel2.select("ndbi").median().rename(f"ndbi_{get_period_name(period)}").toFloat(), region
        ),
        "ndwi": apply_region_mask(
            sentinel2.select("ndwi").median().rename(f"ndwi_{get_period_name(period)}").toFloat(), region
        ),
        "evi": apply_region_mask(
            sentinel2.select("evi").median().rename(f"evi_{get_period_name(period)}").toFloat(), region
        ),
    }

    return ee.Image([vars["ndvi"], vars["ndbi"], vars["ndwi"], vars["evi"]])


def process_sentinel2(period, region):
    start_date, end_date = period
    sentinel2 = get_s2_sr_cloud_probability(region, start_date, end_date, scene_cloud_threshold, cloud_mask_probability)

    vars = {
        "red": apply_region_mask(
            sentinel2.select("red").median().rename(f"red_{get_period_name(period)}").toFloat(), region
        ),
        "green": apply_region_mask(
            sentinel2.select("green").median().rename(f"green_{get_period_name(period)}").toFloat(), region
        ),
        "blue": apply_region_mask(
            sentinel2.select("blue").median().rename(f"blue_{get_period_name(period)}").toFloat(), region
        ),
        "R1": apply_region_mask(
            sentinel2.select("R1").median().rename(f"R1_{get_period_name(period)}").toFloat(), region
        ),
        "R2": apply_region_mask(
            sentinel2.select("R2").median().rename(f"R2_{get_period_name(period)}").toFloat(), region
        ),
        "R3": apply_region_mask(
            sentinel2.select("R3").median().rename(f"R3_{get_period_name(period)}").toFloat(), region
        ),
        "nir": apply_region_mask(
            sentinel2.select("nir").median().rename(f"nir_{get_period_name(period)}").toFloat(), region
        ),
        "R4": apply_region_mask(
            sentinel2.select("R4").median().rename(f"R4_{get_period_name(period)}").toFloat(), region
        ),
        "swir1": apply_region_mask(
            sentinel2.select("swir1").median().rename(f"swir1_{get_period_name(period)}").toFloat(), region
        ),
        "swir2": apply_region_mask(
            sentinel2.select("swir2").median().rename(f"swir2_{get_period_name(period)}").toFloat(), region
        ),
    }

    return ee.Image(
        [
            vars["red"],
            vars["green"],
            vars["blue"],
            vars["R1"],
            vars["R2"],
            vars["R3"],
            vars["nir"],
            vars["R4"],
            vars["swir1"],
            vars["swir2"],
        ]
    )


# %% [markdown] id="PlKmP-1EWo4X"
# ### Sentinel-1 Processing
#
#

# %% [markdown] id="vYDgNz7orgm1"
# This section processes Sentinel-1 radar data:
# - Applies radiometric slope correction to account for terrain effects.
# - Orbit propierties of the satelite are set to **"Descending"**
# - Generates two bands:
#   - **VV Polarization:** Vertical transmit and receive.
#   - **VH Polarization:** Vertical transmit, horizontal receive.
#
# The `process_sentinel1` function computes the mean value for each band, clips it to the region of interest, and names it based on the period.
#
# The `slope_correction` function adjusts Sentinel-1 radar images for terrain-induced distortions. It uses a digital elevation model (DEM) to calculate the terrain's slope and aspect, which are then used to correct for the angle of radar backscatter. The function applies a volume model correction to normalize radar signals and creates masks for layover (areas where slopes face the sensor, causing signal overlap) and shadow (areas blocked from radar signals). These corrections ensure the radar data reflects true ground conditions, reducing distortions caused by terrain. The processed bands (VV and VH polarizations) are averaged, clipped to the region of interest, and prepared for further analysis.


# %% id="I7DD8TVRBE8_"
def slope_correction(collection, elevation=None, model="volume", buffer=10):
    """
    Applies radiometric slope correction to a Sentinel-1 collection.

    Args:
        collection: ee.ImageCollection of Sentinel-1 images.
        elevation: ee.Image of DEM (optional, default is NASADEM).
        model: Correction model to apply ('volume' or 'surface').
        buffer: Buffer distance in meters for layover/shadow mask (default is 10).

    Returns:
        ee.ImageCollection with corrected images and additional bands.
    """
    if elevation is None:
        elevation = ee.Image("NASA/NASADEM_HGT/001")

    def _volumetric_model_SCF(theta_iRad, alpha_rRad):
        """
        Calculates the volumetric model SCF.
        """
        ninetyRad = ee.Image.constant(90).multiply(3.14159265359 / 180)
        nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denominator = (ninetyRad.subtract(theta_iRad)).tan()
        return nominator.divide(denominator)

    def _masking(alpha_rRad, theta_iRad, buffer):
        """
        Creates masks for layover and shadow.
        """
        layover = alpha_rRad.lt(theta_iRad).rename("layover")
        ninetyRad = ee.Image.constant(90).multiply(3.14159265359 / 180)
        shadow = alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename("shadow")

        if buffer > 0:
            layover = (
                layover.Not()
                .fastDistanceTransform(30)
                .sqrt()
                .multiply(ee.Image.pixelArea().sqrt())
                .gt(buffer)
                .rename("layover")
            )
            shadow = (
                shadow.Not()
                .fastDistanceTransform(30)
                .sqrt()
                .multiply(ee.Image.pixelArea().sqrt())
                .gt(buffer)
                .rename("shadow")
            )

        no_data_mask = layover.And(shadow).rename("no_data_mask")
        return layover.addBands(shadow).addBands(no_data_mask)

    def _correct(image):
        """
        Applies slope correction to a single image and adds layover and shadow masks.
        """
        theta_iRad = image.select("angle").multiply(3.14159265359 / 180)
        alpha_sRad = ee.Terrain.slope(elevation).multiply(3.14159265359 / 180)
        phi_sRad = ee.Terrain.aspect(elevation).multiply(3.14159265359 / 180)

        phi_iRad = ee.Image.constant(0).multiply(3.14159265359 / 180)  # Assuming flat incidence direction

        phi_rRad = phi_iRad.subtract(phi_sRad)
        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

        gamma0 = image.divide(10.0).pow(10).divide(theta_iRad.cos())

        if model == "volume":
            scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)
        else:
            raise ValueError("Only 'volume' model is supported in this implementation.")

        gamma0_flat = gamma0.divide(scf)
        gamma0_flatDB = ee.Image.constant(10).multiply(gamma0_flat.log10()).rename(image.bandNames())

        masks = _masking(alpha_rRad, theta_iRad, buffer)

        return gamma0_flatDB.addBands(masks).copyProperties(image, image.propertyNames())

    return collection.map(_correct)


def process_sentinel1_band(period, region, band_name):
    """
    Processes a specific Sentinel-1 band, applying slope correction and clipping it to the region.

    Args:
        period: Tuple with the start and end dates of the period (start_date, end_date).
        region: Region of interest as an `ee.FeatureCollection`.
        band_name: Name of the band ('VV' or 'VH').

    Returns:
        Processed and clipped image of the selected band, or None if no valid data is available.
    """
    start_date, end_date = period

    # Filter the Sentinel-1 collection
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", band_name))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    # Check if the collection is empty
    collection_size = collection.size().getInfo()
    if collection_size == 0:
        print(f"No images found for band {band_name} in period {start_date} to {end_date}")
        return None

    # Apply slope correction
    corrected = slope_correction(collection)

    # Select the corrected band, calculate the mean, and clip it to the region
    band_mean = corrected.select(band_name).mean()
    band_clipped = band_mean.clip(region).rename(f"{band_name}_{get_period_name(period)}").toFloat()

    return band_clipped


def process_sentinel1(period, region):
    """
    Processes Sentinel-1 for VV and VH bands, applying slope correction and clipping them to the region.

    Args:
        period: Tuple with the start and end dates of the period (start_date, end_date).
        region: Region of interest as an `ee.FeatureCollection`.

    Returns:
        Image with processed VV and VH bands, or None if no valid data is available.
    """
    vh = process_sentinel1_band(period, region, "VH")
    vv = process_sentinel1_band(period, region, "VV")

    # Check if any band is None
    if vh is None or vv is None:
        print(f"Skipping Sentinel-1 processing for period {get_period_name(period)} due to missing data.")
        return None

    return ee.Image([vh, vv])


# %% [markdown] id="aaJ2IzR8WvIM"
# ### EVI Statistics
#
#

# %% [markdown] id="VyoJ-Mrerkpz"
# Calculates summary statistics (sum, min, max, standard deviation) for the Enhanced Vegetation Index (EVI) over the region of interest for each period.


# %% id="DpeDTsnX-fQc"
def calculate_evi_statistics_by_period(period, region):
    start_date, end_date = period
    sentinel2 = get_s2_sr_cloud_probability(region, start_date, end_date, scene_cloud_threshold, cloud_mask_probability)

    sentinel2_evi = sentinel2.map(
        lambda img: img.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": img.select("nir"),
                "RED": img.select("red"),
                "BLUE": img.select("blue"),
            },
        ).rename("evi")
    )

    vars = {
        "evi_min": apply_region_mask(
            sentinel2_evi.reduce(ee.Reducer.min()).rename(f"evi_min_{get_period_name(period)}").toFloat(), region
        ),
        "evi_max": apply_region_mask(
            sentinel2_evi.reduce(ee.Reducer.max()).rename(f"evi_max_{get_period_name(period)}").toFloat(), region
        ),
        "evi_sd": apply_region_mask(
            sentinel2_evi.reduce(ee.Reducer.stdDev()).rename(f"evi_sd_{get_period_name(period)}").toFloat(), region
        ),
    }

    return ee.Image([vars["evi_min"], vars["evi_max"], vars["evi_sd"]])


# %% [markdown] id="QkUo6gh3Wx2b"
# ## Adding Data to Collections
#
#

# %% [markdown] id="E-8tsbtWc8Qy"
# For each period, the script:
# - Processes Sentinel-2 data and adds it to `sentinel2_collections`.
# - Processes Sentinel-2 indices and adds them to `sentinel2_index_variables_collections`.
# - Processes Sentinel-1 data and adds it to `sentinel1_collections`.
# - Calculates EVI statistics and adds them to `evi_statistics`.

# %% id="4Woo2k8hBI9Q"
sentinel2_collections = []
sentinel2_index_variables_collections = []
sentinel1_collections = []
evi_statistics = []

for period in periods:
    sentinel2_collections.append(apply_region_mask(process_sentinel2(period, region_buffered), region))
    sentinel2_index_variables_collections.append(
        apply_region_mask(process_sentinel2_index_variables(period, region_buffered), region)
    )
    sentinel1_collections.append(apply_region_mask(process_sentinel1(period, region_buffered), region))
    evi_statistics.append(apply_region_mask(calculate_evi_statistics_by_period(period, region_buffered), region))


# %% [markdown] id="8CvryX1vdAcC"
# Exports the processed data to Google Drive in batches:
# 1. **Batch 1:** NASADEM Variables.
# 2. **Batch 2:** TPI, Landform an Climate Variables.
# 3. **Batch 3:** Sentinel-1, Sentinel-2 with EVI statistics and Index Variables for Period 1.
# 4. **Batch 4:** Sentinel-1, Sentinel-2 with EVI statistics and Index Variables for Period 2.
# 5. **Batch 5:** Sentinel-1, Sentinel-2 with EVI statistics and Index Variables for Period 3.
# Each batch waits for the previous one to complete before starting.

# %% [markdown] id="8rV1DoYO7gWO"
# ## Export Data to Drive


# %% id="-O6X7UiTVIZq"
# Define a function to export images and wait for the tasks to complete
def export_batch(batch_tasks, defined_crs):
    """
    Export a batch of tasks and wait for them to complete before proceeding.

    Args:
        batch_tasks: List of tuples, each containing (image, description, file_name, region).
    """
    active_tasks = []

    for image, description, file_name, region in batch_tasks:
        print(f"Starting export of {description} to Google Drive...")
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=drive_folder,
            fileNamePrefix=file_name,
            region=region.geometry(),
            scale=30,
            crs=defined_crs,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
            shardSize=1024,
            formatOptions={"noData": -9999},
        )
        task.start()
        active_tasks.append((task, description))

    # Monitor tasks until all are complete
    for task, description in active_tasks:
        print(f"Waiting for task {description} to complete...")
        while task.active():
            time.sleep(60)  # Check every minute
        status = task.status()
        if status["state"] == "COMPLETED":
            print(f"Task {description} completed successfully.")
        else:
            print(f"Task {description} failed: {status.get('error_message', 'No error message provided')}.")


# Define the 5 batches for exports
batch_1 = [
    (
        nasadem_variables,
        "NASADEM Variables",
        "nasadem_variables",
        region,
    )  # 17 minutes aprox (+ 15 minutes to change crs)
]

batch_2 = [
    (tpi, "TPI", "tpi", region),  # 05 minutes aprox
    (landform, "Landform", "landform", region),  # 03 minutes aprox
    (climate_variables, "Climate Variables", "climate_variables", region),  # 06 minutes aprox
]

batch_3 = [
    (sentinel2_collections[0], "Sentinel-2 P1", "sentinel-2_P1", region),  # 27 minutes aprox
    (
        sentinel2_index_variables_collections[0],
        "Sentinel-2 Index Variables P1",
        "sentinel-2-index-variables_P1",
        region,
    ),  # 12 minutes aprox
    (evi_statistics[0], "EVI Stats P1", "evi-stats_P1", region),  # 11 minutes aprox
    (sentinel1_collections[0], "Sentinel-1 P1", "sentinel-1_P1", region),  # 11 minutes aprox
]

batch_4 = [
    (sentinel2_collections[1], "Sentinel-2 P2", "sentinel-2_P2", region),  # 20 minutes aprox
    (
        sentinel2_index_variables_collections[1],
        "Sentinel-2 Index Variables P2",
        "sentinel-2-index-variables_P2",
        region,
    ),  # 11 minutes aprox
    (evi_statistics[1], "EVI Stats P2", "evi-stats_P2", region),  # 07 minutes aprox
    (sentinel1_collections[1], "Sentinel-1 P2", "sentinel-1_P2", region),  # 13 minutes aprox
]

batch_5 = [
    (sentinel2_collections[2], "Sentinel-2 P3", "sentinel-2_P3", region),  # 17 minutes aprox
    (
        sentinel2_index_variables_collections[2],
        "Sentinel-2 Index Variables P3",
        "sentinel-2-index-variables_P3",
        region,
    ),  # 16 minutes aprox
    (evi_statistics[2], "EVI Stats P3", "evi-stats_P3", region),  # 11 minutes aprox
    (sentinel1_collections[2], "Sentinel-1 P3", "sentinel-1_P3", region),  # 11 minutes aprox
]  # Total: 213 minutes aprox (5,8 gb)

# %% [markdown] id="9YCCzN85Zhy3"
# ### Download Batches

# %% id="tfCu6YuAXzXf"
print("Starting Batch 1")
export_batch(batch_1, "EPSG:4326")  # Do not change this CRS

# %% id="eOq0BWEnXzvc"
print("Starting Batch 2")
export_batch(batch_2, export_crs)

# %% id="8mKCu3wXXz0l"
print("Starting Batch 3")
export_batch(batch_3, export_crs)

# %% id="QF4p6w7fXryk"
print("Starting Batch 4")
export_batch(batch_4, export_crs)

# %% id="068EOT8ZVQdg"
print("Starting Batch 5")
export_batch(batch_5, export_crs)

# %% [markdown] id="-idQOyaeJGv7"
# #### Convert NASADEM variables to desired CRS

# %% id="0w8Wgu_pJDyU"
# Path to the original NASADEM file
nasadem_tif = f"/content/drive/MyDrive/{drive_folder}/nasadem_variables.tif"  # Change this path as needed

# Path to a reference raster (any correctly aligned file)
reference_tif = (
    f"/content/drive/MyDrive/{drive_folder}/evi-stats_P2.tif"  # Change this path to a correctly aligned file
)

# Temporary file for reprojection
temp_nasadem = nasadem_tif.replace(".tif", "_temp.tif")

try:
    # Open the reference raster to get the correct extent, width, height, and transform
    with rasterio.open(reference_tif) as ref_src:
        ref_transform = ref_src.transform
        ref_width = ref_src.width
        ref_height = ref_src.height
        ref_crs = ref_src.crs
        ref_bounds = ref_src.bounds

    # Open the NASADEM raster
    with rasterio.open(nasadem_tif) as src:
        src_crs = src.crs

        # Retrieve original band names
        band_names = src.descriptions if src.descriptions else [f"Band {i}" for i in range(1, src.count + 1)]

        # Update metadata with the correct width, height, transform, and CRS
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": ref_crs,
                "transform": ref_transform,
                "width": ref_width,
                "height": ref_height,
                "nodata": -9999,
                "compress": "DEFLATE",
                "predictor": 2,
                "zlevel": 9,
            }
        )

        # Create the reprojected NASADEM file
        with rasterio.open(temp_nasadem, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                # Create an empty array for the reprojected data
                dst_array = np.empty((ref_height, ref_width), dtype=src.dtypes[i - 1])

                # Perform the reprojection
                reproject(
                    source=rasterio.band(src, i),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )

                # Write the reprojected data to the output file
                dst.write(dst_array, i)

                # Restore band names
                dst.set_band_description(i, band_names[i - 1])

    # Replace the original NASADEM file with the corrected one
    os.remove(nasadem_tif)  # Remove the original file
    os.rename(temp_nasadem, nasadem_tif)  # Rename the temporary file

    print(f"NASADEM file reprojected and aligned successfully: {nasadem_tif}")

except Exception as e:
    print(f"Error reprojecting NASADEM file: {e}")
