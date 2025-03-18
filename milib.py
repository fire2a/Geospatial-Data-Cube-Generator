
# %% id="mZbDmAKo76a1"
def apply_region_mask(image, region, nodata_value=-9999):
    """
    Clips an image to the region of interest (ROI) and sets NoData values outside the ROI.

    Args:
        image: The Earth Engine image to clip and mask.
        region: The region of interest to clip the image to.
        nodata_value: The value to assign to NoData pixels.

    Returns:
        An Earth Engine image clipped to the ROI with NoData values assigned.
    """
    clipped = image.clip(region)
    masked = clipped.updateMask(clipped.mask()).unmask(nodata_value)
    return masked
