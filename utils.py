
import ee
from typing import Dict, List, Tuple


def extractROI(asset_folder: str)->List[ee.Geometry]:
    
    """ Extraction of coordinates for a list of basis 
    - With given asset folder with shapefiles in zip format, the function 
        extracts geometry for each basin

    Returns:
        List[ee.Geometry]: boundaries of selected basins 
    """

    assets = ee.data.listAssets({'parent': asset_folder})['assets']
    assets_ids = [asset['id'] for asset in assets]
    
    geometryLst = list()
    basin_names = list()
    
    for featureCollection in assets_ids:
        # Create Feature Collection and extract geometry (coordinates's list)
        feature = ee.FeatureCollection(featureCollection)
        basinGeometry = feature.first().geometry()
        
        geometryLst.append(basinGeometry)
        basin_names.append(featureCollection.split('/')[-1])
    
    return geometryLst, basin_names