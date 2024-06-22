
import ee
from typing import Dict, List, Tuple
from zipfile import ZipFile
from google.cloud import storage
import time
import os 


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



DIR_PATH = './shapefiles'
PROJECT_ID = 'virtual-rarity-426212-p6'


# create zipfiles for shapefiles 

def upload_shapefiles(dir_path):
    mode = 0o755
    zipped_dir = os.path.join(dir_path, "zipped_files")
    
    if not os.path.exists(zipped_dir):
        os.mkdir(zipped_dir, mode)
    
    for dirpath, dirs, files in os.walk(dir_path):
        if os.path.basename(dirpath) == "zipped_files":
            continue
        
        zipped_folder = os.path.basename(dirpath) + ".zip"
        zipped_path = os.path.join(zipped_dir, zipped_folder)
        
        with ZipFile(zipped_path, 'w') as zipf:
            for file in files:
                file_path = os.path.join(dirpath, file)
                zipf.write(file_path, os.path.relpath(file_path, dirpath))
        # print(f"Zipped {dirpath} to {zipped_path}")




def upload_to_gee(gcs_uri, asset_id):
    
    metadata = {
        'id': asset_id,
        "sources": [{
            "uris": [ gcs_uri]}], 
    }
    
    
    task_id = ee.data.newTaskId()[0]
    ee.data.startTableIngestion(task_id, metadata)
    
    print(f'Ingestion task started with ID: {task_id}')
    print(f'You can check the task status here: https://code.earthengine.google.com/tasks')


# Upload zipfiles from the bucket in GCS to GEE
bucket_name = 'zipped-shapefiles'

for zip_file in os.listdir('./shapefiles/zipped_files'):
    if zip_file.endswith('.zip'):
        gcs_uri = f'gs://{bucket_name}/{zip_file}'
        asset_id = f'projects/virtual-rarity-426212-p6/assets/shapefiles/{os.path.splitext(zip_file)[0]}'
        upload_to_gee(gcs_uri, asset_id)
        
        time.sleep(5)