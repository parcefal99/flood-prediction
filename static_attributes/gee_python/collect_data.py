
import ee
import yaml
import pandas as pd
from basin_static_attributes import PrcpSeasonality, BasinMetrics, \
                                    Vegetation, SoilMetrics 




def main():
    
    with open("./config.yaml") as file:
        cfg = yaml.safe_load(file)
    
    ee.Authenticate()
    ee.Initialize(project="virtual-rarity-426212-p6") # name of your project (replace)!
    
    soil_data = SoilMetrics(cfg['asset_folder'])
    basin_metrics = BasinMetrics(cfg['asset_folder'])
    vegi_data = Vegetation(cfg['start_date'], cfg['end_date'], cfg['lai_band'],
                           cfg['forest_band'], cfg['green_band'], cfg['asset_folder'])
    
    # seasonality_data = PrcpSeasonality(cfg['meteo_files'])
    
    soil_data = soil_data.computeAllSoilStats()
    basin_metrics = basin_metrics.computeAllElevationSlopeStats()
    vegi_data = vegi_data.computeAllStats()
    
    cntd_data = pd.concat(objs=[basin_metrics, vegi_data, soil_data], axis=1)
    cntd_data.to_csv("./static_data.csv")
    
    # print(vegi_data)
    

if __name__ == "__main__":
    main()