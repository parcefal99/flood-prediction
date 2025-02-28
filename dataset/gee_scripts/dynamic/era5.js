var startDate = '2000-01-01';
var endDate = '2022-12-31';

var dataset = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                .select([
                  'total_precipitation_sum',
                  'temperature_2m',
                  'temperature_2m_min', 
                  'temperature_2m_max',
                  'dewpoint_temperature_2m',
                  'u_component_of_wind_10m',
                  'v_component_of_wind_10m',
                  'surface_net_solar_radiation_sum'
                ])
                .filterDate(startDate, endDate);


var basinAssets = [
  'projects/[project-name]/assets/shapefiles/11001',
];


function extractAttributes(asset) {
    var basin = ee.FeatureCollection(asset);
    var basinGeometry = basin.geometry();
  

    var dailyAttributes = dataset.map(function(image) {

        var stats = image.reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: basinGeometry,
          bestEffort: true
        });
        

        return ee.Feature(null, {
          'basin_id': asset.slice(-5), 
          'date': image.date().format('YYYY-MM-dd'),
          'prcp': stats.get('total_precipitation_sum'),
          'temp_mean': stats.get('temperature_2m'),
          'temp_min': stats.get('temperature_2m_min'),
          'temp_max': stats.get('temperature_2m_max'),
          'dew_mean': stats.get('dewpoint_temperature_2m'),
          'u_comp_wind': stats.get('u_component_of_wind_10m'),
          'v_comp_wind': stats.get('v_component_of_wind_10m'),
          'srad_joules': stats.get('surface_net_solar_radiation_sum')
        });
    });
  
    var dailyAttributesFC = ee.FeatureCollection(dailyAttributes);
  
    Export.table.toDrive({
      collection: dailyAttributesFC,
      description: 'ERA5_' + asset.slice(-5), 
      fileFormat: 'CSV',
      folder: 'my-gee/ERA5_prcp',
      selectors: ['basin_id', 'date', 'prcp', 'temp_mean', 'temp_min', 'temp_max', 'dew_mean', 'u_comp_wind', 'v_comp_wind', 'srad_joules']
    });
}


basinAssets.forEach(extractAttributes);
