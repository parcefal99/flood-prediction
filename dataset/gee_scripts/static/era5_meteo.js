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
  'projects/ee-kd12358/assets/shapefiles/11001',
  'projects/ee-kd12358/assets/shapefiles/11063',
  'projects/ee-kd12358/assets/shapefiles/11068',
  'projects/ee-kd12358/assets/shapefiles/11077',
  'projects/ee-kd12358/assets/shapefiles/11090',
  'projects/ee-kd12358/assets/shapefiles/11094',
  'projects/ee-kd12358/assets/shapefiles/11108',
  'projects/ee-kd12358/assets/shapefiles/11117',
  'projects/ee-kd12358/assets/shapefiles/11124',
  'projects/ee-kd12358/assets/shapefiles/11126',
  'projects/ee-kd12358/assets/shapefiles/11129',
  'projects/ee-kd12358/assets/shapefiles/11131',
  'projects/ee-kd12358/assets/shapefiles/11137',
  'projects/ee-kd12358/assets/shapefiles/11139',
  'projects/ee-kd12358/assets/shapefiles/11143',
  'projects/ee-kd12358/assets/shapefiles/11146',
  'projects/ee-kd12358/assets/shapefiles/11147',
  'projects/ee-kd12358/assets/shapefiles/11157',
  'projects/ee-kd12358/assets/shapefiles/11160',
  'projects/ee-kd12358/assets/shapefiles/11163',
  'projects/ee-kd12358/assets/shapefiles/11164',
  'projects/ee-kd12358/assets/shapefiles/11170',
  'projects/ee-kd12358/assets/shapefiles/11187',
  'projects/ee-kd12358/assets/shapefiles/11188',
  'projects/ee-kd12358/assets/shapefiles/11189',
  'projects/ee-kd12358/assets/shapefiles/11199',
  'projects/ee-kd12358/assets/shapefiles/11207',
  'projects/ee-kd12358/assets/shapefiles/11219',
  'projects/ee-kd12358/assets/shapefiles/11233',
  'projects/ee-kd12358/assets/shapefiles/11242',
  'projects/ee-kd12358/assets/shapefiles/11272',
  'projects/ee-kd12358/assets/shapefiles/11275',
  'projects/ee-kd12358/assets/shapefiles/11291',
  'projects/ee-kd12358/assets/shapefiles/11293',
  'projects/ee-kd12358/assets/shapefiles/11395',
  'projects/ee-kd12358/assets/shapefiles/11397',
  'projects/ee-kd12358/assets/shapefiles/11421',
  'projects/ee-kd12358/assets/shapefiles/11424',
  'projects/ee-kd12358/assets/shapefiles/11432',
  'projects/ee-kd12358/assets/shapefiles/11433',
  'projects/ee-kd12358/assets/shapefiles/11453',
  'projects/ee-kd12358/assets/shapefiles/11461',
  'projects/ee-kd12358/assets/shapefiles/11468',
  'projects/ee-kd12358/assets/shapefiles/11469',
  'projects/ee-kd12358/assets/shapefiles/11661',
  'projects/ee-kd12358/assets/shapefiles/12001',
  'projects/ee-kd12358/assets/shapefiles/12002',
  'projects/ee-kd12358/assets/shapefiles/12008',
  'projects/ee-kd12358/assets/shapefiles/12029',
  'projects/ee-kd12358/assets/shapefiles/12031',
  'projects/ee-kd12358/assets/shapefiles/12032',
  'projects/ee-kd12358/assets/shapefiles/12072',
  'projects/ee-kd12358/assets/shapefiles/12075',
  'projects/ee-kd12358/assets/shapefiles/12564',
  'projects/ee-kd12358/assets/shapefiles/13002',
  'projects/ee-kd12358/assets/shapefiles/13005',
  'projects/ee-kd12358/assets/shapefiles/13016',
  'projects/ee-kd12358/assets/shapefiles/13029',
  'projects/ee-kd12358/assets/shapefiles/13035',
  'projects/ee-kd12358/assets/shapefiles/13038',
  'projects/ee-kd12358/assets/shapefiles/13048',
  'projects/ee-kd12358/assets/shapefiles/13061',
  'projects/ee-kd12358/assets/shapefiles/13064',
  'projects/ee-kd12358/assets/shapefiles/13090',
  'projects/ee-kd12358/assets/shapefiles/13091',
  'projects/ee-kd12358/assets/shapefiles/13095',
  'projects/ee-kd12358/assets/shapefiles/13105',
  'projects/ee-kd12358/assets/shapefiles/13115',
  'projects/ee-kd12358/assets/shapefiles/13128',
  'projects/ee-kd12358/assets/shapefiles/13142',
  'projects/ee-kd12358/assets/shapefiles/13148',
  'projects/ee-kd12358/assets/shapefiles/13198',
  'projects/ee-kd12358/assets/shapefiles/13201',
  'projects/ee-kd12358/assets/shapefiles/13221',
  'projects/ee-kd12358/assets/shapefiles/19010',
  'projects/ee-kd12358/assets/shapefiles/19013',
  'projects/ee-kd12358/assets/shapefiles/19021',
  'projects/ee-kd12358/assets/shapefiles/19022',
  'projects/ee-kd12358/assets/shapefiles/19033',
  'projects/ee-kd12358/assets/shapefiles/19034',
  'projects/ee-kd12358/assets/shapefiles/19130',
  'projects/ee-kd12358/assets/shapefiles/19180',
  'projects/ee-kd12358/assets/shapefiles/19195',
  'projects/ee-kd12358/assets/shapefiles/19196',
  'projects/ee-kd12358/assets/shapefiles/19205',
  'projects/ee-kd12358/assets/shapefiles/19208',
  'projects/ee-kd12358/assets/shapefiles/19211',
  'projects/ee-kd12358/assets/shapefiles/19218',
  'projects/ee-kd12358/assets/shapefiles/19220',
  'projects/ee-kd12358/assets/shapefiles/19229',
  'projects/ee-kd12358/assets/shapefiles/19239',
  'projects/ee-kd12358/assets/shapefiles/19240',
  'projects/ee-kd12358/assets/shapefiles/19243',
  'projects/ee-kd12358/assets/shapefiles/19246',
  'projects/ee-kd12358/assets/shapefiles/19247',
  'projects/ee-kd12358/assets/shapefiles/19255',
  'projects/ee-kd12358/assets/shapefiles/19257',
  'projects/ee-kd12358/assets/shapefiles/19289',
  'projects/ee-kd12358/assets/shapefiles/19300',
  'projects/ee-kd12358/assets/shapefiles/19301',
  'projects/ee-kd12358/assets/shapefiles/19302',
  'projects/ee-kd12358/assets/shapefiles/19462',
  'projects/ee-kd12358/assets/shapefiles/19463',
  'projects/ee-kd12358/assets/shapefiles/77818',
  'projects/ee-kd12358/assets/shapefiles/77819',
  'projects/ee-kd12358/assets/shapefiles/77895'
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
