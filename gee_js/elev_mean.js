var basinAssets = [
  // add your paths to all basins     
];
  
// load the dataset   
var meritDEM = ee.Image('MERIT/DEM/v1_0_3');
  
// function to calculate mean elevation for a basin
function calculateMeanElevation(basinAsset) {
  var basin = ee.FeatureCollection(basinAsset);
  var basinGeometry = basin.geometry();
  var clippedDEM = meritDEM.clip(basinGeometry);
    
  var meanElevation = clippedDEM.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: basinGeometry,
    scale: 30,  
    maxPixels: 1e9
  });
    
  return ee.Feature(null, {
    'basin_id': basinAsset,
    'elev_mean': meanElevation.get('dem')
  });
}
  
var elevationResults = basinAssets.map(calculateMeanElevation);
  
var elevationFeatureCollection = ee.FeatureCollection(elevationResults);
  
// Export the results to Google Drive/Cloud as .csv file 
Export.table.toDrive({
  collection: elevationFeatureCollection,
  description: 'MeanElevation',
  fileFormat: 'CSV',
  selectors: ['basin_id', 'elev_mean']
});