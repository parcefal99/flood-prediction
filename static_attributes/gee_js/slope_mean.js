var basinAssets = [
    // add your paths to the basins 
  ];
  
var meritDEM = ee.Image('MERIT/DEM/v1_0_3');
  
var slope = ee.Terrain.slope(meritDEM);
  
function calculateMeanSlope(basinAsset) {
  var basin = ee.FeatureCollection(basinAsset);
  var basinGeometry = basin.geometry();
  var clippedSlope = slope.clip(basinGeometry);
  var meanSlope = clippedSlope.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: basinGeometry,
    scale: 30,  // MERIT DEM has a resolution of 30 meters
    maxPixels: 1e9
  });
    
  return ee.Feature(null, {
    'basin_id': basinAsset,
    'slope_mean': meanSlope.get('slope')
  });
}
  
  var slopeResults = basinAssets.map(calculateMeanSlope);
  
  var slopeFeatureCollection = ee.FeatureCollection(slopeResults);
  
  // Export the results to Google Drive
  Export.table.toDrive({
    collection: slopeFeatureCollection,
    description: 'slope_mean',
    fileFormat: 'CSV',
    selectors: ['basin_id', 'slope_mean']
  });