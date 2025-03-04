var basinAssets = [
    // add your paths to basins 
  ];
  
// function to calculate the area for a catchment
function calculateArea(basinAsset) {
  var basin = ee.FeatureCollection(basinAsset);
  var basinGeometry = basin.geometry();
  
  var area = basinGeometry.area().divide(1e6); // Convert to square kilometers
  
  return ee.Feature(null, {
    'basin_id': catchmentAsset,
    'area_gages2': area
  });
}
  
// Map the calculateArea function over the list of catchment assets
var areaResults = basinAssets.map(calculateArea);
  
// Create a FeatureCollection from the results
var areaFeatureCollection = ee.FeatureCollection(areaResults);
  
// Export the results to Google Drive/Cloud as .csv file
Export.table.toDrive({
  collection: areaFeatureCollection,
  description: 'area_gages2',
  fileFormat: 'CSV',
  selectors: ['basin_id', 'area_gages2']
});
  
  
  