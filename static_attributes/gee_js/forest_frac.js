var basinAssets = [
    // add your paths to the basins
  ];
  
var worldCover = ee.ImageCollection('ESA/WorldCover/v100').first();
var landCover = worldCover.select('Map');
  
// Define the forest class for Kazakhstani basins
var forestClass = 10;
  
// Create a binary image where forest pixels are 1 and non-forest pixels are 0
var forestMask = landCover.eq(forestClass);
  
// Function to calculate forest fraction
function calculateForestFraction(basinAsset) {
  var basin = ee.FeatureCollection(basinAsset);
  var geometry = basin.geometry();
    
  var forestArea = forestMask.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: geometry,
    scale: 10,  // WorldCover dataset has a 10m resolution
    maxPixels: 1e9
  }).get('Map');
  
  // Calculate the total area of the catchment 
  var catchmentArea = geometry.area().divide(1e6); // Convert from m² to km²
  
  // Calculate the forest fraction
  forestArea = ee.Number(forestArea).divide(1e6); // Convert from m² to km²
  var forestFraction = forestArea.divide(catchmentArea);
  
  // Set the properties to be exported
  return ee.Feature(null, {
    'basin_id': catchmentAsset.slice(-5),
    'forest_frac': forestFraction
  });
}
  
// Apply the function to each catchment
var results = basinAssets.map(calculateForestFraction);
  
// Print to the console of Google Earth Enginge
print(results);
  
// Export the results to a CSV file
Export.table.toDrive({
  collection: ee.FeatureCollection(results),
  description: 'ForestFractionResults',
  fileFormat: 'CSV',
  selectors: ['basin_id', 'forest_frac']
});
  