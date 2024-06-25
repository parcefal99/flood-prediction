var basinAssets = [
    // add your paths to the basins
  ];
  
  var startDate = ee.Date('2000-01-01');
  var endDate = ee.Date('2023-12-31');
  
  // Load the CHIRPS dataset
  var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                .filterDate(startDate, endDate);
  
  // Function to calculate mean daily precipitation for a basin
  function calculateMeanPrecip(basinAsset) {
    var basin = ee.FeatureCollection(basinAsset);
    var basinGeometry = basin.geometry();

    var precip = chirps.mean().clip(basinGeometry); // Daily precipitation is already in mm
    var meanPrecip = precip.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: basinGeometry,
      scale: 1000,
      maxPixels: 1e9
    });
    
    return ee.Feature(null, {
      'basin_id': basinAsset,
      'p_mean': meanPrecip.get('precipitation')
    });
  }
  
  var precipResults = basinAssets.map(calculateMeanPrecip);
  
  var precipFeatureCollection = ee.FeatureCollection(precipResults);
  
  // Export the results to Google Drive
  Export.table.toDrive({
    collection: precipFeatureCollection,
    description: 'p_mean',
    fileFormat: 'CSV',
    selectors: ['basin_id', 'p_mean']
  });