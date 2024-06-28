var basinAssets = [
    // add your paths to all basins
  ];
  
  var startDate = ee.Date('2000-01-01');
  var endDate = ee.Date('2023-12-31');
  
  // Load the MODIS/MOD16A2 dataset
  var modisPET = ee.ImageCollection('MODIS/006/MOD16A2')
                  .filterDate(startDate, endDate)
                  .select('PET');
  
  // Function to calculate mean daily potential evapotranspiration for a basin
  function calculateMeanPET(basinAsset) {
    var basin = ee.FeatureCollection(basinAsset);
    var basinGeometry = basin.geometry();
    
    // Sum PET over the time period and then average per day
    var totalPET = modisPET.sum().clip(basinGeometry);
    var totalDays = endDate.difference(startDate, 'days');
    var meanDailyPET = totalPET.divide(totalDays); // Convert from total to daily
    
    var meanPET = meanDailyPET.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: basinGeometry,
      scale: 1000,
      maxPixels: 1e9
    });
    
    return ee.Feature(null, {
      'basin_id': basinAsset,
      'pet_mean': meanPET.get('PET')
    });
  }
  
  var petResults = basinAssets.map(calculateMeanPET);
  
  var petFeatureCollection = ee.FeatureCollection(petResults);
  
  // Export the results to Google Drive
  Export.table.toDrive({
    collection: petFeatureCollection,
    description: 'pet_mean',
    fileFormat: 'CSV',
    selectors: ['basin_id', 'pet_mean']
  });