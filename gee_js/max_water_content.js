var basinAssets = [
    // add your paths
  ];
  
  
  var wcsat = ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/wcsat");
  // 2. Maximum water content of the soil. result: fraction between 0.4 and 0.6
  function calculate_max_water_content (basinAsset){
      var basin = ee.FeatureCollection(basinAsset);
      var basinGeometry = basin.geometry();
      // filter the Wcsat image collection to get the image for your region
      var wcsat_image = wcsat.first(); 
      // clip the Wcsat image to your basin
      var wcsat_clipped = wcsat_image.clip(basin);
      var wcsat_stats = wcsat_clipped.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: basinGeometry,
        scale: 250, 
        maxPixels: 1e9
      });
      wcsat_stats = ee.Number(wcsat_stats.get("b1")).divide(10000);
      return ee.Feature(null, {
      'Basin_ID': basinAsset,
      'Max_water_content': wcsat_stats
    });
    }
  
  var elevationResults = basinAssets.map(function(basinAsset) {
    print(calculate_max_water_content(basinAsset));
    return calculate_max_water_content(basinAsset);
  });
  
  var elevationFeatureCollection = ee.FeatureCollection(elevationResults);
  
  // Export the results to Google Drive
  Export.table.toDrive({
    collection: elevationFeatureCollection,
    description: 'maxWaterContent',
    fileFormat: 'CSV',
    selectors: ['Basin_ID', 'Max_water_content']
  });
  