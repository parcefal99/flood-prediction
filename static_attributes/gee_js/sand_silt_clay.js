var basinAssets = [
    // add your paths
  ];
  
  var isric_silt = ee.Image("projects/soilgrids-isric/silt_mean");
  var isric_sand = ee.Image("projects/soilgrids-isric/sand_mean");
  var isric_clay = ee.Image("projects/soilgrids-isric/clay_mean");
  // 3. Fraction of sand in the soil. %
  // 4. Fraction of silt in the soil. %
  // 5. Fraction of clay in the soil. %
  function calculate_soil_attributes(basinAsset){
    var basin = ee.FeatureCollection(basinAsset);
    var basinGeometry = basin.geometry();
    // clip SoilGrids data to my basin
    var sand_basin = isric_sand.clip(basin);
    var silt_basin = isric_silt.clip(basin);
    var clay_basin = isric_clay.clip(basin);
    // calculate the percentage
    var total_percent = silt_basin.add(sand_basin).add(clay_basin);
    var sand_percent = sand_basin.divide(total_percent).multiply(100);
    var silt_percent = silt_basin.divide(total_percent).multiply(100);
    var clay_percent = clay_basin.divide(total_percent).multiply(100);
    // mean values (cm)
    var sand_mean = sand_percent.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: basinGeometry,
      scale: 250
    })
    var silt_mean = silt_percent.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: basinGeometry,
      scale: 250
    }) 
    var clay_mean = clay_percent.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: basinGeometry,
      scale: 250
    })
    // final mean value for each
    var sand_mean_value = (ee.Number(sand_mean.get('sand_0-5cm_mean'))
    .add(ee.Number(sand_mean.get('sand_100-200cm_mean')))
    .add(ee.Number(sand_mean.get('sand_15-30cm_mean')))
    .add(ee.Number(sand_mean.get('sand_30-60cm_mean')))
    .add(ee.Number(sand_mean.get('sand_5-15cm_mean')))
    .add(ee.Number(sand_mean.get('sand_60-100cm_mean')))
    .divide(6));


    var silt_mean_value = (ee.Number(silt_mean.get('silt_0-5cm_mean'))
      .add(ee.Number(silt_mean.get('silt_100-200cm_mean')))
      .add(ee.Number(silt_mean.get('silt_15-30cm_mean')))
      .add(ee.Number(silt_mean.get('silt_30-60cm_mean')))
      .add(ee.Number(silt_mean.get('silt_5-15cm_mean')))
      .add(ee.Number(silt_mean.get('silt_60-100cm_mean')))
      .divide(6));
    
    var clay_mean_value = (ee.Number(clay_mean.get('clay_0-5cm_mean'))
      .add(ee.Number(clay_mean.get('clay_100-200cm_mean')))
      .add(ee.Number(clay_mean.get('clay_15-30cm_mean')))
      .add(ee.Number(clay_mean.get('clay_30-60cm_mean')))
      .add(ee.Number(clay_mean.get('clay_5-15cm_mean')))
      .add(ee.Number(clay_mean.get('clay_60-100cm_mean')))
      .divide(6));
      
    return ee.Feature(null, {
      'basin_id': basinAsset,
      'sand_frac': sand_mean_value,
      'silt_frac': silt_mean_value,
      'clay_frac': clay_mean_value
    });
  }
  
  var elevationResults = basinAssets.map(function(basinAsset) {
    return calculate_soil_attributes(basinAsset);
  });
  
  var elevationFeatureCollection = ee.FeatureCollection(elevationResults);
  
  // Export the results to Google Drive
  Export.table.toDrive({
    collection: elevationFeatureCollection,
    description: 'SandSiltClayFraction',
    fileFormat: 'CSV',
    selectors: ['Basin_ID', 'sand_frac', 'silt_frac', 'clay_frac']
  });
