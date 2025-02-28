var basinAssets = [
  // add your paths to all basins
];

var startDate = ee.Date('2002-07-04'); // according to the MCD15A3H
var endDate = ee.Date('2023-12-31');

var modisLAI = ee.ImageCollection('MODIS/006/MCD15A3H')
                .filterDate(startDate, endDate)
                .select('Lai');

function calculateLAIStats(basinAsset) {
  var basin = ee.FeatureCollection(basinAsset).first();
  var basinGeometry = basin.geometry();
  
  // Define a function to calculate monthly mean LAI
  var monthlyLai = ee.ImageCollection.fromImages(
    ee.List.sequence(1, 12).map(function(month) {
      var monthlyLai = modisLAI.filter(ee.Filter.calendarRange(month, month, 'month'))
                               .mean();
      return monthlyLai.set('month', month);
    })
  );

  var maxLai = monthlyLai.max().clip(basinGeometry).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: basinGeometry,
    scale: 500,  // MCD15A3H resolution is 500m
    maxPixels: 1e9
  }).get('Lai');

  var minLai = monthlyLai.min().clip(basinGeometry).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: basinGeometry,
    scale: 500,  // MCD15A3H resolution is 500m
    maxPixels: 1e9
  }).get('Lai');

  // since the dataset is 4-day composite --> divide by 4
  maxLai = ee.Number(maxLai).divide(4);
  minLai = ee.Number(minLai).divide(4);
  var laiDiff = ee.Number(maxLai).subtract(minLai);

  return ee.Feature(null, {
    'basin_id': basinAsset.slice(-5),
    'lai_max': maxLai,
    'lai_diff': laiDiff
  });
}

var laiStatsResults = ee.FeatureCollection(basinAssets.map(calculateLAIStats));

// Print to the console on GEE
print('LAI Statistics Results:', laiStatsResults);

// Export to Google Drive/Cloud as .csv file
Export.table.toDrive({
  collection: laiStatsResults,
  description: 'LAIStatistics',
  fileFormat: 'CSV',
  selectors: ['basin_id', 'lai_max', 'lai_diff']
});
