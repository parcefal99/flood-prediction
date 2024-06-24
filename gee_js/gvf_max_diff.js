var basinAssets = [
  // add the paths to the basins
];

// adjust if needed 
var startDate = '2002-02-18'; // according to the MOD13Q1 
var endDate = '2023-12-31';

// function to calculate GVF statistics for a single basin
function calculateGVFStats(basinAsset) {
  var basin = ee.FeatureCollection(basinAsset).first();
  var basinGeometry = basin.geometry();

  var ndviCollection = ee.ImageCollection('MODIS/006/MOD13Q1')
                        .select('NDVI')
                        .filterBounds(basinGeometry)
                        .filterDate(startDate, endDate);  

  // function to convert NDVI to GVF
  var ndviToGvf = function(image) {
    var ndviMin = -0.2; // according to the MOD13Q1 documentation
    var ndviMax = 1;  // according to the MOD13Q1 documentation 
    var gvf = image.expression(
      '((NDVI - ndviMin) / (ndviMax - ndviMin))', {
        'NDVI': image.select('NDVI').multiply(0.0001),  // scaling is according to the MOD13Q1 documentation
        'ndviMin': ndviMin,
        'ndviMax': ndviMax
      }).rename('GVF');
    
    return image.addBands(gvf);
  };

  // map NDVI to GVF across the ImageCollection
  var gvfCollection = ndviCollection.map(ndviToGvf).select('GVF');

  // calculate mean monthly GVF
  var monthlyGvf = ee.ImageCollection.fromImages(
    ee.List.sequence(1, 12).map(function(month) {
      var monthlyMeanGvf = gvfCollection.filter(ee.Filter.calendarRange(month, month, 'month'))
                                        .mean()
                                        .set('month', month);
      return monthlyMeanGvf;
    })
  );

  // calculate mean GVF in the whole basin 
  var monthlyMeanGvfInBasin = monthlyGvf.map(function(image) {
    var meanGvf = image.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: basinGeometry,
      scale: 250,  // according to the MOD13Q1 documentation
      maxPixels: 1e9
    }).get('GVF');
    return ee.Feature(null, {'month': image.get('month'), 'meanGvf': meanGvf});
  });

  var monthlyMeanGvfFeatures = ee.FeatureCollection(monthlyMeanGvfInBasin);

  // find max and min mean monthly GVF values
  var maxMonthlyMeanGvfFeature = monthlyMeanGvfFeatures.sort('meanGvf', false).first();
  var minMonthlyMeanGvfFeature = monthlyMeanGvfFeatures.sort('meanGvf', true).first();

  // extract max and min GVF values
  var gvf_max = ee.Number(maxMonthlyMeanGvfFeature.get('meanGvf'));
  var gvf_min = ee.Number(minMonthlyMeanGvfFeature.get('meanGvf'));

  // return results as a Feature
  return ee.Feature(null, {
    'basin_id': ee.String(basinAsset).slice(-5),  // Extract basin ID from asset string
    'gvf_max': gvf_max,
    'gvf_diff': ee.Number(gvf_max).subtract( gvf_min)
  });
}

// Map over all basin assets and calculate GVF stats for each basin
var gvfStatsResults = basinAssets.map(calculateGVFStats);

// Create a FeatureCollection from the results
var gvfStatsCollection = ee.FeatureCollection(gvfStatsResults);

// print the results to the console
print('GVF Statistics Results:', gvfStatsCollection);

// Export the results to Google Drive/Cloud as CSV
Export.table.toDrive({
  collection: gvfStatsCollection,
  description: 'gvf_max_diff',
  fileFormat: 'CSV',
  selectors: ['basin_id', 'gvf_max', 'gvf_diff']
});
