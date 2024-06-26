
import pandas as pd 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from tqdm import tqdm
import os
import ee
from typing import Dict, List, Tuple
from tqdm import tqdm
from utils_gee import extractROI






##################### Precipitation Seasonality ########################################


class PrcpSeasonality():
    
    def __init__(self, dataset_fldr: str):
        
        self.meteo_fldr = dataset_fldr
        self.txt_files = self._uploadFiles()
        self.logs = pd.DataFrame()
        
    def _uploadFiles(self):
        
        txt_files = list()
        for path, dirs, files in os.walk(self.meteo_fldr):
            for file in files:
                if str(file).endswith(".txt"):
                    full_path = os.path.join(path, file)
                    txt_files.append(full_path)
                    
        return txt_files
    
    
    def prcp_model(self, t, mean_P, d_P, s_P):
        return mean_P*(1 + d_P*np.sin(2*np.pi*(t-s_P)/365))

    def temp_model(self, t, mean_T, delta_T, s_T):
        return mean_T + delta_T*np.sin(2*np.pi*(t-s_T)/365)
    
    
    def computeAllSeasonality(self, output_logs: bool=False)->pd.DataFrame:
        
        """
        Computes the precipitation seasonality for all basins in the dataset.

        This method iterates over all the text files in the dataset folder, computes the precipitation 
        seasonality for each basin using the `_computeOneSeasonality` method, and compiles the results 
        into a pandas DataFrame.

        Parameters:
        -----------
        output_logs : bool, optional
            If True, detailed logs including calculation results and graphs are saved for each basin.
            The default is False.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the basin names and their corresponding precipitation seasonality values.

        """
        
        
        results = []
        basin_names = []
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} - PrcpSeasonality]'
        
        for idx, txt_file in enumerate(tqdm(self.txt_files, bar_format=bar_format)):
            
            # if idx == 5:  # remove to obtain stats about all basins
            #     break
            
            basin_name = os.path.basename(txt_file).rstrip(".txt")
            
            prcp_seasonality = self._computeOneSeasonality(txt_file, basin_name, output_logs)
            results.append(prcp_seasonality)
            basin_names.append(basin_name)
            
            if len(self.txt_files ) == self.logs.shape[0]:
                self.logs.to_csv(os.path.join("./output_logs", "basin_info" + ".csv"))
                

        df_main = pd.DataFrame(data={"basin_name": basin_names, "p_seasonality": results})
        
        return df_main
        
        
    def _computeOneSeasonality(self, data_dir: str, basin_name: str, output_logs)->float:
        
        """
            Computes the precipitation seasonality for a single basin.  
            This method processes the meteorological data for a given basin, fits sinusoidal models to the 
            temperature and precipitation data, and calculates a seasonality metric based on the model parameters.  
            Parameters:
            -----------
            data_dir : str
                The directory path of the text file containing the meteorological data for the basin.
            basin_name : str
                The name of the basin.
            output_logs : bool
                If True, detailed logs including calculation results and graphs are saved for the basin.    
            Returns:
            --------
            float
                The calculated precipitation seasonality metric for the basin, rounded to 7 decimal places. 
            Notes:
            ------
            - The method reads the meteorological data, performs interpolation and rolling averages, and 
              truncates the data to a specific date range.
            - The method fits sinusoidal models to both temperature and precipitation data using curve fitting.
            - The method calculates the seasonality metric based on the fitted model parameters.
            - If `output_logs` is True, the method saves the results and graphs in the 'output_logs' directory.
            
        """
        
        ### Data preprocessing step
        
        data_csv = pd.read_csv(data_dir, sep=r'\s+', engine='python', parse_dates=['date'])
        data_csv['date'] = pd.to_datetime(data_csv['date'])
        data = data_csv.loc[:, ['date', 'prcp', 't_mean']]


        data.set_index(data['date'], inplace=True)
        data.drop(columns=['date'], axis=1, inplace=True)
        data['prcp'].interpolate(method='time', inplace=True)
        data['t_mean'].interpolate(method='time', inplace=True)

        data['mov_prcp'] = data['prcp'].rolling(window=30, min_periods=15).mean()
        data['mov_temp'] = data['t_mean'].rolling(window=10, min_periods=5).mean()

        truncated_data = data.truncate(before=pd.Timestamp('2000-01-15'),
                                        after=pd.Timestamp('2024-03-15'))
        truncated_data['month_day'] = truncated_data.index.strftime('%m-%d')
        mean_data = truncated_data.groupby('month_day').mean()
        # mean_data.drop(columns=['prcp', 't_mean'], inplace=True)
        mean_data.reset_index(inplace=True)  # this dataframe should be fed to plot_graphs() function!

        ### Optimization step for paramaters of equations 2 and 3
        
        t = mean_data.index.values
        T = mean_data['t_mean'].values
        P = mean_data['prcp'].values

        initial_guess_T = [np.mean(T), (np.max(T) - np.min(T)) / 2]
        initial_guess_P = [np.mean(P), (np.max(P) - np.min(P)) / (2 * np.mean(P))]
        best_fit_T = None
        best_fit_P = None
        min_residual_T = float('inf')
        min_residual_P = float('inf')

        for s_T in range(0, 365):
            try:
                # popt, pcov = curve_fit(func, xdata, ydata)
                params_T, _ = curve_fit(lambda t, T_mean, delta_T: self.temp_model(t, T_mean, delta_T, s_T), t, T, p0=initial_guess_T)
                residual_T = np.sum((T - self.temp_model(t, *params_T, s_T)) ** 2)
                if residual_T < min_residual_T:
                    min_residual_T = residual_T
                    best_fit_T = (params_T[0], params_T[1], s_T)
            except RuntimeError:
                pass
            
        for s_P in range(0, 365):
            try:
                params_P, _ = curve_fit(lambda t, P_mean, d_P: self.prcp_model(t, P_mean, d_P, s_P), t, P, p0=initial_guess_P)
                residual_P = np.sum((P - self.prcp_model(t, *params_P, s_P)) ** 2)
                if residual_P < min_residual_P:
                    min_residual_P = residual_P
                    best_fit_P = (params_P[0], params_P[1], s_P)
            except RuntimeError:
                pass


        T_mean, Delta_T, s_T = best_fit_T
        P_mean, d_P, s_P = best_fit_P

        # print(f"Temperature - Mean: {T_mean:.3f}, Amplitude: {Delta_T:.3f}, Phase Shift: {s_T:.3f}")
        # print(f"Precipitation - Mean: {P_mean*365:.3f}, Amplitude: {d_P:.3f}, Phase Shift: {s_P:.3f}")

        prcp_seasonality = d_P*np.sign(Delta_T)*np.cos(2*np.pi*(s_P-s_T)/365) # final scalar value
        
        optim_params = {"T_mean": T_mean, "Delta_T": Delta_T, "s_T": s_T,
                        "P_mean": P_mean, "d_P": d_P, "s_P": s_P}
        
        # Save calculations and graphs into the log folder
        # Save calculations and graphs into the log folder
        if output_logs:
            os.makedirs("./output_logs", exist_ok=True)
            log = pd.DataFrame(data={**optim_params, "p_seasonality": [prcp_seasonality]}, index=[basin_name])
            self.logs = pd.concat([self.logs, log], ignore_index=False)
            graph_path = os.path.join("./output_logs", basin_name + ".png")
            self._plot_graph(mean_data, graph_path, **optim_params)

        
        return round(prcp_seasonality, 7)



    def _plot_graph(self, data: pd.DataFrame, graph_path: str, T_mean: float, Delta_T: float, 
                    s_T: float, P_mean: float, d_P: float, s_P: float):
        """
        Creates two graphs comparing observed and modeled temperature and precipitation data.

        Parameters:
        -----------
        data : pd.DataFrame
            A DataFrame containing the observed data with columns 't_mean' for temperature and 'prcp' for precipitation.
        T_mean : float
            The mean temperature used in the temperature model.
        Delta_T : float
            The amplitude of the temperature variation used in the temperature model.
        s_T : float
            The phase shift of the temperature model.
        P_mean : float
            The mean precipitation used in the precipitation model.
        d_P : float
            The amplitude of the precipitation variation used in the precipitation model.
        s_P : float
            The phase shift of the precipitation model.

        Returns:
        --------
        None
            The function generates and displays two plots: one for temperature and one for precipitation.

        The first plot shows observed and modeled temperature data.
        The second plot shows observed and modeled precipitation data.
        """
        
        t = data.index.values
        
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle("Observed and Modeled Temperature and Precipitation Curves")
        fig.set_size_inches(13, 8)
        
        # Plot temperature data
        ax1.plot(t, data['t_mean'].values, markersize=2, label='Observed Temperature')
        ax1.plot(t, self.temp_model(t, T_mean, Delta_T, s_T), '-', label='Fitted Sinusoidal Model', color='red')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Data and Fitted Sinusoidal Model')
        ax1.legend()

        # Plot precipitation data
        ax2.plot(t, data['prcp'].values, markersize=2, label='Observed Precipitation')
        ax2.plot(t, self.prcp_model(t, P_mean, d_P, s_P), '-', label='Fitted Sinusoidal Model', color='red')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Precipitation (mm/d)')
        ax2.set_title('Precipitation Data and Fitted Sinusoidal Model')
        ax2.legend()

        # Format x-axis to show month names
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        fig.savefig(graph_path)
        plt.close(fig)
        # plt.show()



######################### Slope-Elevation Stats ##################################



class BasinMetrics():
    
    def __init__(self, asset_fldr: str):
        self.roi_lst, self.basin_names = extractROI(asset_fldr)
        self.terrain = ee.Image("USGS/SRTMGL1_003") # contains one image
        
    def computeBasinArea(self, roi: ee.Geometry)->float:
        area_km2 = round(roi.area().getInfo()/1e6, 7)
        return area_km2
    
        
    def _computeOneElevationSlope(self, roi: ee.Geometry)->Tuple[Dict, Dict]:
        
        
        """
        Compute mean elevation and slope statistics for a given region of interest (ROI) using Google Earth Engine.

        Parameters:
        - roi (ee.Geometry): Region of interest defined as an Earth Engine Geometry object.

        Returns:
        - Tuple[Dict, Dict]: Tuple containing dictionaries with elevation and slope statistics.
        The dictionaries include 'mean', 'min', and 'max' keys for both elevation and slope.

        Notes:
        - Uses the 'terrain' dataset to compute elevation and 'ee.Terrain.slope' to compute slope.
        - Statistics are computed at a scale of 30 meters and are based on mean, minimum, and maximum values.
        
        """
        
        
        elev_stats = self.terrain.reduceRegion(
            reducer = ee.Reducer.mean().combine(
                reducer2 = ee.Reducer.minMax(),
                sharedInputs = True
            ), 
            geometry = roi, 
            scale = 30,
            maxPixels = 1e9
        ).getInfo()
        
        slope = ee.Terrain.slope(self.terrain)
        
        slope_stats = slope.reduceRegion(
            reducer = ee.Reducer.mean().combine(
                reducer2 = ee.Reducer.minMax(),
                sharedInputs = True
            ),
            geometry = roi,
            scale = 30,
            maxPixels = 1e9
        ).getInfo()
        
       
        return slope_stats, elev_stats
        
    def computeAllElevationSlopeStats(self):
        
        
        table = {"area_gages2": [], "elev_mean": [], "slope_mean": []}
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} - BasinMetrics]'
        
        for idx, roi in enumerate(tqdm(self.roi_lst, bar_format=bar_format)):
            
            # if idx==5:
            #     break
            
            area_gages2 = self.computeBasinArea(roi)
            slope_stats, elev_stats = self._computeOneElevationSlope(roi)
            
            table['area_gages2'].append(area_gages2)
            table['slope_mean'].append(slope_stats['slope_mean'])
            table['elev_mean'].append(elev_stats['elevation_mean'])
            
        dataframe = pd.DataFrame(table, index=self.basin_names)
        return dataframe
        


######################### Vegetation Stats ##################################



class Vegetation():
    
    def __init__(self, start_date: str, end_date: str, lai_band: str, 
                 forest_band: str, green_band: str, asset_folder: str, 
                 land_cover: str = 'MODIS/061/MOD15A2H'):
        
        self.start_date = ee.Date(start_date)
        self.end_date = ee.Date(end_date) 
        self.landcover_dataset = ee.ImageCollection(land_cover).\
                    filterDate(self.start_date, self.end_date).select(lai_band)
                    
        self.forest_fraction = ee.ImageCollection("MODIS/006/MOD44B").\
                    filterDate('2000-03-05', '2020-03-05').select(forest_band)
                    
        self.green_fraction = ee.ImageCollection("MODIS/061/MOD13Q1").\
                    filterDate(self.start_date, self.end_date).select(green_band)
        
        self.roi_lst, self.basin_names = extractROI(asset_folder)
        self.months  = ee.List.sequence(1, 12)
        self.gvf_range = {"ndvi_min": ee.Number(-0.2), "ndvi_max": ee.Number(1)}
     
        
    
    def _compute_monthly_lai_stats(self, month: str)->ee.ImageCollection:
        
        monthlyLai = self.landcover_dataset.filter(ee.Filter.calendarRange(month, month, 'month'))
        
        monthly_mean_lai = monthlyLai.mean().rename('lai_mean')
        monthly_max_lai = monthlyLai.max().rename('lai_max') # max of a month across the years 
        monthly_min_lai = monthlyLai.min().rename('lai_min') # min of a month across the years 
        
        monthly_stats_lai = monthly_mean_lai.addBands([monthly_max_lai, monthly_min_lai])
        
        return monthly_stats_lai.set('month', month)
        
        
    def _computeOneLaiStats(self, roi: ee.Geometry)->Tuple[float, float]:
        
        monthly_lai_stats = self.landcover_dataset.\
                            fromImages(self.months.map(lambda month: self._compute_monthly_lai_stats(month)))
        
        
        clipped_lai_images =  monthly_lai_stats.map(lambda img: img.clip(roi))
        lai_mean_images = clipped_lai_images.select('lai_mean')
        
    
        max_lai_mean = lai_mean_images.reduce(ee.Reducer.max()).reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=roi,
                            scale=500,  
                            maxPixels=1e9
                            )
        
        min_lai_mean = lai_mean_images.reduce(ee.Reducer.min()).reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=roi,
                            scale=500,  
                            maxPixels=1e9
                            )
        
        num_max_lai_mean = max_lai_mean.getInfo()['lai_mean_max']
        num_min_lai_mean = min_lai_mean.getInfo()['lai_mean_min']
        mean_diff = num_max_lai_mean - num_min_lai_mean
        
        return round(num_max_lai_mean/4, 7), round(mean_diff/4,7)  # 31/8-day composite
    
    
    
    def _computeForestFractionStats(self, roi: ee.Geometry)->float:
        
        clipped_tree_cover = self.forest_fraction.map(lambda img: img.clip(roi))
        
        mean_tree_cover = clipped_tree_cover.reduce(ee.Reducer.mean()).reduceRegion(
                            reducer = ee.Reducer.mean(),
                            geometry=roi,
                            scale=250,
                            maxPixels=1e9
                          )
        
        return round(mean_tree_cover.getInfo()['Percent_Tree_Cover_mean']/100, 7)
    
    # aggregate green fractions for a month across the years 
    def _computeOneGreenFraction(self, month: ee.Number)->ee.Image:
        
        monthly_ndvi = self.green_fraction.filter(ee.Filter.calendarRange(month, month, 'month'))
        mean_monthly_ndvi = monthly_ndvi.mean().rename('mean_ndvi')
        return mean_monthly_ndvi.set('month', month)
    
    
    # TODO: Check everything!
    def _computeGreenFractionStats(self, roi: ee.Geometry) -> Tuple[float, float, float]:
        
        """
        Compute Green Vegetation Fraction (GVF) statistics for a given region of interest (ROI) using Google Earth Engine.

        Parameters:
        - roi (ee.Geometry): Region of interest defined as an Earth Engine Geometry object.

        Returns:
        - Tuple[float, float]: Tuple containing mean GVF maximum value and GVF range (difference between max and min).
        Values are rounded to 7 decimal places.

        Notes:
        - Uses 'green_fraction' dataset and computes mean monthly NDVI (Normalized Difference Vegetation Index) for each month (1-12).
        - Clips images to the provided ROI and calculates GVF using the formula: (ndvi - ndvi_min) / (ndvi_max - ndvi_min).
        - Computes statistics at a scale of 250 meters based on mean GVF values within the ROI.
        
        """
        
        # calculate for each month (1-12) its mean 
        mean_monthly_ndvi = self.green_fraction.\
                            fromImages(self.months.map(lambda month: self._computeOneGreenFraction(month)))
        # clip the images with provided roi 
        clipped_imgs = mean_monthly_ndvi.map(lambda img: img.clip(roi)).select('mean_ndvi') 
        equation = '(ndvi - ndvi_min)/(ndvi_max - ndvi_min)'
        #convert ndvi values to gvf 
        gvf = clipped_imgs.map(lambda img: img.expression(equation, {'ndvi': img.select('mean_ndvi'),
                                                          'ndvi_max': self.gvf_range['ndvi_max'],
                                                          'ndvi_min': self.gvf_range['ndvi_min']}).rename('gvf'))
        
        gvf = gvf.select('gvf').reduce(ee.Reducer.minMax()).reduceRegion(
            reducer = ee.Reducer.mean(),
            geometry = roi,
            scale = 250,
            maxPixels=1e9
        )
        gvf_diff = gvf.getInfo()['gvf_max'] - gvf.getInfo()['gvf_min'] 
        
        return round(gvf.getInfo()['gvf_max']/10000,7), round(gvf_diff/10000,7)


        
    def computeAllStats(self):
        
        table = {"lai_max": [], "lai_diff": [], "forest_frac": [],
                 "gvf_max": [], "gvf_diff": []}
        
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} - Vegetation]'
        
        for idx, roi in enumerate(tqdm(self.roi_lst, bar_format=bar_format)):
            
            # if idx==5:
            #     break
            lai_max, lai_diff = self._computeOneLaiStats(roi)
            forest_frac = self._computeForestFractionStats(roi)
            gvf_max, gvf_diff = self._computeGreenFractionStats(roi)
            
            table['lai_max'].append(lai_max)
            table['lai_diff'].append(lai_diff)
            table['forest_frac'].append(forest_frac)
            table['gvf_max'].append(gvf_max)
            table['gvf_diff'].append(gvf_diff)
            
            
        dataframe = pd.DataFrame(table, index=self.basin_names)
        return dataframe
    
    
    
    
########################### Soil Stats with water content ##############################
    
    
    
class SoilMetrics:
    
    def __init__(self, asset_fldr: str):
        self.roi_lst, self.basin_names = extractROI(asset_fldr)
        self.clay_data = ee.Image("projects/soilgrids-isric/clay_mean")
        self.sand_data = ee.Image("projects/soilgrids-isric/sand_mean")
        self.silt_data = ee.Image("projects/soilgrids-isric/silt_mean")
        self.water_content = ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/wcsat")
        self.ksat_data = ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/ksat")
        
    def _computeOneSoilStat(self, roi: ee.Geometry):
        
        """
        Compute soil statistics for a specified region of interest (ROI) using Google Earth Engine.

        Parameters:
        - roi (ee.Geometry): Region of interest defined as an Earth Engine Geometry object.

        Returns:
        - Tuple[float, float, float, float, float]: Tuple containing soil statistics including:
        1. Clay fraction (0-5 cm depth), scaled to 0-1 range.
        2. Sand fraction (0-5 cm depth), scaled to 0-1 range.
        3. Silt fraction (0-5 cm depth), scaled to 0-1 range.
        4. Maximum water content (cm³/cm³) available in the soil.
        5. Estimated saturated hydraulic conductivity (Ksat) in cm/hour.

        Notes:
        - Uses 'clay_data', 'sand_data', 'silt_data', 'water_content', and 'ksat_data' datasets from Google Earth Engine.
        - Statistics are computed at a scale of 250 meters based on mean values within the ROI.
        - Ksat value is normalized by dividing by 24,000 (a hypothetical scaling factor) to adjust for specific dataset units.

        """
        
        
        clay_stats = self.clay_data.clip(roi).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi, 
            scale=250,
            maxPixels=1e9
        ).getInfo()
        
        sand_stats = self.sand_data.clip(roi).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi, 
            scale=250,
            maxPixels=1e9
        ).getInfo()
        
        silt_stats = self.silt_data.clip(roi).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi, 
            scale=250,
            maxPixels=1e9
        ).getInfo()
        
        max_water_content = self.water_content.reduce(ee.Reducer.mean()).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=roi,
            scale=250,
            maxPixels=1e9
        ).getInfo()

        
        ksat_stats = self.ksat_data.reduce(ee.Reducer.mean()).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi, 
            scale=250,
            maxPixels=1e9
        ).getInfo()
        
        ksat_value = list(ksat_stats.values())[0]
        max_water_content_value = list(max_water_content.values())[0] / 10000 

        return (
            clay_stats['clay_0-5cm_mean'] / 10, 
            sand_stats['sand_0-5cm_mean'] / 10, 
            silt_stats['silt_0-5cm_mean'] / 10,
            max_water_content_value,
            ksat_value/24000
            )
        
    def computeAllSoilStats(self):
        
        table = {"clay_frac": [], "sand_frac": [], "silt_frac": [], 
                "max_water_content": [], "hydraulic_conductivity": []}
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} - SoilMetrics]'
        
        for idx, roi in enumerate(tqdm(self.roi_lst, bar_format=bar_format)):
            
            # if idx == 5:
            #     break
            
            clay_stats, sand_stats, silt_stats, water_cnt, ksat_value = self._computeOneSoilStat(roi)
            
            table['clay_frac'].append(clay_stats)
            table['sand_frac'].append(sand_stats)
            table['silt_frac'].append(silt_stats)
            table['max_water_content'].append(water_cnt)
            table['hydraulic_conductivity'].append(ksat_value)
            
        dataframe = pd.DataFrame(table, index=self.basin_names)
        return dataframe