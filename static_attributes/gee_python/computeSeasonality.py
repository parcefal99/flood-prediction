
import pandas as pd 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from tqdm import tqdm
import os


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
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        
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


# season = PrcpSeasonality("/Users/abzal/Desktop/issai-srp/KazDataset/meteo")
# result = season.computeAllSeasonality(output_logs=True)
# result
