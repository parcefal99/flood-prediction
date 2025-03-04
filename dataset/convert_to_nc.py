'''
This python script creates timeseries with NetCDF formatted files from the merging 
of meteo and streamflow data (.csv files). This is due to model specifications,
since the data is loaded only by the following format.

Please define the necessary directories, where the meteo and streamflow data is located, 
and use the script as follows:
    python convert_to_nc.py 
'''

import os
import pandas as pd
import xarray as xr

# Define directories
dir1 = ''  # Directory with (left-side) meteo data files
dir2 = ''  # Directory with (right-side) streamflow data files
output_dir = ''  # Directory to save .nc files

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through all CSV files in the first directory
for filename in os.listdir(dir1):
    if filename.endswith('.csv'):
        file_id = os.path.splitext(filename)[0]  # Extract the id (without .csv)

        # Construct the full paths for the files in both directories
        file1_path = os.path.join(dir1, filename)
        file2_path = os.path.join(dir2, filename)

        # Check if the corresponding file exists in the second directory
        if os.path.exists(file2_path):
            # Read both CSV files
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(file2_path)

            # Ensure 'date' is parsed as datetime for both dataframes
            df1["date"] = pd.to_datetime(df1["date"])
            df2['date'] = pd.to_datetime(df2['date'])
            # df1 = df1.set_index("date")

            # Perform left-side merge on 'date' column
            merged_df = pd.merge(df1, df2, on='date', how='left')
            merged_df = merged_df.set_index('date')

            # Convert the merged DataFrame to xarray.Dataset
            ds = xr.Dataset.from_dataframe(merged_df)

            # Define the output NetCDF filename and save
            output_file = os.path.join(output_dir, f'{file_id}.nc')
            ds.to_netcdf(output_file)

            print(f'Merged and saved as NetCDF: {output_file}')
        else:
            print(f'No corresponding file in directory 2 for {filename}')

print('All files processed successfully.')