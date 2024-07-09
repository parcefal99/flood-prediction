from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import numpy as np
from io import StringIO
import os
import re
import time
import glob
import shutil

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
driver = webdriver.Chrome(options=options)

sidebar_options = [
    'ul[data-expanded="2.1Temperature"] > li:nth-child(1)',
    'ul[data-expanded="2.1Temperature"] > li:nth-child(2)',
    'ul[data-expanded="2.1Temperature"] > li:nth-child(3)',
    'ul[data-expanded="2Dailydata"] > li:nth-child(2)',
    'ul[data-expanded="2Dailydata"] > li:nth-child(3)',
    'ul[data-expanded="2Dailydata"] > li:nth-child(4)',
    'ul[data-expanded="2Dailydata"] > li:nth-child(5)',
    'ul[data-expanded="2Dailydata"] > li:nth-child(6)',
    'ul[data-expanded="2Dailydata"] > li:nth-child(7)',
    'ul[data-expanded="2Dailydata"] > li:nth-child(8)',
    'ul[data-expanded="2Dailydata"] > li:nth-child(9)',
    'ul[data-expanded="2Dailydata"] > li:nth-child(10)'
]

col_names = [
    ["station", "date", "t_mean", "t_max", "t_min"],
    ["station", "date", "st_mean", "st_max", "st_min"],
    ["station", "date", "dew_min"],
    ["station", "date", "pp_mean"],
    ["station", "date", "hum_mean", "hum_min"],
    ["station", "date", "sat_mean", "sat_max"],
    ["station", "date", "atm_sta", "atm_sea"],
    ["station", "date", "cl_o", "cl_h"],
    ["station", "date", "wind_mean", "wind_max8", "wind_max"],
    ["station", "date", "prcp"],
    ["station", "date", "soil"],
    ["station", "date", "snow_cover", "snow_height"]
]

js_script = """$.fn.dataTable.ext.errMode = 'throw';$.extend(true, $.fn.dataTable.defaults, {"pageLength": -1, "paging": false, "scrollY": ""});"""

regions = ["KZ-ABA", "KZ-AKM", "KZ-AKT", "KZ-ALM", "KZ-ATY", "KZ-KAR", "KZ-KUS", "KZ-KZY", "KZ-MAN", "KZ-PAV", "KZ-SEV", "KZ-TUR", "KZ-ULY", "KZ-VOS", "KZ-ZAP", "KZ-ZHA", "KZ-ZHE"]

stations = [15, 16, 17, 20, 9, 18, 18, 9, 7, 15, 11, 14, 6, 15, 13, 13, 12]

side, reg, sta = 1, 1, 1


def clean_cell(value):
    if pd.isna(value):
        return value
    new_value = re.sub(r'[^0-9.-]', '', str(value).replace(',', '.'))
    return np.nan if new_value == '-' else ('0' if (new_value == '-0' or new_value == '-0.0') else new_value)


def get_station():
    global side, reg, sta

    if not os.path.exists(os.path.join(os.path.join(os.path.join("kazhydromet", "meteo"), regions[reg - 1]), str(sta))):
        os.makedirs(os.path.join(os.path.join(os.path.join("kazhydromet", "meteo"), regions[reg - 1]), str(sta)))

    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[2]/div/div/div'))).click()
    station = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, f'//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[2]/div/div/div/div/div[2]/div[1]/div[{sta}]')))
    driver.execute_script("arguments[0].click();", station)
    station_name = station.get_attribute("innerText")
    time.sleep(5)

    table_html = WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[2]/div/div/div[4]/div[2]/table'))).get_attribute('outerHTML')
    df = pd.read_html(StringIO(table_html))[0]

    df.columns = col_names[side - 1]

    if station_name != df['station'].iloc[0]:
        raise StaleElementReferenceException(f"{df['station'].iloc[0]} is different than {station_name}")

    df.drop_duplicates(inplace=True)
    df['station'] = df['station'].str.replace(' ', '_')
    for col in df.columns[2:]:
        df[col] = df[col].apply(clean_cell)
    if side == 10:
        df['prcp'] = df['prcp'].fillna(0.0)
    
    file_name = f"{regions[reg - 1]}_{station_name}_{side}.csv"
    file_path = os.path.join(os.path.join(os.path.join(os.path.join("kazhydromet", "meteo"), regions[reg - 1]), str(sta)), file_name)
    df.to_csv(file_path, sep=';', index=False)
    print(f"================ Station: {sta} ================")


def get_region():
    global side, reg, sta

    if not os.path.exists(os.path.join(os.path.join("kazhydromet", "meteo"), regions[reg - 1])):
        os.makedirs(os.path.join(os.path.join("kazhydromet", "meteo"), regions[reg - 1]))

    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[1]/div/div/div'))).click()
    driver.execute_script("arguments[0].click();", WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, f'//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[1]/div/div/div/div[2]/div[1]/div[{reg}]'))))
    time.sleep(5)
    
    while sta <= stations[reg - 1]:
        get_station()
        sta += 1

    print(f"================ Region: {reg} ================\n")
    sta = 1


def get_sidebar():
    global side, reg, sta

    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sidebar_options[side-1]))).click()
    time.sleep(10)

    if side > 7:
        WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[2]/div/div[contains(@class, "dataTables_wrapper")]')))
        driver.execute_script(js_script)

    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[3]/div/div/input[2]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-days"]//descendant::th[@class="datepicker-switch"]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-months"]//descendant::th[@class="datepicker-switch"]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-years"]//descendant::th[@class="datepicker-switch"]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-decades"]//descendant::th[@class="datepicker-switch"]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-centuries"]//descendant::tbody/tr/td/span[3]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-decades"]//descendant::tbody/tr/td/span[2]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-years"]//descendant::tbody/tr/td/span[2]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-months"]//descendant::tbody/tr/td/span[1]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-days"]//descendant::tbody/tr/td[6]'))).click()

    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[3]/div/div/input[1]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-days"]//descendant::th[@class="datepicker-switch"]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-months"]//descendant::th[@class="datepicker-switch"]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-years"]//descendant::th[@class="datepicker-switch"]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-decades"]//descendant::th[@class="datepicker-switch"]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-centuries"]//descendant::tbody/tr/td/span[1]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-decades"]//descendant::tbody/tr/td/span[2]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-years"]//descendant::tbody/tr/td/span[2]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-months"]//descendant::tbody/tr/td/span[1]'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-days"]//descendant::tbody/tr/td[2]'))).click()

    while reg <= 17:
        get_region()
        reg += 1

    print(f"================ Sidebar: {side} ================\n")
    reg = 1


if __name__ == '__main__':
    print("\n================ STARTING... ================\n")
    driver.get("http://ecodata.kz:3838/dm_climat_en/")

    if not os.path.exists("kazhydromet"):
        os.makedirs("kazhydromet")
        os.makedirs(os.path.join("kazhydromet", "meteo"))

    while True:
        try:
            time.sleep(10)
            WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'ul.sidebar-menu > li:nth-child(4)'))).click()
            WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'ul[data-expanded="2Dailydata"] > li:first-child'))).click()
            WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'ul[data-expanded="2.1Temperature"]'))).click()

            while side <= 12:
                get_sidebar()
                side += 1
            break
        except Exception as e:
            print(f"\nSidebar: {side}\nRegion: {reg}\nStation: {sta}")
            print(f"Error: {e}\n")
            driver.refresh()
    
    print("================ DOWNLOADED ================\n")
    driver.quit()
    
    print("================ COMBINING... ================\n")
    for reg in range(17):
        
        for sta in range(stations[reg]):
                
            df_list = []

            for side in range(1, 13):
                file_name = f"{regions[reg]}_*_{side}.csv"
                file_path = glob.glob(os.path.join(os.path.join(os.path.join(os.path.join("kazhydromet", "meteo"), regions[reg]), str(sta + 1)), file_name))[0]

                df = pd.read_csv(file_path, sep=';')
                df_list.append(df)
        
            combined_df = df_list[0]
            for df in df_list[1:]:
                combined_df = pd.merge(combined_df, df, on=['station', 'date'], how='outer')
            combined_df.drop_duplicates(inplace=True)
            
            file_name = f"{regions[reg]}_{combined_df['station'].iloc[0]}_meteo.csv"
            file_path = os.path.join(os.path.join(os.path.join("kazhydromet", "meteo"), regions[reg]), file_name)
            combined_df.to_csv(file_path, sep=';', index=False)
            shutil.rmtree(os.path.join(os.path.join(os.path.join("kazhydromet", "meteo"), regions[reg]), str(sta + 1)))
            print(f"================ Station: {sta + 1} ================")
        print(f"================ Region: {reg + 1} ================\n")
    print("================ SUCCESS ================\n")
