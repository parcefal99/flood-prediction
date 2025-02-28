from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import numpy as np
from io import StringIO
import os
import re
import time

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
driver = webdriver.Chrome()

sidebar_options = [
    'ul.sidebar-menu > li:nth-child(4)',
    'ul.sidebar-menu > li:nth-child(3)'
]

col_names = [
    ["post_code", "date", "discharge"],
    ["post_code", "date", "level", "symbol"]
]

js_script = """$.fn.dataTable.ext.errMode = 'throw';$.extend(true, $.fn.dataTable.defaults, {"pageLength": -1, "paging": false, "scrollY": ""});"""

cyrillic_to_latin = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E', 'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 
    'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 
    'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch', 'Ъ': '', 'Ы': 'Y', 'Ь': "'", 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 
    'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 
    'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya', '¦' : ''
}


def clean_cell(value):
    if pd.isna(value):
        return value
    new_value = re.sub(r'[^0-9.-]', '', str(value).replace(',', '.'))
    return np.nan if new_value == '-' else ('0' if (new_value == '-0' or new_value == '-0.0') else new_value)


def capitalize_after_space(text):
    return re.sub(r' (.)', lambda x: x.group(1).upper(), text)


def replace_cyrillic(text, mapping):
    for cyrillic, latin in mapping.items():
        text = text.replace(cyrillic, latin)
    return text


def get_post_list():
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'ul.sidebar-menu > li:nth-child(5)'))).click()
    time.sleep(3)
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[2]/div/div/div/label/select'))).click()
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[2]/div/div/div/label/select/option[4]'))).click()
    time.sleep(2)

    post_list_parts = []

    while True:
        table_html = WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[2]/div/div/table'))).get_attribute('outerHTML')
        post_list_part = pd.read_html(StringIO(table_html))[0]
        post_list_parts.append(post_list_part)
        time.sleep(0.5)
        next_btn = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[2]/div/div/div[5]/a[contains(@class, "paginate_button") and contains(@class, "next")]')))
        if next_btn.get_attribute("tabindex") == "-1":
            break
        next_btn.click()
        time.sleep(1)

    post_list = pd.concat(post_list_parts, ignore_index=True)
    post_list.columns = ["basin", "water_object", "object_code", "post_name", "post_code"]
    post_list['water_object'] = post_list['water_object'].apply(lambda x: replace_cyrillic(x, cyrillic_to_latin))
    post_list['post_name'] = post_list['post_name'].apply(lambda x: replace_cyrillic(x, cyrillic_to_latin))
    post_list['water_object'] = post_list['water_object'].apply(capitalize_after_space)
    post_list['post_name'] = post_list['post_name'].apply(capitalize_after_space)

    if not os.path.exists(os.path.join("kazhydromet", "hydro")):
        os.makedirs(os.path.join("kazhydromet", "hydro"))

    file_path = os.path.join(os.path.join("kazhydromet", "hydro"), "post_list.csv")
    post_list.to_csv(file_path, sep=';', index=False)
    print("================ POST LIST DOWNLOADED ================\n")
    return post_list


if __name__ == '__main__':
    print("\n================ STARTING... ================\n")
    driver.get("insert_the_website_address")
    post_list = get_post_list()

    st, sti = 0, 0

    while True:
        try:
            for index, row in post_list.iterrows():
                if index < st:
                    continue
                sti = index
                basin, w_obj, obj_code, post_name, post_code = row['basin'], row['water_object'], row['object_code'], row['post_name'], row['post_code']

                if not os.path.exists(os.path.join(os.path.join("kazhydromet", "hydro"), basin)):
                    os.makedirs(os.path.join(os.path.join("kazhydromet", "hydro"), basin))
                
                df_list = []
                found = [False, False]

                for i, sidebar in enumerate(sidebar_options):
                    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sidebar))).click()
                    
                    if index == st:
                        time.sleep(5)
                        WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[2]/div/div[contains(@class, "dataTables_wrapper")]')))
                        driver.execute_script(js_script)
                    
                    WebDriverWait(driver, 0).until(EC.element_to_be_clickable((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[1]/div/div/div'))).click()
                    driver.execute_script("arguments[0].click();", WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, f'//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[1]/div/div/div/div[2]/div[1]/div[@data-value="{basin}"]'))))
                    time.sleep(3 if index == st else 1)

                    while True:
                        try:
                            WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[2]/div/div/div'))).click()
                            driver.execute_script("arguments[0].click();", WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.XPATH, f'//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[1]/div[2]/div/div/div/div/div[2]/div[1]/div[@data-value="{post_code}"]'))))
                            found[i] = True

                            if index == st:
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
                                WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-decades"]//descendant::tbody/tr/td/span[2]'))).click()
                                WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-years"]//descendant::tbody/tr/td/span[2]'))).click()
                                WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-months"]//descendant::tbody/tr/td/span[1]'))).click()
                                WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@class="datepicker-days"]//descendant::tbody/tr/td[2]'))).click()

                            time.sleep(4)
                            table_html = WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.XPATH, '//*[@role="tabpanel" and contains(@class, "tab-pane") and contains(@class, "active")]/div/div[2]/div/div/table'))).get_attribute('outerHTML')
                            df = pd.read_html(StringIO(table_html))[0]
                            break
                        except TimeoutException:
                            if found[i]:
                                continue
                            break
                    
                    if found[i]:
                        df.columns = col_names[i]
                        if i == 1:
                            df = df.drop(columns=['symbol'])
                        df_list.append(df)

                if not (found[0] or found[1]):
                    continue
                if found[0] and found[1]:
                    combined_df = pd.merge(df_list[0], df_list[1], on=['post_code', 'date'], how='outer').dropna(how='all')
                else:
                    combined_df = df_list[0]
                    if found[0]:
                        combined_df['level'] = np.nan
                    else:
                        combined_df['discharge'] = np.nan
                
                if post_code != df['post_code'].iloc[0]:
                    raise StaleElementReferenceException(f"{df['post_code'].iloc[0]} is different than {post_code}")

                combined_df['post_name'] = post_name
                combined_df['basin'] = basin
                combined_df['water_object'] = w_obj
                combined_df['object_code'] = obj_code
                new_order = ['post_code', 'post_name', 'basin', 'water_object', 'object_code', 'date', 'level', 'discharge']
                combined_df = combined_df[new_order]

                combined_df['level'] = (combined_df['level'].astype(str)).str.replace('нб', '0.0')
                combined_df['discharge'] = (combined_df['discharge'].astype(str)).str.replace('нб', '0.0')
                combined_df['level'] = (combined_df['level'].astype(str)).str.replace(',', '.')
                combined_df['discharge'] = (combined_df['discharge'].astype(str)).str.replace(',', '.')
                combined_df['level'] = combined_df['level'].apply(clean_cell)
                combined_df['discharge'] = combined_df['discharge'].apply(clean_cell)
                
                file_name = f"{basin}_{post_code}_hydro.csv"
                file_path = os.path.join(os.path.join(os.path.join("kazhydromet", "hydro"), basin), file_name)
                combined_df.to_csv(file_path, sep=';', index=False)
                print(f"================ Station: {index + 1}-{post_code} ================")
            break
        except Exception as e:
            print(f"Error: {e}")
            st = sti
            driver.refresh()

    print("\n================ SUCCESS ================\n")
    driver.quit()
