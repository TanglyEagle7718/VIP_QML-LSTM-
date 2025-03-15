from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import csv
import time

def scrape_pages(start_page, end_page):
    
    url_base = "https://www.investing.com/news/stock-market-news/"

    options = Options()
    options.add_argument("--headless") 
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0")

    service = Service(ChromeDriverManager().install())

    headlines = []
    
    for page_num in range(start_page, end_page + 1):
        driver = webdriver.Chrome(service=service, options=options)
        url = url_base + str(page_num) if page_num > 1 else url_base

        driver.get(url)

        time.sleep(2)

        print(str(round(((page_num - (start_page - 1))/(end_page - (start_page - 1))) * 100, 2)) + "%") #for my sanity

        articles = driver.find_elements(By.CSS_SELECTOR, "a.text-inv-blue-500")
        for article in articles:
            header = article.text.strip()
            try:
                date = article.find_element(By.XPATH, "..").find_element(By.TAG_NAME, "time").get_attribute('datetime')
                if header is not None and len(header) > 25: #limit the size of articles to avoid empty strings and headlines that are too short
                    headlines.append((date, header))
            except:
                continue
        driver.quit()
    save_to_csv(headlines, filename=("headlines" + str(start_page) + "-" + str(end_page) + ".csv"))

def save_to_csv(data, filename="headlines.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Headline"])
        writer.writerows(data)

for i in range(2006, 3000, 10):
    try:
        scrape_pages(i, i+9)
    except:
        i -= 10