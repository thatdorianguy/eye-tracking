from selenium import webdriver
import time

from selenium.webdriver import ActionChains, Keys

driver = webdriver.Chrome()
driver.get("https://en.wikipedia.org/wiki/Taylor_Swift")
time.sleep(4)
driver.execute_script("window.scrollBy(0,40)")
time.sleep(0.5)
