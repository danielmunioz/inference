from TempMail import TempMail
import poe
import time
import re
import os
from multiprocessing import Queue
from typing import Optional
import traceback
import subprocess
# from tqdm.autonotebook import tqdm

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


class NeedsNewMail(Exception):
    pass


def tqdm_print(*msg):
    # return
    print(*msg, flush=True)
    #tqdm.write(" ".join([str(m) for m in msg]))


def get_tokens(token_queue: Optional[Queue] = None):
    os.makedirs("lib_scraping/scrape/result", exist_ok=True)
    while True:
        try:
            tmp = TempMail()
            inbox = TempMail.generateInbox(tmp)
            tqdm_print("Temp email:", inbox.address)

            service = Service("/usr/bin/chromedriver")
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_experimental_option("excludeSwitches", ["enable-logging"])
            driver = webdriver.Chrome(service=service, options=options)
            driver.delete_all_cookies()
            driver.get("https://relay.firefox.com")

            try:
                WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//a[text()='Sign Up']"))).click()
            except TimeoutException:
                WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "use-different"))).click()

            tqdm_print("Clicked sign up")
            email_input = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, "input-text")))
            email_input.send_keys(Keys.CONTROL + "a")
            email_input.send_keys(Keys.DELETE)
            email_input.send_keys(inbox.address)
            email_input.send_keys(Keys.RETURN)

            tqdm_print("Entered mail")
            WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.ID, "password"))).send_keys("password@123456")
            driver.find_element(By.ID, "vpassword").send_keys("password@123456")
            driver.find_element(By.ID, "age").send_keys("23")
            driver.find_element(By.ID, "submit-btn").click()

            code = None
            while not code:
                tqdm_print("Waiting for relay code...")
                time.sleep(5)
                for mail in TempMail.getEmails(tmp, inbox):
                    if "Confirm your account" not in mail.subject:
                        continue

                    code = re.findall(r"\d{6}", mail.html)
                    code = [x for x in code if x != '000000'][0]

            tqdm_print("Verification code:", code)
            code_input = driver.find_element(By.ID, "otp-code")
            code_input.send_keys(code)
            code_input.send_keys(Keys.RETURN)

            while True:
                button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//button[starts-with(@title, 'Generate new')]")))
                driver.execute_script("arguments[0].scrollIntoView();", button)
                driver.execute_script("arguments[0].click();", button)
                relay_mail = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.TAG_NAME, "samp"))).text
                token = None
                try:
                    token = get_poe_token(tmp, inbox, relay_mail)
                    poe.Client(token)
                    tqdm_print("Token:", token)
                    with open('lib_scraping/scrape/result/poe_tokens.txt', 'a') as f:
                        f.write(token + "\n")
                    if token_queue is None:
                        return token  # for sequentially testing
                    else:
                        token_queue.put(token)
                except NeedsNewMail:
                    break
                except Exception as e:
                    tqdm_print(repr(e))
                    pass
                finally:
                    try:
                        expand_button = driver.find_element(By.XPATH, "//*[name()='svg' and contains(@aria-label,'mask details')]/parent::button")
                        if expand_button.get_attribute("aria-expanded") == "false":
                            expand_button.click()
                    except Exception:
                        pass
                    delete_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(@class,'AliasDeletionButton')]")))
                    driver.execute_script("arguments[0].scrollIntoView();", delete_button)
                    driver.execute_script("arguments[0].click();", delete_button)
                    delete_checkbox = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "confirmDeletion")))
                    delete_checkbox.click()
                    delete_checkbox.send_keys(Keys.RETURN)
                    tqdm_print("Deleted relay email")
                    time.sleep(2)
            driver.close()
        except Exception:
            tqdm_print(traceback.format_exc())
            subprocess.run(["pkill", "chrome"])


def get_poe_token(tmp, inbox, relay_mail):
    service = Service("/usr/bin/chromedriver")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://poe.com/login")

    button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//button[text()='Use email']")))
    button.click()

    tqdm_print("Relay email:", relay_mail)
    email_input = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, "//input[@type='email']")))
    email_input.send_keys(relay_mail)
    email_input.send_keys(Keys.RETURN)

    code = None
    count = 0
    while count < 3:
        tqdm_print(f"Waiting for Poe code... {count+1}/3")
        time.sleep(10)
        try:
            for mail in TempMail.getEmails(tmp, inbox):
                if "Your verification code" not in mail.subject:
                    continue

                code = re.search(r"(\d{6})</div>", mail.html).group(1)
                count = 3
                break
            else:
                count += 1
        except Exception:
            tqdm_print("Creating a new temp mail...")
            driver.close()
            raise NeedsNewMail
    if not code:
        driver.close()
        raise RuntimeError
    
    tqdm_print("Verification code:", code)
    code_input = driver.find_element(By.XPATH, "//input[@placeholder='Code']")
    code_input.send_keys(code)
    code_input.send_keys(Keys.RETURN)

    time.sleep(5)
    token = driver.get_cookie("p-b")["value"]
    driver.close()
    return token


if __name__ == "__main__":
    token = get_tokens()
    tqdm_print(token)
