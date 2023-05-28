import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import aiohttp
import requests
import asyncio
import re

async def scraping_main():
    logging.info('Started WebScraping')
    driver = webdriver.Chrome(executable_path = 'artifacts/chromedriver.exe')
    Links = []
    LinksIndex = 0
    for bedrooms in range(1,7):
        if bedrooms==6:
            bedrooms = '>5'
        url = f'https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom={bedrooms}&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment&cityName=Mumbai'
        driver.get(url)
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(5)
                new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        webpage = driver.page_source
        LinkIndexes = [m.start() for m in re.finditer('https://www.magicbricks.com/propertyDetails/', webpage)]
        for index in LinkIndexes:
            Links.append('')
            for urlindex in range(index,index+200):
                if webpage[urlindex] == '"':
                    break
                else:
                    Links[LinksIndex] += webpage[urlindex]
            LinksIndex+=1
    Links = list(set(Links))
    print('Number of Links: ',len(Links))
    Prices = []
    Area = []
    Locations = []
    Amenities = []
    Bedrooms = []
    driver.close()
    Amenities_List = ['Reserved Parking','Lift','Power Back Up','Piped Gas','Park','Kids play area','Gymnasium','Swimming Pool',
                  'Club House','Air Conditioned','Internet/Wi-Fi Connectivity']
    async def scraping(url,session):
        async with session.get(url) as page:
            try:
                page1 = await page.text()
                HTMLPage = BeautifulSoup(page1, 'html.parser')
                price = HTMLPage.find(class_='mb-ldp__dtls__price').text
                pagesqft = HTMLPage.find(class_='mb-ldp__dtls__body__list')
                areaindex = pagesqft.text.find('Area')
                sqftindex = pagesqft.text.find('sqft')
                sqft = ''
                for index in range(areaindex,sqftindex+4):
                    sqft+=pagesqft.text[index]
                sqft = sqft[4:]
                pagelocation = HTMLPage.find_all(class_='mb-ldp__more-dtl__list--value')
                for text in pagelocation:
                    text = text.text
                    if 'Mumbai' in text:
                        location = text
                amenities = ''
                for Amenity in Amenities_List:
                    if Amenity in page1:
                        amenities += Amenity + ','
                pageBHK = HTMLPage.find(class_='mb-ldp__dtls__title--text1--text')
                pageBHK = pageBHK.text
                BHKIndex = pageBHK.find('BHK')
                BHK = ''
                for index in range(BHKIndex-2,BHKIndex+3):
                    BHK+=pageBHK[index]
                Area.append(sqft)
                Prices.append(price)
                Amenities.append(amenities)
                Bedrooms.append(BHK)
                Locations.append(location)
            except:
                pass
    async def main():
        async with aiohttp.ClientSession() as session:
            tasks = [await scraping(url,session) for url in Links]
            logging.info('Finished WebScraping')

    await main()
    return Prices,Area,Locations,Amenities,Bedrooms
@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    async def  initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            Prices,Area,Locations,Amenities,Bedrooms = await scraping_main()
            df = pd.DataFrame({'Price': pd.Series(Prices),
                  'Location': pd.Series(Locations),
                  'Area': pd.Series(Area),
                  'Amenities':pd.Series(Amenities),
                  'BHK': pd.Series(Bedrooms)})
            logging.info('Completed the WebScraping')
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")
        except Exception as e:
            raise CustomException(e,sys)

async def data_ingestion():
    obj=DataIngestion()
    await obj.initiate_data_ingestion()