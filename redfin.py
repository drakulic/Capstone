import pymongo
import json
import bs4
from pymongo import MongoClient
from pymongo import errors
from pymongo.errors import DuplicateKeyError, CollectionInvalid
from bs4 import BeautifulSoup
import requests
import time
import pdb
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import os

req = requests.get('http://www.zillow.com/homes/recently_sold/size_sort/')
html = BeautifulSoup(req.content, 'html.parser')

html.find_all('a', attrs ={'href' : '/homedetails/'})
