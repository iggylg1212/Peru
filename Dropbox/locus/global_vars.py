import requests
import docker
import time
import json
import pandas as pd
import re
import csv
import random
from selenium import webdriver
from bs4 import BeautifulSoup
import sys
import numpy as np
import difflib
import math
import pickle
import datetime
import psutil
from sys import platform

def getFileHeader(given_path, method = 'r'):
    request = open(given_path, method)
    return request.readline()

def getFileBody(given_path, method = 'r'):
    request = open(given_path, method)
    request.readline()
    return request

def common_data(list1, list2): 
    result = False
    for x in list1: 
        for y in list2: 
            if x == y: 
                result = True
                return result  
    return result 

def getValFromFileLine(line, rowName, file_path):
    theHeader = getFileHeader(file_path)
    if theHeader.count("\",\"") < 5:
        initialIndex = theHeader[:theHeader.find(rowName)].count(',')
        counter = 0
        lineSplit = line.split("\"")
        valList = []
        for x in lineSplit:
            if (counter % 2) == 0:
                sublist = []
                for y in x.split(","):
                    sublist.append(y)
                if counter != 0:
                    sublist.pop(0)
                if counter != len(lineSplit)-1:
                    sublist.pop()
                valList.extend(sublist)
            else:
                valList.append(x)
            counter += 1
        return valList[initialIndex]
    else:
        initialIndex = theHeader[:theHeader.find(rowName)].count("\",\"")
        result = line.split("\",\"")[initialIndex]
        if initialIndex == 0:
            return result[1:]
        return result

def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res


global LOCAL_LOCUS_PATH
global LOCAL_WEBDRIVER_PATH
global YEAR_LIST

global manhattan_zips, non_manhattan_zips
global correct_nyc_city, nyc_city

YEAR_LIST = ["2012","2013","2014","2015","2016","2017","2018","2019"]
YEAR_TEST = ["2019"]

manhattan_zips = ["10151","10026","10027","10030","10037","10039","10001","10011","10018","10019","10020","10036","10029","10035","10010","10016","10017","10022","10012","10013","10014","10004","10005","10006","10007","10038","10280","10002","10003","10009","10021","10028","10044","10065","10075","10128","10023","10024","10025","10031","10032","10033","10034","10040","10041","10043","10045","10055","10060","10069","10119","10103","10080","10081","10087","10090","10102","10103","10104","10105","10106","10107","10109","10110","10111","10112","10114","10115","10117","10118","10119","10120","10121","10122","10123","101124","10125","10126","10128","10130","10131","10132","10133","10138","10152","10153","10154","10155","10157","10158","10160","10162","10164","10165","10166","10167","10168","10169","10170","10171","10172","10173","10174","10175","10176","10177","10178","10179","10199","10199","10203","10211","10212","10213","10256","10258","10259","10260","10261","10265","10269","10270","10271","10273","10275","10277","10278","10279","10281","10282","10285","10286"]
non_manhattan_zips = ["10453","10457","10460","10458","10467","10468","10451","10452","10456","10454","10455","10459","10474","10463","10471","10466","10469","10470","10475","10461","10462","10464","10465","10472","10473","11212","11213","11216","11233","11238","11209","11214","11228","11204","11218","11219","11230","11234","11236","11239","11223","11224","11229","11235","11201","11205","11215","11217","11231","11203","11210","11225","11226","11207","11208","11211","11222","11220","11232","11206","11221","11237","11361","11362","11363","11364","11354","11355","11356","11357","11358","11359","11360","11365","11366","11367","11412","11423","11432","11433","11434","11435","11436","11101","11102","11103","11104","11105","11106","11374","11375","11379","11385","11691","11692","11693","11694","11695","11697","11004","11005","11411","11413","11422","11426","11427","11428","11429","11414","11415","11416","11417","11418","11419","11420","11421","11368","11369","11370","11372","11373","11377","11378","10302","10303","10310","10306","10307","10308","10309","10312","10301","10304","10305","10314","11241","11242","11243","11245","11249","11251","11252","11256","10311","11120","11351","11359","11371","11381","11405","11425","11430","11437","11439","11451","11499"]

correct_nyc_city = ['QUEENS','QUEENS','NEW YORK','NEW YORK','NEW YORK','QUEENS','QUEENS','QUEENS','NEW YORK','NEW YORK','STATEN ISLAND','STATEN ISLAND','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','BROOKLYN','BROOKLYN','BROOKLYN','QUEENS','QUEENS','NEW YORK','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','NEW YORK','NEW YORK','QUEENS','QUEENS','NEW YORK','NEW YORK','NEW YORK','NEW YORK','NEW YORK','NEW YORK','NEW YORK','NEW YORK','BROOKLYN','QUEENS','STATEN ISLAND','BROOKLYN','BROOKLYN','BROOKLYN','BROOKLYN','BRONX','BRONX','BRONX','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','NEW YORK','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS', 'QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS','QUEENS']
nyc_city = ['QUEENS VILL','ROCKAWAY','ROOSEVELT ISL','ROOSEVELT ISLAND','ROOSEVELT','SOUTH OZONE PK','SOUTH OZONE','SPRINGFEILD GARDENS','WARDS ISLAND','WARDS IS','SSTATEN ISLAND','STATE ISLAND','LONG ISLAND CTY','LONG ISLAN CITY','LONG ISLAND','L I C','HOLLIS HILLS','BROOKLY','BROOKLTN','BROOKLKYN','JACKSON HTS','S.OZONE PARK','MW YORK','LONG ISLND CITY','LONG ISLAND','KEW GARDENS HIL','CAMBRIA','BELLROSE','SPRINGFLD GDNS','ROCKAWAY PRK','ROCKAWAY POINT','ROCKAWAY PARK','ROCKAWAY BEAC','ROCKAWAY PT','ROCKAWAY BEACH','SPRNGFLD GDNS','SUNNY SIDE','S RICHMOND HL','MNAHTTAN','MANHTTAN','FLUSING','BROAD CHANNEL','NEW YORK','NYC','NY','NEWYORK','NEW YORK CITY','NEW YROK','NEW YORK NY','BEW YORK','BKLYN','QUEENS','STATEN ISLAND','BROOKLYN','BROOKYLN','BROOKLYN,','BROOKLY','BRONX','BRONX,','BRONX +','ARVERNE','ASTORIA','BAYSIDE','BAYSIDE HILLS','BELLEROSE','BELLEROSE MANOR','BREEZY POINT','CAMBRIA HEIGHTS','CAMBRIA HTS','COLLEGE POINT','CORONA','EAST ELMHURST','EASST ELHURST','ELMHURST','FAR ROCKAWAY','FLORAL PARK','FLORAL PK','S FLORAL PARK','FLUSHING','FOREST HILLS','FRESH MEADOWS','FREASH MEADOWS','FRESH MEADOW','FRESH MEDOWS','GLEN OAKS','HOLLIS','HOLLIS QUEENS','HOWARD BEACH','JACKSON HEIGHTS','JAMAICA HTS','JAMAICA HEIGHT','JAMAICA','JAMICA','JAMAICA ESTATES','JAMAICA,','KEW GARDENS','KEW GARDEN','MANHATTAN','KEW GARDENS HILLS','KEW GARDEN HILLS','KEW GARDEN HL','L.I.C.','LITTLE NECK','LONG ISLAND CITY','LONG IS CITY','LONG ISLAND CIT','LONG ISLAND CIYT','LONG ISLAND CIT','LIC','MASPETH','MIDDLE VILLAGE','MIDDLE VLG','OAKLAND GARDENS','OAKLAND GRDNS','OAKLAND GDNS','OZONE PARK','S OZONE PARK','S OZONE PK', 'SOUTH OZONE PARK','SOUTH OZONE PAR','SOUTHOZONE PARK','QUEENS VILLAGE','QUEENS VLG','REGO PARK','RICHMOND HILL','RICHMOND HL','S RICHMOND HILL','S RICHMOND HL', 'SOUTH RICHMOND','SOUTH RICHMOND HILL','RIDGEWOOD','ROCKAWAY PARK', 'ROSEDALE','ROSEDALE NY','SAINT ALBANS','ST ALBANS','SPRINGFIELD GARDENS','SPRINGFIELD GAR','SPRINGFIELD GARDEN','SUNNYSIDE','WHITESTONE','WHITE STONE','WHITEATONE','WOODHAVEN','WOODAVEN','WOODSIDE','WWODSIDE'] 

if platform == "darwin":
    # MAC OS X
    LOCAL_LOCUS_PATH = "/Users/jeremyben-meir/Dropbox/locus/"
    LOCAL_WEBDRIVER_PATH = "/usr/local/bin/chromedriver"
elif platform == "linux":
	# LINUX
    LOCAL_LOCUS_PATH = "/home/ubuntu/locus/"
    LOCAL_WEBDRIVER_PATH = "/home/ubuntu/chromedriver"
else:
    # WINDOWS
    LOCAL_LOCUS_PATH = "C:/Users/jsbmm/Dropbox/locus/"
    LOCAL_WEBDRIVER_PATH = "C:/Program Files/chromedriver.exe"

class Progress_meter():


    def __init__(self, limit, length = 1000.0):
        self.limit = limit
        self.meter = 0
        self.timestart = 0
        self.timecount = 0
        self.length = length

    def tick(self, display=False):
        if not display:
            self.meter = self.meter + 1
        else:
            self.display()
        return self

    def display(self):
        progress = int((float(self.meter)/float(self.limit)) * self.length)
        if self.meter == 0:
            print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            print('Time remaining: calculating...')
            print("Writing CSV" + ": [" + "-"*int(self.length) + "]")
            self.timestart = time.time()
        elif self.meter % int(float(self.limit)/float(self.length)) == 0:
            print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            print("CPU Usage: " + str(psutil.cpu_percent()))
            print("Memory Usage: " + str(psutil.virtual_memory().percent))
            self.timecount+=1.0
            seconds_away = int(float(time.time() - self.timestart)*float(self.length)/float(self.timecount) - float(time.time() - self.timestart))
            # y = datetime.datetime.now() + datetime.timedelta(0,seconds_away)

            print('Time remaining: ' + str(seconds_away)+ " seconds ()")
            print("Writing CSV" + ": [" + "#"*int(progress) + "-"*int(self.length-progress) + "] "+ str(int(progress/10.0))+"%")
            # print("Writing CSV" + ": [" + "#"*int(self.timecount) + "-"*int(self.length-self.timecount) + "] "+ str(int(progress/10.0))+"%")
        self.meter = self.meter + 1
        return self
    
    def get_tick(self):
        return self.meter