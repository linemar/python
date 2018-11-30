# encoding:UTF-8
import urllib.request
import socket
import re
import sys
import os
import scrapy
from html.parser import HTMLParser
from bs4 import BeautifulSoup           # HTML
from bs4 import BeautifulStoneSoup      # XML
import bs4                              # ALL

#伪装头文件
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '  
                        'Chrome/51.0.2704.63 Safari/537.36'}
#请求网址
url = 'http://www.dianping.com/changsha/ch10/r6027'
#网页请求
request = urllib.request.Request(url=url,headers=headers)
respose = urllib.request.urlopen(request).read();
soup = BeautifulSoup(respose, 'lxml')
#soup = BeautifulSoup(respose)


content = soup.findAll(attrs={"class":"tit"})
#print(content)

list = []
time = 0
for i in content:
    #print(i)
    #print(time)
    #time = time + 1
    #print(i.a)
    if i.a:
        list.append(i.a['data-shopid'])
        print(i.a['data-shopid'])

#for i in list:
   #print(i)

