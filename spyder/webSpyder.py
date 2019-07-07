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
import bs4
import time                              # ALL

#文件保存路径
target_path = r"E:\workspace\python\webSpyder"
# 打开文件以便写入     　 

#伪装头文件
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '  
                        'Chrome/51.0.2704.63 Safari/537.36'}

#请求网址
url = "http://www.dytt8.net/html/gndy/dyzz/index.html"

def find_download_link():
    return 

if __name__ == '__main__':
    #网页请求
    req = urllib.request.Request(url=url,headers=headers)

    data = urllib.request.urlopen(url).read();
    soup = BeautifulSoup(data, 'lxml')

    #寻找电影具体页面
    #bookLists = soup.find_all(id="prd_md_by_img");
        
    address_lists = soup.find_all('a', class_ = "ulink")

    pages_list = []
    i = 0
    for address in address_lists:
        #print('http://www.dytt8.net/' + address['href'])
        pages_list.append("http://www.dytt8.net" + address['href'])

    download_links = []
    for page in pages_list:
        print(page)
        page_data = urllib.request.urlopen(page).read();
        #print(page_data)
        page_soup = BeautifulSoup(page_data, 'lxml')
        #link = page_soup.find_all('a', attrs="thunderrestitle") #!!!!!
        link = page_soup.find_all('a', attrs={"thunderrestitle":True}) #!!!!!
        print(link)
        download_links.append(link)
        time.sleep(3)
        #print(page_data)
        break 

#        for download_link in download_links:
#            print(download_link)
    