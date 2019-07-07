#! python3
# coding:utf-8
import requests
import time
import sys
import os
import json
import re 
import selenium.webdriver as driver
from bs4 import BeautifulSoup
import traceback
import selenium.common

#请求网址
url = 'https://www.meituan.com/meishi/'

browser = driver.Chrome()


#根据商店id获取其评价
def get_comments_by_id(shop_id):
    pass




browser.get('http://www.meituan.com/meishi/2373531/')
time.sleep(10)

page = browser.page_source
soup = BeautifulSoup(browser.page_source, "lxml")


#推荐菜品获取  class="recommend"
recommend_pattern = '<span>[\u4e00-\u9fa5]*<\span>'
recommend_regex = re.compile(recommend_pattern)

divs = soup.find('div', class_='recommend')
list = divs.find('div', class_='list clear')

recommends = list.find_all('span')

for recommend in recommends:
    print(recommend.string)

#browser.find_element_by_xpath("//ul[@class='pagination clear']/li/span[contains(text(),'2')]").click()

current_node = browser.find_elements_by_xpath("//ul[@class='pagination clear']/li/span")
for node in current_node:
    print(node.text)

page_count = current_node[6].text


i = 1
#获取所有评价
while(1):

    # 获取网页源码
    #browser.get('http://www.meituan.com/meishi/2373531/')
    soup = BeautifulSoup(browser.page_source, "lxml")
    page_num = str(i)

    #browser.find_element_by_xpath("//ul[@class='pagination clear']/li/span[contains(text(), page_num)]").click()

    divs = soup.find('div', class_ = 'comment')

    print('当前是：' + str(i) + '页')
    comments = divs.find_all('div', class_='desc')
    for comment in comments:
        print(comment.string)

    try:

        next_node = browser.find_element_by_xpath(
            "//ul[@class='pagination clear']/li/span[@class='iconfont icon-btn_right']").click()

        time.sleep(10)

    except selenium.common.exceptions.WebDriverException:

        msg = traceback.format_exc()
        print(msg)

        print('最后一页 ！')
        next_node = browser.find_element_by_xpath(
            "//ul[@class='pagination clear']/li/span[@class='iconfont icon-btn_right disabled']")

        exit()

    except:

        msg = traceback.format_exc()
        print(msg)

        exit()


    else:
        i = i + 1



