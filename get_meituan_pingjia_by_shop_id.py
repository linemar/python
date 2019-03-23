#! python3
# coding:utf-8
import requests
import time
import sys
import os
import json
import re 
import selenium.webdriver as driver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

#请求网址
url = 'https://www.meituan.com/meishi/'

browser = driver.Chrome()
browser.get('http://www.meituan.com/meishi/2373531/')
time.sleep(10)

page = browser.page_source
soup = BeautifulSoup(browser.page_source, "lxml")


#推荐菜品获取  class="recommend"

recommend_pattern = '<span>[\u4e00-\u9fa5]*<\span>'
recommend_regex  = re.compile(recommend_pattern)

divs = soup.find('div', class_ = 'recommend')
list = divs.find('div', class_='list clear')

recommends = list.find_all('span')

for recommend in recommends:
    print(recommend.string)


#获取下一页按钮
next_page_btn= soup.find('span', class_ = 'iconfont icon-btn_right')
#判断下一页按钮是否可以点击，如果可以点击则跳转到下一个继续获取评论

#while()
#翻页，获取所有评价

#获取评价
# divs = soup.find('div', class_ = 'comment')
#
# comments = divs.find_all('div', class_='desc')
# for comment in comments:
#     print(comment.string)


