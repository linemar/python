#! python3
# coding:utf-8
import requests
import time
import sys
import os
import json

#伪装头文件
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.5702.400 QQBrowser/10.2.1893.400'}

#请求网址
url_start = 'https://chs.meituan.com/meishi/api/poi/getPoiList? \
             cityName=%E9%95%BF%E6%B2%99&cateId=0&areaId=0&sort=&dinnerCountAttrId=&page='

url_end = '&userId=&uuid=0a08613b-495f-4093-a620-d318e8055b77&platform=1& \
           partner=126&originUrl=https%3A%2F%2F \
           chs.meituan.com%2Fmeishi%2Fpn2%2F&riskLevel=1&optimusCode=1&_ \
           token=eJyFT8luo0AQ%2FZc%2BI%2FcCGLCUA9gwgIkTLzHE0Rywwey4TTcmZj \
           T%2FnraUHOY0Uklvqaenqj%2Bg8xIwwwgZCEnglnZgBvAETaZAApyJjaooCBMBU5 \
           lI4PSvp02Fd%2Bz2CzD7UDRZMnT0%2B2FshP7ABkESRg%2Frh6uCE0XMI%2BWJEMg \
           5p2wG4SlnkyYteB%2B3k9OlgYKzvIC0JVBc8p8UEHXNTtQJrL4x%2Fkb%2Bo5%2FFZ \
           6KKFVkrWOoPdVnyfhjN9eYFFhnFz9vOr8h6Hqyd0PR89Mr8Y4hZUFg6rzNrqDaxls9h \
           Rqx2Gdh2hbvPQm%2FWaBh6zsoDNXXL8F45tC5nWU3L8SaTS6OprlsFRC%2FcBb1acyey1 \
           bJZueGytJL7wVD4O%2Fd2dsrpfc%2Fdg1%2FpK0ZfHKQE8bY8atvlyY6O9S%2BThjWTb%2B \
           %2BX7j4Ph%2BV92iTubbHAQ0B1q9i%2FaQpXq7GOkizv1H40oh455NruGuSct6trGhO3NYcoO \
           qfamMImUX0ze3oCf78AMkGcXQ%3D%3D'
#urls=['http://jandan.net/pic/page-{}#comments'.format(i) for i in range(1,100)]#这个列表包含了1-99页的地址

for i in range (1, 10):
    url = url_start + str(i)+ url_end
    #网页请求
    request = requests.get(url=url, headers=headers)
    html = request.text
    #response = urllib.request.urlopen(request).read()
    print(html)
    datas = json.loads(html)
    fp = open('shop_list_page' + str(i) + '.txt', 'w') #直接打开一个文件，如果文件不存在则创建文件
    fp.write(html)
    fp.flush()
    fp.close()
    #print(datas['data'])
    time.sleep(5)


