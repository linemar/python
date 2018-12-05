import requests
import os
import json

#伪装头文件
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/63.0.3239.26 Safari/537.36 \
            Core/1.63.5702.400 QQBrowser/10.2.1893.400'}

#print('headers : ' + headers)

#file_name = 'shop_list_page'
for i in range (1, 9):
    f = open('shop_list_page' + str(i) + '.txt', 'r', encoding='utf-8')  #默认打开模式就为r
    data = f.read()
    json_data = json.loads(str(data))
    print('page : ' + str(i))
    for index in range (0, 31):
        print(json_data['data']['poiInfos'][index]['poiId'])