# -*- coding: utf-8 -*-
import requests
import json


# 用户评价
def getCommodityComments(url):
    if url[url.find('id=') + 14] != '&':
        id = url[url.find('id=') + 3:url.find('id=') + 15]
    else:
        id = url[url.find('id=') + 3:url.find('id=') + 14]

    doc = open('taobao.txt', 'w', encoding='utf-8')
    url = 'https://rate.taobao.com/feedRateList.htm?auctionNumId=' + id + '&currentPageNum=1'
    res = requests.get(url)
    jc = json.loads(res.text.strip().strip('()'))
    max = jc['total']
    users = []
    comments = []
    count = 0
    page = 1
    print('该商品共有评论' + str(max) + '条,具体如下: loading...')
    while count < max:
        res = requests.get(url[:-1] + str(page))
        if page > 100:
            break
        page = page + 1
        jc = json.loads(res.text.strip().strip('()'))
        jc = jc['comments']
        for j in jc:
            users.append(j['user']['nick'])
            comments.append(j['content'])
            if comments[count] == '此用户没有填写评价。':  # 去掉没有评价
                pass
            else:
                # print(count + 1, '>>', users[count], '\n        ', comments[count], file=doc)
                print(comments[count], file=doc)
            count += 1


getCommodityComments(
    'https://detail.tmall.com/item.htm?spm=a230r.1.14.1.3427494fFekrpi&id=42758503467&ns=1&abbucket=10')
