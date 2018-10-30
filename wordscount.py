import re # 正则表达式库
import collections # 词频统计库
import numpy as np # numpy数据处理库
import jieba # 结巴分词
import wordcloud # 词云展示库
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt # 图像展示库


import string

class Word():
    word = None
    count = 0

class WordsCount():
    text = None
    wordlist = []

    def get_word_list(self):
        seg_list = jieba.lcut(self.text)  # 默认是精确模式
        for i in seg_list:
            print(i)
            if i in self.wordlist:
                self.wordlist[self.wordlist.index(i)].count = self.wordlist[self.wordlist.index(i)].count + 1
            elif i not in self.wordlist:
                word = Word()
                word.word = i
                self.wordlist.append(word)

        return self.wordlist


text = '从创立之初，百度' \
       '\ 在面对用户的搜索产品不断丰富的同时，百度还创新性地推出了基于搜索的营销推广服务，并成为最受企业青睐的互联网营销推广平台。目前，中国已有数十万家企业使用了百度的搜索推广服务，不断提升着企业自身的品牌及运营效率。'
'\ 为推动中国数百万中小网站的发展，百度借助超大流量的平台优势，联合所有优质的各类网站，建立了世界上最大的网络联盟，使各类企业的搜索推广、品牌营销的价值、覆盖面均大面积提升。与此同时，各网站也在联盟大家庭的互助下，获得最大的生存与发展机会。'
'\ 移动互联网时代来临，百度在业界率先实现移动化转型，迎来更为广阔的发展机遇。通过开放地连接传统行业的3600行，百度从“连接人和信息”延伸到“连接人和服务”，让网民直接通过百度移动产品获得服务。目前，百度正通过持续的商业模式和产品、技术创新，推动金融、医疗、教育、汽车、生活服务等实体经济的各行业与互联网深度融合发展，为推动经济创新发展，转变经济发展方式发挥积极作用。'
'\ 作为一家以技术为信仰的高科技公司，百度将技术创新作为立身之本，着力于互联网核心技术突破与人才培养，在搜索、人工智能、云计算、大数据等技术领域处于全球领先水平。百度认为，互联网发展正迎来第三幕——人工智能，这也是百度重要的技术战略方向。百度建有世界一流的研究机构——百度研究院，广揽海内外顶尖技术英才，致力于人工智能等相关前沿技术的研究与探索，着眼于从根本上提升百度的信息服务水平。目前，百度人工智能研究成果已全面应用于百度产品，让数亿网民从中受益;同时，百度还将语音、图像、机器翻译等难度高、投入大的领先技术向业界开放，以降低大众创业、万众创新的门槛，进一步释放创业创新活力。'
'\ 作为国内的一家知名企业，百度也一直秉承“弥合信息鸿沟，共享知识社会”的责任理念，坚持履行企业公民的社会责任。成立来，百度利用自身优势积极投身公益事业，先后投入巨大资源，为盲人、少儿、老年人群体打造专门的搜索产品， 解决了特殊群体上网难问题，极大地弥补了社会信息鸿沟问题。此外，在加速推动中国信息化进程、净化网络环境、搜索引擎教育及提升大学生就业率等方面，百度也一直走在行业领先的地位。2011年初，百度还捐赠成立百度基金会，围绕知识教育、环境保护、灾难救助等议题，更加系统规范地管理和践行公益事业。'
'\ 今天，百度已经成为中国最具价值的品牌之一。在2016年MIT Technology Review 《麻省理工科技评论》评选的全球最聪明50家公司中，百度的排名超越其他科技公司高踞第二。而“亚洲最受尊敬企业”、“全球最具创新力企业”、“中国互联网力量之星”等一系列荣誉称号的获得，也无一不向外界展示着百度成立数年来的成就。'
'\ 百度从不满足于自身取得的成绩，也从未停止发展的步伐，自2005年在纳斯达克上市以来，截至2015年，百度的市值已达 800亿美元。如今，百度已经发展成一家国际性企业，在日本、巴西、埃及中东地区、越南、泰国、印度尼西亚建立分公司， 未来，百度将覆盖全球50%以上的国家，为全球提供服务。'
'\ 多年来，百度董事长兼CEO李彦宏，率领百度人所形成的“简单可依赖”的核心文化，深深地植根于百度。这是一个充满朝气、求实坦诚的公司，以技术改变生活，推动人类的文明与进步，促进中国经济的发展为己任，正朝着更为远大的目标而迈进。'
'\ 虫的介绍出发，引入一个简单爬虫的技术架构，然后通过是什么、怎么做、现场演示三步骤，解释爬虫技术架构中的三个模块。最后，一套优雅精美的爬虫代码实战编写，向大家演示了实战抓取百度百科1000个页面的数据全过程'

wordcount = WordsCount()
wordcount.text = text
wordcount.get_word_list()


seg_list_exact = jieba.cut(text, cut_all = False) # 精确模式分词
object_list = []
remove_words = [u'的', u'，',u'和', u'是', u'随着', u'对于', u'对',u'等',u'能',u'都',u'。',u' ',u'、',u'中',u'在',u'了',
                u'通常',u'如果',u'我们',u'需要'] # 自定义去除词库

for word in seg_list_exact: # 循环读出每个分词
    if word not in remove_words: # 如果不在去除词库中
        object_list.append(word) # 分词追加到列表

# 词频统计
word_counts = collections.Counter(object_list) # 对分词做词频统计
word_counts_top10 = word_counts.most_common(10) # 获取前10最高频的词
print (word_counts_top10) # 输出检查

# 词频展示
#mask = np.array(Image.open('wordcloud.jpg')) # 定义词频背景
wc = wordcloud.WordCloud(font_path="msyh.ttc",background_color="black",max_words=1000,max_font_size=100,
              width=1500,height=1500)

wc.generate_from_frequencies(word_counts) # 从字典生成词云
#image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
wc.recolor() # 将词云颜色设置为背景图方案
plt.imshow(wc) # 显示词云
plt.axis('off') # 关闭坐标轴
plt.show() # 显示图像