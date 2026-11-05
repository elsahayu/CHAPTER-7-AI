# Example 7.13 Training Haar Cascade Code – Prepare Images
# pip install icrawler
from icrawler.builtin import BingImageCrawler
# Find some Otter images and save them in the /p directory
classes=['otter']
number=100
for c in classes:
    bing_crawler=BingImageCrawler(storage={'root_dir':f'p/'})
    bing_crawler.crawl(keyword=c,filters=None,max_num=number,offset=0)
# Find some non-Otter images and save them in the /n directory
classes=['trees','roads','cars']
number=100
for c in classes:
    print(number)
    bing_crawler=BingImageCrawler(storage={'root_dir':f'n/'})
    bing_crawler.crawl(keyword=c,filters=None,max_num=number,offset=0)
    number = number + number
