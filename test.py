import requests

# '192.168.99.100' - works for my Windows, use instead 'localhost' 
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# https://www.growjoy.com/store/pc/catalog/sungold_heirloom_cherry_tomato_plant__1926_detail.png
# https://www.seeds-gallery.shop/8876-large_default/semena-tomatov-roma.jpg
# https://www.123seeds.com/media/catalog/product/cache/2/thumbnail/488x/9df78eab33525d08d6e5fb8d27136e95/s/u/supersweet_100_488_.jpg

data = {'url': 'https://cdn.shopify.com/s/files/1/0871/0950/products/beefsteakpic.jpg'}

result = requests.post(url, json=data).json()
print(result)
