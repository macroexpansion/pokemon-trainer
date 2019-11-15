# terminal
# pip install google_images_download
# 
# python 3.6.x

from google_images_download import google_images_download as gg

response = gg.googleimagesdownload()

arguments = {
    'keywords': 'pikachu',
    'limit': 10,
    'print_urls': False,
    'format': 'jpg, png',
}

paths = response.download(arguments)
print(paths)