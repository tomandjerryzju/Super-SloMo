# -*- coding: utf-8 -*-
'''
author: huangyanchun
date: 2018/10/11
'''

import os
from PIL import Image, ImageSequence
import sys

def create_webp(img_path, webp_name, min_size=320.0):
    frames = []

    # w = 800
    # h = 600
    image_list = os.listdir(img_path)
    for image_name in sorted(image_list):
        input = os.path.join(img_path, image_name)
        im = Image.open(input)
        if im.mode != 'RGB':
            im = im.convert("RGB")
        im = im.crop((0, 0, 400, 225))
        frames.append(im)
    frames[0].save(webp_name, save_all=True, append_images=frames[1:], loop=0, allow_mixed=True)


if __name__ == "__main__":
    create_webp('/Users/hyc/workspace/Super-SloMo/test_slow_1/0/', 'test_slow_1_ori.gif', min_size=320.0)