# -*- coding: utf-8 -*-
'''
author: huangyanchun
date: 2018/10/12
'''

import os
from PIL import Image, ImageSequence
import numpy as np

def read_webp(webpPath):
    im = Image.open(webpPath)
    iter = ImageSequence.Iterator(im)
    index = 0
    for frame in iter:
        frame = frame.convert('RGB')
        # print np.array(frame).shape
        print("image %d: mode %s, size %s" % (index, frame.mode, frame.size))
        frame.save("/Users/hyc/workspace/Super-SloMo/test_slow_1/0/{}.jpg".format(str(index).zfill(4)))
        index += 1

    return


def main():
    webpPath = "b392be57d1bbd9f26b9fca212f7ab0c23204.gif"
    read_webp(webpPath)


if __name__ == "__main__":
    main()