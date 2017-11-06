from PIL import ImageFont, Image, ImageDraw

with open('chinesechars.txt') as to_read:chinese_character = to_read.read().strip()
font_size = 28
font = ImageFont.truetype('经典粗黑简.TTF', font_size)
count = 0
import os
import uuid
import numpy as np
from skimage import io,transform

for m_char in chinese_character:
    scene_text = Image.new('RGBA', (font_size, font_size),(0,0,0))
    draw = ImageDraw.Draw(scene_text)
    draw.text((0, 0), m_char, fill=(255,255,255), font=font)
    dir_name = '../../chinese_character_pics/%s'%m_char
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if m_char.islower() or m_char.isupper() or m_char.isdigit():
        processed = np.array(scene_text)
        for i in range(font_size):
            # if any row contains value
            if sum(processed[:,-i-1,0]) > 0:
                processed = processed[:,:-i,:]
                processed = transform.resize(processed,(font_size,font_size))
                break
        io.imsave('../../chinese_character_pics/%s/%s.png'%(m_char,uuid.uuid4()),processed)
    else:
        scene_text.save('../../chinese_character_pics/%s/%s.png'%(m_char,uuid.uuid4()))

    count += 1