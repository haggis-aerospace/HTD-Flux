#!/usr/bin/python3

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def DrawTarget(char='T', size=(28,28), bgcolor=(255,0,0)):
    img=Image.new('RGB', size, color=(255,255,255))
    _img=ImageDraw.Draw(img)
    _img.rectangle([(size[0]/4, size[1]/4 ),(3*size[0]/4, 3*size[1]/4 )], fill=bgcolor)
    fnt = ImageFont.truetype('/usr/share/fonts/TTF/Arial.ttf', int(size[0]/3))
    w, h = fnt.getsize(char)
    _img.text(((size[0]-w)/2, (size[1]-h)/2), char, font=fnt, fill=(255,255,255), align="center")
    return img

def noise_generator (noise_type,image):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    image=np.array(image)
    row,col,ch= image.shape
    if noise_type == "gauss":       
        mean = 0.0
        var = 0.05
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return Image.fromarray(noisy.astype('uint8'))
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return Image.fromarray(out)
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return Image.fromarray(noisy)
    elif noise_type =="speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return Image.fromarray(noisy)
    else:
        return Image.fromarray(image)
