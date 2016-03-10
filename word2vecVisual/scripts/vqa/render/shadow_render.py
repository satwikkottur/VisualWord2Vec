from PIL import Image, ImageFilter
import cv2
#import cv
import numpy as np
import pdb

def makeShadow(image,  offset, shadowColour='lime', iterations=20, border=0):

    fullWidth = image.size[0] + abs(offset[0]) + 2*border
    fullHeight = image.size[1] + abs(offset[1]) + 2*border
    shadow = Image.new(image.mode, (fullWidth, fullHeight), 0)
    shadowLeft = border + max(offset[0], 0) #if <0, push the rest of the image right
    shadowTop  = border + max(offset[1], 0) #if <0, push the rest of the image down

    dilate = image
    _, _, _, alpha = dilate.split()
    _, im = cv2.threshold(np.array(alpha), 0, 256, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)

    #dil = cv.fromarray(cv2.dilate(im, kernel, iterations=2))
    #mask = Image.frombytes("L", cv.GetSize(dil), dil.tostring())
    dil = cv2.dilate(im, kernel, iterations=2)
    mask = Image.frombytes("L", (dil.shape[1], dil.shape[0]), dil.tostring())

    shadow.paste(shadowColour, [shadowLeft, shadowTop,
                                shadowLeft + image.size[0], shadowTop + image.size[1]], mask)

    imgLeft = border - min(offset[0], 0) #if the shadow offset was <0, push right
    imgTop  = border - min(offset[1], 0) #if the shadow offset was <0, push down
    shadow.paste(image, (imgLeft, imgTop), image)
    return shadow
