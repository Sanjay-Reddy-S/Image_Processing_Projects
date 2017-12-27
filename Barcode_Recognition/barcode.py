import numpy as np
import cv2

#HORIZONTAL BARCODES ONLY!!

def barcode(img_name):
    image=cv2.imread(img_name)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray_image",gray)
    #cv2.waitKey(0)

    x_grad=cv2.Sobel(gray,ddepth=cv2.cv.CV_32F,dx=1,dy=0,ksize=-1) #horizontal
    y_grad=cv2.Sobel(gray,ddepth=cv2.cv.CV_32F,dx=0,dy=1,ksize=-1) #vertical

    grad=cv2.subtract(x_grad,y_grad) #FROM x-grad subtract y-grad
    grad=cv2.convertScaleAbs(grad)

    avg_img=cv2.blur(grad,(9,9))
    (_,thres)=cv2.threshold(avg_img,225,225,cv2.THRESH_BINARY)

    mask=cv2.getStructuringElement(cv2.MORPH_RECT,(21,7)) #bigger width than height (close gaps b/w vertical stripes)
    closed_img=cv2.morphologyEx(thres,cv2.MORPH_CLOSE,mask)

    closed_img=cv2.erode(closed_img,None,iterations=4)
    closed_img=cv2.dilate(closed_img,None,iterations=4)

    (cnts,_)=cv2.findContours(closed_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=sorted(cnts,key=cv2.contourArea,reverse=True)[0]

    rect=cv2.minAreaRect(contour)
    box=np.int0(cv2.cv.BoxPoints(rect))

    cv2.drawContours(image,[box],-1,(0,255,0),2)
    cv2.imshow("Image",image)
    cv2.waitKey(0) #Press 0 to exit
    #cv2.imwrite(img_name[:10]+"_ans.jpg",image)
    return


barcode('barcode_01.jpg')
barcode('barcode_02.jpg')
barcode('barcode_03.jpg')
barcode('barcode_04.jpg')
barcode('barcode_05.jpg')
barcode('barcode_06.jpg')
barcode('barcode_07.jpg')
