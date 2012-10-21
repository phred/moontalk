import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv

classifier = cv.Load('/usr/local/Cellar/opencv/2.4.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
storage = cv.CreateMemStorage()

def detect_faces(src, dest):
    scale = dest.width / src.width

    objs = cv.HaarDetectObjects(src, classifier, storage, min_size=(100,100))
    for (x,y,w,h), n in objs:
        cv.Rectangle(src, (x,y), (x+w,y+h), 255)
        x *= scale
        y *= scale
        w *= scale
        h *= scale
        cv.Rectangle(dest, (x,y), (x+w,y+h), 255)


cv.NamedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
camera_index = 0
capture = cv.CaptureFromCAM(camera_index)

def downsample(img):
   cv.Flip(img, flipMode=1)
   grey = cv.CreateImage((img.width, img.height), 8, 1)
   small = cv.CreateImage((img.width / 2, img.height / 2), 8, 1)
   cv.CvtColor(img, grey, cv.CV_RGB2GRAY)
   cv.Resize(grey, small, cv.CV_INTER_LINEAR)
   cv.EqualizeHist(small, small)
   return small

def detect_loop():
  frame = cv.QueryFrame(capture)
  clean = downsample(frame)

  detect_faces(clean, frame)
  cv.ShowImage("w1", frame)
  cv.ShowImage("w2", clean)

def run():
  while True:
      frame = cv.QueryFrame(capture)  
      cv.ShowImage("w1", frame)

def run2():
    while True:
        detect_loop()

