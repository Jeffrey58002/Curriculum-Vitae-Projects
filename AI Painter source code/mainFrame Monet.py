#%%
'''
import common module
'''
import os
import cv2
import time
import json
import faiss
import random
import threading
import numpy as np
#%%
#%%
'''
K-means in gpu
'''
class FaissKMeans:
    def __init__(self, n_clusters=100, n_init=1, max_iter=1):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   gpu = True,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]
    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1],self.cluster_centers_
'''
Multi-thread run camera query frame in real time
'''
class CamCapture:
    def __init__(self):
        self.Frame = []
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    def start(self):
        print('cam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()
    def stop(self):
        self.isstop = True
        print('cam stopped!')
    def getframe(self):
        return self.Frame
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        self.capture.release()
#%%
def fetchImg(fileName, screen_width = 1080,screen_height = 1920,rootPath = 'C:/hyxie/ai_painter_final/temp2/'):
    img = cv2.imread(rootPath+str(fileName))
    img = cv2.resize(img,(screen_width,screen_height))
    cv2.imwrite(cwd+'/cluster/resize.jpg', img)
    return img
def genIntputData(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    intputData = np.reshape(img, (img.shape[0]*img.shape[1],3))
    N = len(intputData)
    x_random = np.random.uniform(low=-1, high=1, size=N) * img.shape[0]//6
    y_random = np.random.uniform(low=-1, high=1, size=N) * img.shape[1]//7
    intputData = np.c_[ intputData, np.linspace(0, N-1, num=N)%img.shape[1] + x_random, np.linspace(0, N-1, num=N)//img.shape[1] + y_random]
    return intputData

def clustering(intputData, cluster_num = 100):
    clt = FaissKMeans(n_clusters = cluster_num)
    clt.fit(intputData)
    label,cluster_centers = clt.predict(intputData)
    return label,cluster_centers

def layoutIO(label,img,screen_width = 1920,screen_height = 1080,cluster_num = 100,path = 'C:/hyxie/ai_painter_final/cluster/'):
    template = cv2.imread(cwd+'/template.jpg')
    template = cv2.resize(template,(screen_width,screen_height))
    for i in range(cluster_num):
        template_c = template.copy()
        cluster = np.where(label == i)[0]
        cluster_X = cluster % img.shape[1]
        cluster_Y = cluster // img.shape[1]
        template_c[cluster_Y,cluster_X] = img[cluster_Y,cluster_X]
        cv2.imwrite(path+str(i)+'.jpg', template_c)
        
def clusterIO(label,cluster_centers,imageShape,fileName= 'label',cluster_num = 100):
    outputData = np.reshape(label, (imageShape[1],imageShape[0]))
    clusterDict = dict()
    gloups_size = []
    for i in range(cluster_num):
        print('cluster_centers:',cluster_centers[i])
        cluster = np.where(outputData == i)
        combined = np.vstack((cluster[0], cluster[1])).T
        clusterDict['cluster_'+str(i)] = dict({'gloup_size':0,'cluster':combined.tolist()})
        gloups_size.append([i,combined.shape[0]])
    gloups_size = sorted(gloups_size,key = lambda s:s[1],reverse=True)
    gloups_index = [0]*100 #### change this if you want to change cluster amount
    for i in range(cluster_num):
        clusterDict['cluster_'+str(i)]['gloup_size'] = gloups_size[i][1]
        gloups_index[i] = gloups_size[i][0]
    indexDict = dict({'index':gloups_size})
    genJsonData(indexDict,cwd+'/cluster/cluster_index.json')
    return clusterDict
def getTimeFstr(seconds):
    timeArray = time.localtime(seconds)
    otherStyleTime = time.strftime("%H.%M.%S",timeArray)
    return otherStyleTime
def genJsonData(outputData,pathFilename):
    with open(pathFilename, 'a') as f:
        f.truncate(0)
        json.dump(outputData, f)
    return None
#%%
from os.path import exists
from urllib.request import urlretrieve
prototxt = "deploy.prototxt"
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
# 下載模型相關檔案
if not exists(prototxt) or not exists(caffemodel):
    urlretrieve(f"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/{prototxt}",
                prototxt)
    urlretrieve(
        f"https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/{caffemodel}",
        caffemodel)
# 初始化模型 (模型使用的Input Size為 (300, 300))
net = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=caffemodel)
def detect(img, min_confidence=0.5):
    # 取得img的大小(高，寬)
    (h, w) = img.shape[:2]

    # 建立模型使用的Input資料blob (比例變更為300 x 300)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 設定Input資料與取得模型預測結果
    net.setInput(blob)
    detectors = net.forward()

    # 初始化結果
    rects = []
    # loop所有預測結果
    for i in range(0, detectors.shape[2]):
        # 取得預測準確度
        confidence = detectors[0, 0, i, 2]

        # 篩選準確度低於argument設定的值
        if confidence < min_confidence:
            continue

        # 計算bounding box(邊界框)與準確率 - 取得(左上X，左上Y，右下X，右下Y)的值 (記得轉換回原始image的大小)
        box = detectors[0, 0, i, 3:7] * np.array([w, h, w, h])
        # 將邊界框轉成正整數，方便畫圖
        (x0, y0, x1, y1) = box.astype("int")
        rects.append({"box": (x0, y0, x1 - x0, y1 - y0), "confidence": confidence})

    return rects
def modify_intensity(img):
    origin_img = img
    maxIntensity = 150.0
    phi = 1
    theta = 1
    increase_img = (maxIntensity/phi)*(origin_img/(maxIntensity/theta))**0.5
    increase_img = np.array(increase_img, dtype=np.uint8)
    return increase_img
def fetchFace():
    global isPreparing
    
    while isLoop:
        img = cap.getframe()
        if cap.status:
            # img = modify_intensity(img)
            # kernel = np.array([[0, -1, 0],
            #         [-1, 5,-1],
            #         [0, -1, 0]])
            # img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
            #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rects = detect(img, 0.5)
            faces =[]
            for rect in rects:
                faces.append(rect["box"])
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # faces = faceCascade.detectMultiScale(
            #     gray,
            #     scaleFactor=1.1,
            #     minNeighbors=30,
            #     minSize=(100, 100),
            # )
            if len(faces) > 0:
                infetchFace = time.time()
                print('Detect Face!!')
                isPreparing = True
                time.sleep(3)
                while isPreparing:
                    inPreparing = time.time() - infetchFace
                    if inPreparing > 10:
                        isPreparing = False
                        break
                    img = cap.getframe()
                    if cap.status:
                        # img = modify_intensity(img)
                        # kernel = np.array([[0, -1, 0],
                        #         [-1, 5,-1],
                        #         [0, -1, 0]])
                        # img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
                        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        rects = detect(img, 0.5)
                        faces =[]
                        for rect in rects:
                            faces.append(rect["box"])
                        if len(faces) > 0:
                            sorted_faces = sorted(list(faces),reverse=True,key=lambda s:s[2]*s[3])
                            theClosestFace = sorted_faces[0]
                            faceX = theClosestFace[0]
                            faceY = theClosestFace[1]
                            faceW = theClosestFace[2]
                            faceH = theClosestFace[3]
                            # mask = cv2.inRange(img, lower_blue, upper_blue)
                            # blueIndex = np.where(mask > 0)
                            # if len(blueIndex[0]) >0:
                            #     img[blueIndex]=(0,0,0)
                            l_X = max(faceX-int(faceW*1.0),0)
                            l_Y = max(faceY-int(faceH*1.0),0)
                            r_X = min(faceX + faceW + int(faceW*1.0),screenWidth)
                            r_Y = min(faceY + faceH + int(faceH*1.0),screenHeight)
                            faceImg = img[l_Y:r_Y,l_X:r_X]                
                            # faceImg = cv2.resize(faceImg,(int((r_X-l_X)*0.7),(r_Y-l_Y)))
                            faceImg = cv2.resize(faceImg,(screenWidth,screenHeight))
                            cv2.imwrite(cwd+'/cluster/resize.jpg', faceImg)
                            timeSpan = getTimeFstr(time.time())
                            print('timeSpan',timeSpan)
                            cv2.imwrite(cwd+'/face_log/face_'+str(timeSpan)+'.jpg',faceImg)
                            start_time = time.time()
                            intputData = genIntputData(faceImg)
                            label,cluster_centers = clustering(intputData)
                            clusterDict_all = clusterIO(label,cluster_centers,faceImg.shape)
                            genJsonData(clusterDict_all,cwd+'/cluster/cluster_all.json')
                            total_time = time.time() - start_time
                            print('total time',total_time)
                            isPreparing = False
                            os.system('cmd /c "processing-java --sketch='+currentPath+' --run"')
                            print('Finish Call AI_Painter...')
#%%
if __name__ == '__main__':
    mode = 'Image' #改成Camera讀鏡頭, Image讀圖檔
    if mode == 'Camera':
        cwd = os.getcwd()
        cwd = cwd.replace("\\", "/")
        global isPreparing
        window_name = 'AI Painter'
        screenWidth, screenHeight = 1920, 1080
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        currentPath = os.path.dirname(os.path.abspath(__file__))
        faceCascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

        isLoop = True
        isPreparing = False

        cap = CamCapture()
        cap.start()
        time.sleep(2)
        threading.Thread(target=fetchFace, daemon=True, args=()).start()
        while isLoop:
            imgsNameList = os.listdir('./ai_works')
            randSeq = random.sample(imgsNameList,len(imgsNameList))
            for imgName in randSeq:
                image = cv2.imread('./ai_works/'+imgName)
                if isPreparing:
                    image = cv2.imread('./preparing.png')
                    cv2.imshow(window_name, image)
                else:
                    cv2.imshow(window_name, image)
                key = cv2.waitKey(5000)
            if key == ord('q'):
                isLoop = False
                break
        cap.stop()
        cv2.destroyAllWindows()
    elif mode == 'Image':
        cwd = os.getcwd()
        cwd = cwd.replace("\\", "/")
        currentPath = os.path.dirname(os.path.abspath(__file__))
        image = cv2.imread('./test.png') # 圖檔路徑:改這邊的路徑執行
        image = cv2.resize(image,(1920,1080)) #畫布大小, 螢幕解析度大小
        cv2.imwrite(cwd+'/cluster/resize.jpg', image)
        intputData = genIntputData(image)
        label,cluster_centers = clustering(intputData)
        clusterDict_all = clusterIO(label,cluster_centers,image.shape)
        genJsonData(clusterDict_all,cwd+'/cluster/cluster_all.json')
        os.system('cmd /c "processing-java --sketch='+currentPath+' --run"')
        print('Finish Call AI_Painter...')
    else:
        print('Camera or Image')