import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0]) #clusters中所有框的宽高，与box宽高比较，取最小 #array([0.00085449, 0.00085449, 0.00085449, 0.00085449, 0.00085449])
    y = np.minimum(clusters[:, 1], box[1]) #clusters中所有框的宽高，与box宽高比较，取最小 #array([0.0015, 0.0015, 0.0015, 0.0015, 0.0015])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0: #x为0或者y为0的元素存在 #这一句的意义是什么呢？ #而且x == 0, y == 0返回的是bool类型
        raise ValueError("Box has no area") #宽高有0的，报错

    intersection = x * y #n行1列 #这个IoU好像有问题，最小的宽，最小的高，这样乘出来没什么意义啊 #这些框又不一定能相交 #这样的结果无非就是最小面积占最大面积的比值
    box_area = box[0] * box[1] #这个box的面积
    cluster_area = clusters[:, 0] * clusters[:, 1] #clusters中每一个box的面积

    iou_ = intersection / (box_area + cluster_area - intersection) #所以，这是一种判定两个框的形状尺寸是否接近的判据

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median): #计算沿指定轴的中位数，返回数组元素的中位数 #如果元素个数是偶数的话，好像会取平均
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0] #(多少个box对应多少行, 宽高两列)

    distances = np.empty((rows, k)) #(多少个box对应多少行, k表示聚成多少类)
    last_clusters = np.zeros((rows,)) #每一个box对应一个0？

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)] #从a中，按照size的shape提取样本，不重复采样 #从所有box中不重复地选出k个来

    while True:
        for row in range(rows): #对于rows中的每一行，AKA对于boxes中的每一个框，计算其与clusters中的每一个框的iou，所以这个distances至少是二维的
            distances[row] = 1 - iou(boxes[row], clusters) #交并比与1取补，这是为什么？ #使增减性相反 #iou越大，距离越小

        nearest_clusters = np.argmin(distances, axis=1) #每一个框与clusters中的5个框中的哪一个最接近 #distances表示每一个框与clusters中的5个框的距离

        if (last_clusters == nearest_clusters).all(): #如果全部等于上一轮簇中心，break #这是一个迭代过程，last_clusters也在while True中每轮更新
            break #每一个框属于哪一个簇不再改变了，break

        for cluster in range(k): #每个簇中心更新一下
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0) #与clusters中的这个cluster最接近的那些框的 宽高 分别中位数 #作为新的簇中心

        last_clusters = nearest_clusters #这是一个迭代过程，last_clusters也在while True中每轮更新 #相对于上一轮的上一轮簇中心

    return clusters #返回簇中心的 宽高
