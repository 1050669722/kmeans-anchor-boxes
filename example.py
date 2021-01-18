import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "Annotations"
CLUSTERS = 5

def load_dataset(path):
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			xmin = int(obj.findtext("bndbox/xmin")) / width
			ymin = int(obj.findtext("bndbox/ymin")) / height
			xmax = int(obj.findtext("bndbox/xmax")) / width
			ymax = int(obj.findtext("bndbox/ymax")) / height

			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
out = kmeans(data, k=CLUSTERS) #out 表示簇中心 宽高
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100)) #每个框 属于最可能簇 的iou 的平均值
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
