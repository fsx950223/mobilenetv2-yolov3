import numpy as np
import matplotlib.pyplot as plt
from yolo3.utils import bind
import tensorflow as tf
from yolo3.data import Dataset
from yolo3.enum import DATASET_MODE

if hasattr(tf, 'enable_eager_execution'):
    tf.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE


class YOLO_Kmeans:

    def parse_tfrecord(self, example_proto):
        feature_description = {
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)
        }
        features = tf.io.parse_single_example(example_proto,
                                              feature_description)
        xmins = features['image/object/bbox/xmin'].values
        xmaxs = features['image/object/bbox/xmax'].values
        ymins = features['image/object/bbox/ymin'].values
        ymaxs = features['image/object/bbox/ymax'].values
        return xmins, xmaxs, ymins, ymaxs
    def parse_text(self, line):
        values = tf.strings.split([line],' ').values
        reshaped_data = tf.reshape(values[1:], [-1, 5])
        xmins = tf.strings.to_number(reshaped_data[:, 0], tf.float32)
        xmaxs = tf.strings.to_number(reshaped_data[:, 2], tf.float32)
        ymins = tf.strings.to_number(reshaped_data[:, 1], tf.float32)
        ymaxs = tf.strings.to_number(reshaped_data[:, 3], tf.float32)
        return xmins, xmaxs, ymins, ymaxs

    def __init__(self, cluster_number, glob_path):
        self.cluster_number = cluster_number
        self.glob_path = glob_path

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(box_number, k,
                                          replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster],
                    axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        train_dataset_builder = Dataset(self.glob_path,
                                        1,
                                        mode=DATASET_MODE.TEST)
        bind(train_dataset_builder, self.parse_tfrecord)
        bind(train_dataset_builder, self.parse_text)
        train_dataset, train_num = train_dataset_builder.build()
        result = []
        for xmins, xmaxs, ymins, ymaxs in train_dataset:
            width = xmaxs - xmins
            height = ymaxs - ymins
            wh = np.transpose(np.concatenate([width, height], 0))
            result.append(wh)
        return np.concatenate(result, 0).astype(np.int32)

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        plt.scatter(all_boxes[:1000, 0], all_boxes[:1000, 1], c='r')
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        plt.scatter(result[:, 0], result[:, 1], c='b')
        plt.show()
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    glob = "../pascal/VOCdevkit/train/*2007*.tfrecords"
    kmeans = YOLO_Kmeans(cluster_number, glob)
    kmeans.txt2clusters()
