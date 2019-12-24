import tensorflow as tf
import numpy as np
from yolo3.utils import bind
from timeit import default_timer as timer
from yolo3.data import Dataset
from yolo3.enums import DATASET_MODE

AUTOTUNE = tf.data.experimental.AUTOTUNE


class MAPCallback(tf.keras.callbacks.Callback):
    """Calculate the AP given the recall and precision array 1st) We compute a
    version of the measured precision/recall curve with precision monotonically
    decreasing 2nd) We compute the AP as the area under this curve by numerical
    integration."""

    def _voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def parse_tfrecord(self, example_proto):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/label': tf.io.VarLenFeature(tf.int64)
        }
        features = tf.io.parse_single_example(example_proto,
                                              feature_description)
        image = features['image/encoded']
        xmins = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        xmaxs = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymins = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        ymaxs = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
        labels = tf.expand_dims(features['image/object/bbox/label'].values, 0)
        bbox = tf.concat([xmins, ymins, xmaxs, ymaxs,
                          tf.cast(labels, tf.float32)], 0)
        return image, tf.transpose(bbox)

    def parse_text(self, line):
        values = tf.strings.split([line], ' ').values
        image = tf.io.read_file(values[0])
        reshaped_data = tf.reshape(values[1:], [-1, 5])
        xmins = tf.strings.to_number(reshaped_data[:, 0], tf.float32)
        xmaxs = tf.strings.to_number(reshaped_data[:, 2], tf.float32)
        ymins = tf.strings.to_number(reshaped_data[:, 1], tf.float32)
        ymaxs = tf.strings.to_number(reshaped_data[:, 3], tf.float32)
        labels = tf.strings.to_number(reshaped_data[:, 4], tf.int64)
        xmins=tf.expand_dims(xmins, 0)
        xmaxs=tf.expand_dims(xmaxs, 0)
        ymins=tf.expand_dims(ymins, 0)
        ymaxs=tf.expand_dims(ymaxs, 0)
        labels=tf.expand_dims(labels, 0)
        
        bbox = tf.concat(
            [xmins, ymins, xmaxs, ymaxs,
             tf.cast(labels, tf.float32)], 0)
        return image, tf.transpose(bbox)

    def calculate_aps(self):
        test_dataset_builder = Dataset(
            self.glob_path,
            self.batch_size,
            input_shape=self.input_shape,
            mode=DATASET_MODE.TEST)
        bind(test_dataset_builder, self.parse_tfrecord)
        bind(test_dataset_builder, self.parse_text)
        test_dataset, test_num = test_dataset_builder.build()
        true_res = {}
        pred_res = []
        idx = 0
        APs = {}
        start = timer()
        for image, bbox in test_dataset:
            out_boxes, out_scores, out_classes = self.model(image)
            out_boxes = out_boxes.numpy()
            out_scores = out_scores.numpy()
            out_classes = out_classes.numpy()
            if len(out_classes) > 0:
                for out_box, out_score, out_class in zip(
                        out_boxes, out_scores, out_classes):
                    top, left, bottom, right = out_box
                    pred_res.append(
                        [idx, out_class, out_score, left, top, right, bottom])
            true_res[idx] = bbox[0].numpy()
            idx += 1
        end = timer()
        print((end - start) / test_num)
        for cls in range(self.num_classes):
            pred_res_cls = [x for x in pred_res if x[1] == cls]
            if len(pred_res_cls) == 0:
                APs[cls]=0
                continue
            true_res_cls = {}
            npos = 0
            for index in true_res:
                objs = [obj for obj in true_res[index] if obj[4] == cls]
                npos += len(objs)
                BBGT = np.array([x[:4] for x in objs])
                true_res_cls[index] = {
                    'bbox': BBGT,
                    'difficult': [False] * len(objs),
                    'det': [False] * len(objs)
                }
            ids = [x[0] for x in pred_res_cls]
            scores = np.array([x[2] for x in pred_res_cls])
            bboxs = np.array([x[3:] for x in pred_res_cls])
            sorted_ind = np.argsort(-scores)
            bboxs = bboxs[sorted_ind, :]
            ids = [ids[x] for x in sorted_ind]

            nd = len(ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for j in range(nd):
                res = true_res_cls[ids[j]]
                bbox = bboxs[j, :].astype(float)
                ovmax = -np.inf
                BBGT = res['bbox'].astype(float)
                if BBGT.size > 0:
                    ixmin = np.maximum(BBGT[:, 0], bbox[0])
                    iymin = np.maximum(BBGT[:, 1], bbox[1])
                    ixmax = np.minimum(BBGT[:, 2], bbox[2])
                    iymax = np.minimum(BBGT[:, 3], bbox[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.)
                           + (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                if ovmax > self.iou:
                    if not res['difficult'][jmax]:
                        if not res['det'][jmax]:
                            tp[j] = 1.
                            res['det'][jmax] = 1
                        else:
                            fp[j] = 1.
                else:
                    fp[j] = 1.

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec)
            APs[cls] = ap
        return APs

    def __init__(self,
                 glob_path,
                 input_shape,
                 class_names,
                 iou=.5,
                 batch_size=1):
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.glob_path = glob_path
        self.iou = iou
        self.batch_size = batch_size

    def on_train_end(self, logs={}):
        logs = logs or {}
        origin_learning_phase = tf.keras.backend.learning_phase()
        tf.keras.backend.set_learning_phase(0)
        APs = self.calculate_aps()
        tf.keras.backend.set_learning_phase(origin_learning_phase)
        for cls in range(self.num_classes):
            if cls in APs:
                print(self.class_names[cls] + ' ap: ', APs[cls])
        mAP = np.mean([APs[cls] for cls in APs])
        print('mAP: ', mAP)
        logs['mAP'] = mAP
