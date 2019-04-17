import tensorflow as tf
import numpy as np
from yolo3.model import yolo_eval,box_iou
from yolo3.utils import letterbox_image
from functools import reduce
from timeit import default_timer as timer

AUTOTUNE = tf.data.experimental.AUTOTUNE

class MAPCallback(tf.keras.callbacks.Callback):
    def tfrecord_dataset(self,files):
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices(files)
            """data generator for fit_generator"""

            def parse(example_proto):
                feature_description = {
                    'image/encoded': tf.io.FixedLenFeature([], tf.string),
                    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/label': tf.io.VarLenFeature(tf.int64)
                }
                features = tf.io.parse_single_example(example_proto, feature_description)
                image = tf.image.decode_image(features['image/encoded'])
                image = tf.image.convert_image_dtype(image, tf.float32)
                xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
                xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
                ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
                ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
                label = tf.expand_dims(features['image/object/bbox/label'].values, 0)
                bbox = tf.concat([xmin, ymin, xmax, ymax, tf.cast(label, tf.float32)], 0)

                return image, bbox

            dataset = dataset.interleave(
                lambda file: tf.data.TFRecordDataset(file),
                cycle_length=len(files)).map(parse)

            return dataset
    """
     Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
    """
    def _voc_ap(self,rec, prec):
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

    def __init__(self,glob_path,input_shape,anchors,class_names,score=.5,iou=.5,nms=.5):
        self.input_shape=input_shape
        self.anchors = anchors
        self.class_names=class_names
        self.num_classes = len(class_names)
        self.glob_path=glob_path
        self.score=score
        self.iou=iou
        self.nms=nms

    def on_train_end(self, logs={}):
        logs = logs or {}
        files = tf.io.gfile.glob(self.glob_path)
        test_num = reduce(lambda x, y: x + y, map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
        test_dataset = self.tfrecord_dataset(files)
        true_res={}
        pred_res=[]
        idx=0
        APs={}
        start=timer()
        for image,bbox in test_dataset:
            if self.input_shape != (None, None):
                assert self.input_shape[0] % 32 == 0, 'Multiples of 32 required'
                assert self.input_shape[1] % 32 == 0, 'Multiples of 32 required'
                boxed_image, image_shape = letterbox_image(image, tuple(
                    reversed(self.input_shape)))
            else:
                height, width, _ = image.shape
                new_image_size = (width - (width % 32), height - (height % 32))
                boxed_image, image_shape = letterbox_image(image, new_image_size)
            image_data = np.expand_dims(boxed_image, 0)
            output = self.model.predict(image_data)
            out_boxes, out_scores, out_classes = yolo_eval(output, self.anchors, self.num_classes, image_shape[0:2],
                                                           score_threshold=self.score, iou_threshold=self.nms)
            if len(out_classes)>0:
                for i in range(len(out_classes)):
                    w=int(image.shape[1])
                    h=int(image.shape[0])
                    top, left, bottom, right=out_boxes[i].numpy()*([h,w]*2)
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
                    right = min(w, np.floor(right + 0.5).astype('int32'))
                    pred_res.append([idx,out_classes[i].numpy(), out_scores[i].numpy(),left,top,right,bottom])
            true_res[idx]=[]
            for item in list(np.transpose(bbox)):
                true_res[idx].append(item)
            idx+=1
        end=timer()
        print((end-start)/test_num)
        for cls in range(self.num_classes):
            pred_res_cls=[x for x in pred_res if x[1]==cls]
            if len(pred_res_cls)==0:
                continue
            true_res_cls={}
            npos=0
            for index in true_res:
                objs=[obj for obj in true_res[index] if obj[4] == cls]
                npos+=len(objs)
                bbox = np.array([x[:4] for x in objs])
                true_res_cls[index]={'bbox':bbox,
                                     'difficult':[False]*len(objs),
                                     'det': [False] * len(objs)}
            ids = [x[0] for x in pred_res_cls]
            scores = np.array([x[2] for x in pred_res_cls])
            bboxs = np.array([x[3:] for x in pred_res_cls])
            sorted_ind = np.argsort(-scores)
            bboxs = bboxs[sorted_ind,:]
            ids = [ids[x] for x in sorted_ind]

            nd = len(ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for j in range(nd):
                res=true_res_cls[ids[j]]
                bbox = bboxs[j,:].astype(float)
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
                    uni = ((bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
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
            APs[cls]=ap
        for cls in range(self.num_classes):
            if cls in APs:
                print(self.class_names[cls]+' ap: ',APs[cls])
        mAP = np.mean([APs[cls] for cls in APs])
        print('mAP: ',mAP)
        logs['mAP']=mAP

