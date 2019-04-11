import tensorflow as tf
import numpy as np
from yolo3.model import yolo_eval,box_iou
from yolo3.utils import letterbox_image
from functools import reduce
AUTOTUNE = tf.data.experimental.AUTOTUNE
def tfrecord_dataset(files,train: bool = True):
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

            xmin=tf.expand_dims(features['image/object/bbox/xmin'].values,0)
            xmax=tf.expand_dims(features['image/object/bbox/xmax'].values,0)
            ymin = tf.expand_dims(features['image/object/bbox/ymin'].values,0)
            ymax = tf.expand_dims(features['image/object/bbox/ymax'].values,0)
            label=tf.expand_dims(features['image/object/bbox/label'].values,0)
            bbox = tf.concat([xmin, ymin, xmax, ymax, tf.cast(label, tf.float32)], 0)
            image=tf.image.decode_image(features['image/encoded'])
            image=tf.image.convert_image_dtype(image,tf.float32)
            return image,bbox

        if train:
            train_num = reduce(lambda x, y: x + y,
                               map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
            dataset = dataset.interleave(
                lambda file: tf.data.TFRecordDataset(file),
                cycle_length=len(files),num_parallel_calls=AUTOTUNE).shuffle(train_num).map(parse, num_parallel_calls=AUTOTUNE).repeat()

        else:
            dataset = dataset.interleave(
                lambda file: tf.data.TFRecordDataset(file),
                cycle_length=len(files),num_parallel_calls=AUTOTUNE).map(parse, num_parallel_calls=AUTOTUNE).repeat()

        return dataset
class MAPCallback(tf.keras.callbacks.Callback):

    """
     Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
    """
    def _voc_ap(rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
         This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #     range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #     range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
         This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
         The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap

    def __init__(self,glob_path,input_shape,anchors,num_classes,minoverlap=0.5):
        self.input_shape=input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.glob_path=glob_path
        self.minoverlap=minoverlap

    def on_train_end(self, logs={}):
        files = tf.io.gfile.glob(self.glob_path)
        test_num = reduce(lambda x, y: x + y, map(lambda file: int(file.split('/')[-1].split('.')[0].split('_')[3]), files))
        test_dataset = tfrecord_dataset(files, 1, train=False)
        true_res={}
        pred_res=[]
        idx=0
        APs={}
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
            out_boxes, out_scores, out_classes = yolo_eval(output, self.anchors, len(self.class_names), image_shape[0:2],
                                                           score_threshold=self.score, iou_threshold=self.nms)
            for i in len(out_classes):
                pred_res.append(str(idx)+' '+str(out_classes[i])+' '+out_scores[i]+' '+' '.join([str(val) for val in out_boxes[i]])+'\n')
            true_res[idx]={'bbox':bbox}
            idx+=1
        for cls in range(self.num_classes):
            true_res_cls={idx:true_res[idx]['bbox'] for idx in true_res if true_res[idx]['bbox'][4]==cls}
            splitlines = [x.strip().split(' ') for x in pred_res]
            ids = [x[0] for x in splitlines]
            scores = np.array([float(x[2]) for x in splitlines])
            bboxs = np.array([[float(z) for z in x[3:]] for x in splitlines])

            sorted_ind = np.argsort(-scores)
            bboxs = bboxs[sorted_ind, :]
            ids = [ids[x] for x in sorted_ind]
            tp = [0]*test_num
            fp = [0]*test_num
            for j in range(test_num):
                res=true_res_cls[ids[j]]
                bbox = bboxs[j, :].astype(float)
                ovmax = -np.inf
                BBGT = res['bbox'].astype(float)
                if BBGT.size > 0:
                    iou=box_iou(bbox,BBGT)
                    ovmax = np.max(iou)
                    jmax = np.argmax(iou)
                if ovmax > self.minoverlap:
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
            rec = tp / float(test_num)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec)
            APs[cls]=ap
        mAP = np.mean([APs[cls] for cls in APs])
        logs.update('mAP',mAP)

