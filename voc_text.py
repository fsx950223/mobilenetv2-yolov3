import tensorflow as tf
import argparse
import xml.etree.ElementTree as ET
import os
import threading
import glob
from io import StringIO
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if hasattr(tf, 'enable_eager_execution'):
    tf.enable_eager_execution()

parser=argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

parser.add_argument('--folder',type=str,help='parse folder path',default='./')
threads=4

FLAGS = parser.parse_args()
xmls=glob.glob(os.path.join(FLAGS.folder,'**/*.xml'),recursive=True)
chunk_num=len(xmls)//threads
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
train_file = StringIO()
val_file = StringIO()
test_file = StringIO()
nums=[0,0,0]
splits=[0.8,0.9,1]
mutex = threading.Lock()
def write_xmls(xmls):
    for xml_path in xmls:
        image_path=glob.glob(os.path.join('/'.join(xml_path.split('/')[:-2]),'**',xml_path.split('/')[-1].split('.')[0]+'.jp*g'),recursive=True)[0]
        if tf.io.gfile.exists(image_path) is not True:
            continue
        rand = tf.random.uniform([],0,1)
        xml_root=ET.parse(xml_path).getroot()
        if rand < splits[0]:
            file=train_file
            index=0
        elif rand >= splits[0] and rand < splits[1]:
            file=val_file
            index = 1
        else:
            file=test_file
            index = 2
        label=image_path
        objects=0
        for obj in xml_root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            label+=' '+' '.join([str(a) for a in b])+' '+ str(cls_id)
            objects+=1
        if objects<=0:
            continue
        label+='\n'
        mutex.acquire()
        file.write(label)
        nums[index]+=1
        mutex.release()
pool=[]
for idx in range(threads):
    thread = threading.Thread(target=write_xmls, args=[xmls[idx*chunk_num:(idx+1)*chunk_num]])
    pool.append(thread)
for thread in pool:
    thread.start()
for thread in pool:
    thread.join()
train_file.close()
val_file.close()
test_file.close()
with open ('voc_train_'+str(nums[0])+'.txt', 'w') as fd:
    train_file.seek(0)
    shutil.copyfileobj(train_file, fd)
with open ('voc_val_'+str(nums[1])+'.txt', 'w') as fd:
    val_file.seek(0)
    shutil.copyfileobj(val_file, fd)
with open ('voc_test_'+str(nums[2])+'.txt', 'w') as fd:
    test_file.seek(0)
    shutil.copyfileobj(test_file, fd)
