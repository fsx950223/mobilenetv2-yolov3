import tensorflow as tf
import argparse
import xml.etree.ElementTree as ET
import os
import threading
import glob
if hasattr(tf, 'enable_eager_execution'):
    tf.enable_eager_execution()

parser=argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

parser.add_argument('--folder',type=str,help='parse folder path',default='./')
threads=4

FLAGS = parser.parse_args()
xmls=glob.glob(os.path.join(FLAGS.folder,'**/*.xml'),recursive=True)
chunk_num=len(xmls)//threads
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
train_file = tf.io.gfile.GFile('train.txt', 'w')
val_file = tf.io.gfile.GFile('val.txt', 'w')
test_file = tf.io.gfile.GFile('test.txt', 'w')
nums=[0,0,0]
splits=[0.8,0.9,1]
mutex = threading.Lock()
def write_xmls(xmls):
    for xml_path in xmls:
        image_path=glob.glob(os.path.join('/'.join(xml_path.split('/')[:-2]),'**',xml_path.split('/')[-1].split('.')[0]+'.jp*g'),recursive=True)[0]
        rand = tf.random.uniform([],0,1)
        xml_root=ET.parse(xml_path.encode('utf-8')).getroot()
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
tf.io.gfile.rename('train.txt','voc_train_'+str(nums[0])+'.txt')
tf.io.gfile.rename('val.txt','voc_val_'+str(nums[1])+'.txt')
tf.io.gfile.rename('test.txt','voc_test_'+str(nums[2])+'.txt')
