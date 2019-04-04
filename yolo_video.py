import numpy as np
import argparse
from yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer
import tensorflow as tf
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        if tf.executing_eagerly():
            content = tf.io.read_file(img,'rb')
            image = tf.image.decode_image(content)
            image = tf.image.convert_image_dtype(image, tf.float32)
        else:
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
        r_image = yolo.detect_image(image)
        r_image.show()
    yolo.close_session()

def test_img(yolo):
    while True:
        list = input('Input image list:')
        file=open(list)
        class_names=yolo._get_class()
        records=[]
        start = timer()
        for name in class_names:
            record = open(name+'.txt','a')
            records.append(record)
        line=file.readline()
        while line:
            line=line.split()[0]
            try:
                image = Image.open(line)
            except:
                print('Open Error! Try again!')
                continue
            else:
                out_boxes, out_scores, out_classes = yolo.detect_image_test(image)
                if out_classes:
                    for i, c in out_classes:
                        box = out_boxes[i]
                        score = out_scores[i]
                        top, left, bottom, right = box
                        top = max(0, np.floor(top + 0.5).astype('int32'))
                        left = max(0, np.floor(left + 0.5).astype('int32'))
                        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                        records[c].write("%s %s %s %s %s %s %s\n"%(line,c,score, left, top, right, bottom))
            image.close()
            line = file.readline()
        end=timer()
        print(end-start)
        for record in records:
            record.close()
        file.close()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    parser.add_argument(
        '--export', default=False, action="store_true",
        help='Export hdf5 model to serving model'
    )
    parser.add_argument(
        '--test', default=False, action="store_true",
        help='Image test mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )
    parser.add_argument(
        "--export_path", nargs='?', type=str, required=False, default='./test',
        help="Serving model save path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif FLAGS.test:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image test mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        test_img(YOLO(**vars(FLAGS)))
    elif FLAGS.export:
        """
        Export model protobuffer
        """
        print("Export model mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        YOLO(**vars(FLAGS)).export_serving_model(FLAGS.export_path)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
