from absl import app,flags
import tensorflow as tf
from yolo3.enum import BACKBONE,MODE,OPT
from train import train
from yolo import YOLO, detect_video, detect_img,export_tflite_model,export_serving_model,calculate_map
import os

FLAGS=flags.FLAGS

flags.DEFINE_enum_class('backbone',default=BACKBONE.EFFICIENTNET,enum_class=BACKBONE,help="Select network backbone, One of {'MOBILENETV2','DARKNET53','EFFICIENTNET'}")
flags.DEFINE_integer('batch_size',default=8,lower_bound=0,help="Train batch size")
flags.DEFINE_multi_integer('epochs',default=[10,10],lower_bound=0,help="Frozen train epochs and Full train epochs")
flags.DEFINE_string('export',default='export_model',help="Export path")
flags.DEFINE_string('input',default=None,help="Input data for various mode")
flags.DEFINE_multi_integer('input_size',default=(380,380),lower_bound=0,help="Input size")
flags.DEFINE_string('model',default=None,help="Model path")
flags.DEFINE_enum_class('mode',default=MODE.TRAIN,enum_class=MODE,help="Select exec mode, One of {'TRAIN','IMAGE','VIDEO','TFLITE','SERVING','MAP','PRUNE'}")
flags.DEFINE_string('gpus',default='0',help="Specific gpu indexes to run")
flags.DEFINE_string('train_dataset',default=None,help="Dataset glob for train")
flags.DEFINE_string('val_dataset',default=None,help="Dataset glob for validate")
flags.DEFINE_string('test_dataset',default=None,help="Dataset glob for test")
flags.DEFINE_enum_class('opt',default=None,enum_class=OPT,help="Select optimization, One of {'XLA','DEBUG','MKL'}")

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    if FLAGS.mode==MODE.TRAIN:
        tf.logging.info('Train mode')
        train(FLAGS)
    elif FLAGS.mode==MODE.IMAGE:
        tf.logging.info('Image detection mode')
        detect_img(YOLO(**vars(FLAGS)))
    elif FLAGS.mode==MODE.VIDEO:
        tf.logging.info('Video detection mode')
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    elif FLAGS.mode==MODE.MAP:
        tf.logging.info('Calculate test dataset map')
        calculate_map(YOLO(**vars(FLAGS)),FLAGS.test_dataset)
    elif FLAGS.mode==MODE.SERVING:
        tf.logging.info('Export hdf5 model to saved model')
        export_serving_model(YOLO(**vars(FLAGS)),FLAGS.export)
    elif FLAGS.mode==MODE.TFLITE:
        tf.logging.info('Export hdf5 model to tflite model')
        export_tflite_model(YOLO(**vars(FLAGS)),FLAGS.export)


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
