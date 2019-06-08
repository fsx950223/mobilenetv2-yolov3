from PIL import Image, ImageFont, ImageDraw
import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2,prediction_service_pb2_grpc
import os
from grpc._cython.cygrpc import CompressionAlgorithm,CompressionLevel
import colorsys

def Client(object):
    def __init__(self,address,class_names,size=416,options=[
            ('grpc.max_receive_message_length', -1),
            ('grpc.default_compression_algorithm', CompressionAlgorithm.gzip),
            ('grpc.default_compression_level', CompressionLevel.high)
        ]):
        channel = grpc.insecure_channel(address, options)
        self.stub=prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.size=size
        self.class_names=class_names
    
    def predict(self,image):
        scale = self.size / max(image.size)
        if scale < 1:
            image = image.resize((int(line * scale) for line in image.size),
                                 Image.BILINEAR)
        image_data = np.array(image, dtype='uint8')
        image_data = np.expand_dims(image_data, 0)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'detection'
        request.model_spec.signature_name = 'serving_default'
        request.inputs['predict_image:0'].CopyFrom(
            tf.make_tensor_proto(image_data))
        result = self.stub.Predict(request)
        out_classes = np.array(result.outputs['yolo/classes:0'].int_val)
        out_classes = [self.class_names[c] for c in out_classes]
        out_scores = np.array(result.outputs['yolo/scores:0'].float_val)
        out_boxes = np.array(result.outputs['yolo/boxes:0'].int_val).reshape(-1,4)
        return out_classes,out_scores,out_boxes
    


