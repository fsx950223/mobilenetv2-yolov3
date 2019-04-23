# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

#from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# The image URL is the location of the image we should send to the server

def main(_):

    image = Image.open('/media/fangsixie/data/keras-yolo3/logs/VOCdevkit/Illegal_parking/标注前/image396.jpeg')
    image_data=np.array(image,dtype='float32')
    image_data/=255.
    image_data=np.expand_dims(image_data,0)
    channel = grpc.insecure_channel('localhost:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'test'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['predict_image:0'].CopyFrom(
        tf.make_tensor_proto(image_data))

    result = stub.Predict(request, 10.0)  # 10 secs timeout
    print(result)

if __name__ == '__main__':
    tf.app.run()
