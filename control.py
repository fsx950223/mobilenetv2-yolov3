from tensorflow_serving.apis import get_model_status_pb2, model_service_pb2_grpc, model_management_pb2
import grpc
from absl import app,flags
from enum import unique,Enum
from kazoo.client import KazooClient
from yolo3.utils import get_classes

@unique
class MODE(Enum):
    STATUS=1
    CONFIG=2
    ZOOKEEPER=3

FLAGS=flags.FLAGS
flags.DEFINE_enum_class("mode",default=MODE.ZOOKEEPER,enum_class=MODE,help='exec mode')
flags.DEFINE_multi_string("addresses",default=["10.12.102.32:8500","10.12.102.33:8500","10.12.102.52:8500","10.12.102.53:8500"],help='grpc servers address')

def get_config(*args):
    return bytes('#'.join(str(arg) for arg in args),encoding="utf8")

def main(_):
    if MODE.STATUS == FLAGS.mode:
        request = get_model_status_pb2.GetModelStatusRequest()
        request.model_spec.name = 'detection'
        request.model_spec.signature_name = 'serving_default'
    elif MODE.CONFIG == FLAGS.mode:
        request = model_management_pb2.ReloadConfigRequest()
        config = request.config.model_config_list.config.add()
        config.name = 'detection'
        config.base_path = '/models/detection/detection'
        config.model_platform = 'tensorflow'
        config.model_version_policy.specific.versions.append(5)
        config.model_version_policy.specific.versions.append(7)
        config2 = request.config.model_config_list.config.add()
        config2.name = 'pascal'
        config2.base_path = '/models/detection/pascal'
        config2.model_platform = 'tensorflow'
    elif MODE.ZOOKEEPER==FLAGS.mode:
        zk = KazooClient(hosts="10.10.67.225:2181")
        zk.start()
        zk.ensure_path('/serving/cunan')
        zk.set('/serving/cunan',get_config('detection',5,224,'serving_default',','.join(get_classes('model_data/cci.names')),"10.12.102.32:8000"))
        return
    for address in FLAGS.addresses:
        channel = grpc.insecure_channel(address)
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        if MODE.STATUS==FLAGS.mode:
            result = stub.GetModelStatus(request)
        elif MODE.CONFIG==FLAGS.mode:
            result = stub.HandleReloadConfigRequest(request)
        print(result)


if __name__ == '__main__':
    app.run(main)
