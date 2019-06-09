from PIL import Image, ImageFont, ImageDraw
import threading
import os
import time
import psutil

SIZE=416
FOLDER='/media/fangsixie/data/keras-yolo3/logs/VOCdevkit/Illegal_parking/'
CLASSES_PATH = os.path.expanduser('model_data/cci.names')
SAVE_PATH='chengyun'
THREADS=10
EPOCHS=10

results=[]
mutex=threading.Lock()
index=-1
def predict_wrapper(client,image_names,save=False):
    mutex.acquire()
    index+=1
    mutex.release()
    image_path=os.path.join(folder,image_names[index])
    image = Image.open(image_path)
    
    for epoch in range(EPOCHS):
        start=int(round(time.time() * 1000))
        out_classes,out_scores,out_boxes=client.predict(image)
        end=int(round(time.time() * 1000))
        results.append(end-start)
    if save:
        save_image(image,out_classes,out_scores,out_boxe)
    else:
        for i, predicted_class in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            print(label, (left, top), (right, bottom))
    
def save_image(image,out_classes,out_scores,out_boxes):
        hsv_tuples = [
            (x / len(class_names), 1., 1.) for x in range(len(class_names))
        ]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[1] + image.size[0]) // 300
        draw = ImageDraw.Draw(image)
        for i, predicted_class in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            print(label, (left, top), (right, bottom))
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
        image.save(os.path.join(SAVE_PATH,image_name))
def print_system_info():
    pid=os.getpid()
    while True:
        process=psutil.Process(pid)
        print("memory: ",process.memory_percent())
        print("cpu: ",process.cpu_percent(None))

def main(_):
    os.mkdir(SAVE_PATH)
    names = os.listdir(FOLDER)[:10]
    with open(CLASSES_PATH) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    client=Client('10.12.102.32:8000',class_names,SIZE)
    threads = []
    info_thread=threading.Thread(target=print_system_info)
    info_thread.start()
    for i in range(THREADS):
        thread = threading.Thread(target=predict_wrapper, args=[image])
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    total_time=0
    results=results.sort()
    for result in results:
        total_time+=result
    average_time=total_time/len(results)
    print("average time: ",average_time)
    print("tps: ",1/average_time)
    print("min time: ",results[0])
    print("max time: ",results[-1])

if __name__ == '__main__':
    main(None)

