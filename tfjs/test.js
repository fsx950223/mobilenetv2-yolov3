import YoloV3Client from './yolov3.js'
if(navigator.onLine&&navigator.connection.effectiveType==='4g'){
    //Request server
}else {
    const main = async () => {
        const MODEL = `${window.location.origin}/keras/model.json`
        const client = new YoloV3Client()
        const res = await client.init(MODEL, [75, 60, 128, 97, 168, 153, 217, 265, 274, 172, 322, 428, 381, 266, 510, 396, 606, 612], 5, [224, 224])
        const picture = document.querySelector("#predict_image")
        const image = tf.browser.fromPixels(picture)
        const resizedImage = tf.image.resizeBilinear(image, [224, 224])
        const data = tf.div(tf.expandDims(resizedImage, 0), 255)
        for (let i = 0; i < 10; i++) {
            console.time('post')
            const result = await client.predict(data.dataSync(), [image.shape[0], image.shape[1]])
            console.timeEnd('post')
            console.log(result)
        }
    }
    main()
}
