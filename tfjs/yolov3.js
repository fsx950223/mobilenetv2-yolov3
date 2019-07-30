import './util.js'

/**
 * Get item's type exactly
 * @param item The item which need get type
 * @returns {string} item's type
 */
function getType(item){
    const prototypeString=Object.prototype.toString.call(item)
    return prototypeString.substring(prototypeString.indexOf(' ')+1,prototypeString.lastIndexOf(']'))
}

/**
 * Worker execute context
 */
function workerHandler(){
    if (typeof OffscreenCanvas !== 'undefined') {
        self.document = {
            createElement: () => {
                return new OffscreenCanvas(224, 224);
            }
        };
        self.window = self;
        self.screen = {
            width: 640,
            height: 480
        };
        self.HTMLVideoElement = function() {};
        self.HTMLImageElement = function() {};
        self.HTMLCanvasElement = OffscreenCanvas;
    }
    window = this
    importScripts(`${window.location.origin}/mobilenetv2-yolov3/tfjs/tf.min.js`,`${window.location.origin}/mobilenetv2-yolov3/tfjs/util.js`)
    class YoloV3{
        constructor(anchors,num_classes,input_shape){
            anchors=anchors.map(item=>Number(item))
            this.anchors=tf.reshape(anchors,[-1,2])
            this.num_classes=Number(num_classes)
            this.input_shape=input_shape.map(item=>Number(item))
            this.num_anchors=parseInt(this.anchors.shape[0]/3)
        }

        /**
         * Build model from factory
         * @param model_dir the model path where needs to be loaded
         * @param anchors anchors list
         * @param num_classes classes total number
         * @param input_shape input image shape
         * @returns {Promise<YoloV3>}
         */
        static async build(model_dir,anchors,num_classes,input_shape){
            const yolov3=new YoloV3(anchors,num_classes,input_shape)
            yolov3.model=await tf.loadLayersModel(model_dir)
            return yolov3
        }
        async predict(image,shape){
            const yolo_outputs=this.model.predict(tf.tensor(image,[1,...this.input_shape,3]))
            return await this.yolo_eval(yolo_outputs,shape)
        }
        yolo_correct_boxes(box_xy, box_wh, image_shape){
            const max_shape = tf.maximum(image_shape[0], image_shape[1])
            const ratio = tf.div(image_shape , max_shape)
            const boxed_shape = tf.mul(this.input_shape , ratio)
            const offset = tf.div(tf.sub(this.input_shape , boxed_shape) , 2)
            const scale = tf.div(image_shape , boxed_shape)
            box_xy = tf.mul(tf.sub(tf.mul(box_xy,this.input_shape) , offset) , scale)
            box_wh = tf.mul(box_wh,tf.mul(this.input_shape , scale))
            const box_mins = tf.sub(box_xy, tf.div(box_wh,2.))
            const box_maxes = tf.add(box_xy, tf.div(box_wh,2.))
            const boxes = tf.concat([box_mins,box_maxes],-1)
            return boxes
        }

        /**
         * Extract infos from feature
         * @param feats the feature need to be extracted
         * @param anchor the anchor of current branch
         * @returns {*[box position,box width and height,box confidence,box class]}
         */
        yolo_head(feats,anchor){
            const anchors_tensor=tf.reshape(anchor, [1, 1, 1, this.num_anchors, 2])
            const grid_shape = [feats.shape[1],feats.shape[2]]
            const grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]),[1, grid_shape[1], 1, 1])
            const grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]),[grid_shape[0], 1, 1, 1])
            let grid = tf.concat([grid_x, grid_y], -1)
            grid = tf.cast(grid, feats.dtype)
            const [feats_xy,feats_wh,feats_confidence,feats_class_probs]=tf.split(feats,[2,2,1,this.num_classes],-1)
            const box_xy = tf.div(tf.add(tf.sigmoid(feats_xy), grid), tf.cast(
                grid_shape, feats.dtype))
            const box_wh = tf.div(tf.mul(tf.exp(feats_wh) , tf.cast(
                anchors_tensor, feats.dtype)) , tf.cast(this.input_shape, feats.dtype))
            const box_confidence = tf.sigmoid(feats_confidence)
            const box_class_probs = tf.sigmoid(feats_class_probs)
            return [box_xy, box_wh, box_confidence, box_class_probs]
        }
        yolo_boxes_and_scores(feats,anchor,image_shape){
            const [box_xy, box_wh, box_confidence, box_class_probs] = this.yolo_head(feats, anchor)
            let boxes = this.yolo_correct_boxes(box_xy, box_wh, image_shape)
            boxes = tf.reshape(boxes, [-1, 4])
            let box_scores = tf.mul(box_confidence,box_class_probs)
            box_scores = tf.reshape(box_scores, [-1, this.num_classes])
            return [boxes, box_scores]
        }
        async yolo_eval(yolo_outputs,image_shape,max_boxes=20,score_threshold=0.2,iou_threshold=0.5){
            const num_layers=yolo_outputs.length
            const anchor_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            const input_shape=[yolo_outputs[0].shape[1]*32,yolo_outputs[0].shape[2]*32]
            let boxes=[]
            let box_scores=[]
            for(let l=0;l<num_layers;l++){
                yolo_outputs[l]=tf.reshape(yolo_outputs[l],[-1,yolo_outputs[l].shape[1],yolo_outputs[l].shape[2],3,this.num_classes + 5])
                const anchor=this.anchors.gather(anchor_mask[l])
                const [_boxes, _box_scores] =this.yolo_boxes_and_scores(yolo_outputs[l],anchor,image_shape)
                boxes.push(_boxes)
                box_scores.push(_box_scores)
            }
            boxes = tf.concat(boxes, 0)
            box_scores = tf.concat(box_scores, 0)
            const max_boxes_tensor = tf.scalar(max_boxes, 'int32')
            let boxes_ = []
            let scores_ = []
            let classes_ = []
            box_scores=tf.split(box_scores,new Array(this.num_classes).fill(1),-1)
            for(let c=0;c<this.num_classes;c++){
                const scores=box_scores[c].squeeze()
                const nms_index = await tf.image.nonMaxSuppressionAsync(boxes,
                    scores,
                    max_boxes_tensor,
                    iou_threshold,
                    score_threshold)
                const class_boxes = tf.gather(boxes, nms_index)
                const class_box_scores = tf.gather(scores, nms_index)
                const classes = tf.mul(tf.onesLike(class_box_scores, 'int32'),c)
                boxes_.push(class_boxes)
                scores_.push(class_box_scores)
                classes_.push(classes)
            }
            boxes_ = tf.concat(boxes_, 0)
            scores_ = tf.concat(scores_, 0)
            classes_ = tf.concat(classes_, 0)
            boxes_ = tf.cast(boxes_, 'int32')
            return [boxes_, scores_, classes_]
        }
    }
    let yolov3=undefined
    const service=new self.Service()
    service.register('init',async (data)=>{
        self.document.createElement = () => {
            return new OffscreenCanvas(...data.input_shape);
        };
        if(!yolov3)yolov3=await YoloV3.build(data.model_dir,data.anchors,data.num_classes,data.input_shape)
        return data
    })
    service.register('predict',async (data)=>{
        const result=await yolov3.predict(data.image,data.shape)
        return {boxes:result[0].dataSync(),scores:result[1].dataSync(),classes:result[2].dataSync()}
    })
}

/**
 * Client that communicates with worker
 */
export default class YoloV3Client{
    constructor(){
        const code=workerHandler.toString()
        const source=code.substring(code.indexOf('{')+1,code.lastIndexOf('}'))
        const blob=new Blob([source],{ type: 'application/javascript' })
        const objectURL=URL.createObjectURL(blob)
        this.worker=new Worker(objectURL);
        this.client=new Client(this.worker)
        URL.revokeObjectURL(objectURL)
    }

    async init(model_dir,anchors,num_classes,input_shape){
        return this.client.fetch('init',{model_dir:model_dir,anchors:anchors,num_classes:num_classes,input_shape:input_shape})
    }

    /**
     * Create a predict request to worker
     * @param input {}Float32Array|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|ImageData|String}
     * @returns {Promise<*|void>}
     */
    async predict(input,shape){
        const type=getType(input,shape)
        if(['Float32Array'].includes(type)) {
            return this.client.fetch('predict',{image:input,shape})
        }else if(['HTMLImageElement','HTMLCanvasElement','HTMLVideoElement','ImageData'].includes(type)){
            let data=tf.expandDims(tf.browser.fromPixels(input),0)
            data=tf.div(data,255)
            return this.client.fetch('predict',{image:data.dataSync(),shape})
        }else if(['String'].includes(type)){
            const picture=document.querySelector(input)
            let data=tf.expandDims(tf.browser.fromPixels(picture),0)
            data=tf.div(data,255)
            return this.client.fetch('predict',{image:data.dataSync(),shape})
        }else if(input.dataSync){
            return this.client.fetch('predict',{image:input.dataSync(),shape})
        }
    }
}

class YoloV3Components extends HTMLCanvasElement{
    constructor(model,anchors,classes){
        super();
        this.client=new YoloV3Client()
        this.ctx=this.getContext("2d");
    }
    async connectedCallback(){
        const [anchors,classes]=await Promise.all([fetch(this.attributes.anchors.value).then(res=>res.text()),fetch(this.attributes.classes.value).then(res=>res.text())])
        this.anchors=anchors.split(',')
        this.classes=classes.split('\n').filter(item=>item)
        const res=await this.client.init(this.attributes.model.value,this.anchors,this.classes.length,[this.attributes.width.value,this.attributes.height.value])
        this.draw()
    }
    disconnectedCallback(){}
    adoptedCallback(){}
    async draw(file){
        const imageBlob=file?file:await (await fetch(this.attributes.src.value)).blob()
        const bitmap=await createImageBitmap(imageBlob)
        this.ctx.drawImage(bitmap,0,0,this.attributes.width.value,this.attributes.height.value)
        const image=tf.browser.fromPixels(this)
        const resizedImage=tf.image.resizeBilinear(image,[Number(this.attributes.width.value),Number(this.attributes.height.value)])
        let data=tf.expandDims(resizedImage,0)
        data=tf.div(data,255)
        this.client.client.resolves={}
        const result=await this.client.predict(data.dataSync(),[image.shape[0],image.shape[1]])
        this.ctx.lineWidth = 3;
        this.ctx.fillStyle = "red";
        this.ctx.strokeStyle = "red";
        for(let i=0;i<result['classes'].length;i++){
            const [left,top,right,bottom]=result['boxes'].slice(i*4,i*4+4)
            this.ctx.fillText(this.classes[result['classes'][i]],left,top)
            this.ctx.strokeRect(left,top,right-left,bottom-top)
        }

    }
    static get observedAttributes() {return [...document.querySelector('canvas').attributes].map(item=>item.name) }
    async attributeChangedCallback(name,oldValue,newValue){
        if(name==='src'&&oldValue!=null){
            this.draw()
        }
    }
}
customElements.define('yolov3-detection',YoloV3Components,{extends:'canvas'})