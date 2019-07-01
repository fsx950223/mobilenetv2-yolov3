
async function serviceListen({data:request}){
    const response=new WorkerResponse(request.url)
    if(!(request.url in this.routes)){
        response.body="Url not found"
        response.statusCode=404
    }else{
        try{
            const body=await this.routes[request.url](request.body)
            response.body=body
        }catch(err){
            console.log(err)
            response.body=err.message
            response.statusCode=500
        }
    }
    self.postMessage(response)
}

class Service{
    constructor(){
        this.routes={}
        self.addEventListener('message',serviceListen.bind(this))
    }

    register(url,callback){
        this.routes[url]=callback
    }

    close(){
        self.removeEventListener('message',serviceListen)
    }
}

class WorkerResponse{
    constructor(url){
        this.statusCode=200
        this.body={}
        this.url=url
    }
}
function listen({data:response}){
    if(response.statusCode==200){
        this.resolves[response.url][0](response.body)
    }else{
        this.resolves[response.url][1](response.body)
    }
}
class Client{
    constructor(worker){
        worker.removeEventListener('message',listen.bind(this))
        worker.addEventListener('message',listen.bind(this))
        this.resolves={}
        this.worker=worker
    }
    async fetch(url,body){
        const request=new WorkerRequest(url,body)
        this.worker.postMessage(request)
        return new Promise((resolve,reject)=>this.resolves[url]=[resolve,reject])
    }
}

class WorkerRequest{
    constructor(url,body){
        this.url=url
        this.body=body
    }
}

self.Service=Service
self.Client=Client
self.WorkerResponse=WorkerResponse
self.WorkerRequest=WorkerRequest
