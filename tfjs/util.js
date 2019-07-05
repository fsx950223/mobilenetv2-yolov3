class Service{
    listen=async ({data:request})=>{
        const response=new WorkerResponse(request.id,request.url)
        if(!(request.url in this.routes)){
            response.body="Url not found"
            response.statusCode=404
        }else{
            try{
                const body=await this.routes[request.url](request.body)
                response.body=body
            }catch(err){
                response.body=err.message
                response.statusCode=500
            }
        }
        self.postMessage(response)
    }

    constructor(){
        this.routes={}
        self.addEventListener('message',this.listen)
    }

    /**
     * Register a service on url
     * @param url Client request url
     * @param callback Function to solve url request
     */
    register(url,callback){
        this.routes[url]=callback
    }

    close(){
        self.removeEventListener('message',this.listen)
    }
}

class WorkerResponse{
    constructor(id,url){
        this.statusCode=200
        this.body={}
        this.url=url
        this.id=id
    }
}

class Client{
    listen=({data:response})=>{
        const resolve=this.resolves[response.id]
        if(resolve) {
            if (response.statusCode == 200) {
                resolve[0](response.body)
            } else {
                resolve[1](response.body)
            }
        }
    }
    constructor(worker){
        this.resolves={}
        this.worker=worker
        this.worker.addEventListener('message',this.listen)
    }
    async fetch(url,body){
        const request=new WorkerRequest(url,body)
        this.worker.postMessage(request)
        return new Promise((resolve,reject)=>this.resolves[request.id]=[resolve,reject])
    }
    clear(){
        this.resolves={}
    }
    close(){
        this.worker.removeEventListener('message',this.listen)
    }
}

class WorkerRequest{
    uuid() {
      const s = [];
      const hexDigits = "0123456789abcdef";
      for (let i = 0; i < 36; i++) {
        s[i] = hexDigits.substr(Math.floor(Math.random() * 0x10), 1);
      }
      s[14] = "4"; // bits 12-15 of the time_hi_and_version field to 0010
      s[19] = hexDigits.substr((s[19] & 0x3) | 0x8, 1); // bits 6-7 of the clock_seq_hi_and_reserved to 01
      s[8] = s[13] = s[18] = s[23] = "-";

      return s.join("");
    }
    constructor(url,body){
        this.id=this.uuid()
        this.url=url
        this.body=body
    }
}

self.Service=Service
self.Client=Client
self.WorkerResponse=WorkerResponse
self.WorkerRequest=WorkerRequest
