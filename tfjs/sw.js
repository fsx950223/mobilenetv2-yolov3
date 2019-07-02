/**
 * Welcome to your Workbox-powered service worker!
 *
 * You'll need to register this file in your web app and you should
 * disable HTTP caching for this file too.
 * See https://goo.gl/nhQhGp
 *
 * The rest of the code is auto-generated. Please don't update this file
 * directly; instead, make changes to your Workbox build configuration
 * and re-run your build process.
 * See https://goo.gl/2aRDsh
 */

importScripts("workbox-v4.3.1/workbox-sw.js");
workbox.setConfig({modulePathPrefix: "workbox-v4.3.1"});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

/**
 * The workboxSW.precacheAndRoute() method efficiently caches and responds to
 * requests for URLs in the manifest.
 * See https://goo.gl/S9QRab
 */
self.__precacheManifest = [
  {
    "url": "cci_anchors.txt",
    "revision": "08f271a420b52977ea1fcc24e9f23958"
  },
  {
    "url": "cci_names.txt",
    "revision": "bbc9820dfde3fbac5676290ebc26b3de"
  },
  {
    "url": "image1.jpeg",
    "revision": "d0d3c444b923b5ca3403e20b19f4f051"
  },
  {
    "url": "image5.jpeg",
    "revision": "a70e498d0f9a7f549d4620d95c7250e9"
  },
  {
    "url": "index.html",
    "revision": "551a23ce7242b938c8a599a1605a7e79"
  },
  {
    "url": "keras/group1-shard1of9.bin",
    "revision": "95fbaa28d5bbaf511379abad5ae5eaec"
  },
  {
    "url": "keras/group1-shard2of9.bin",
    "revision": "1b407a3a212b216ff056da9bbe485fed"
  },
  {
    "url": "keras/group1-shard3of9.bin",
    "revision": "f088714d51e55a20c086b3077768c9be"
  },
  {
    "url": "keras/group1-shard4of9.bin",
    "revision": "ea76f7d956d6f56f3a717a4226331e91"
  },
  {
    "url": "keras/group1-shard5of9.bin",
    "revision": "6bdeb1ee6e92e91f1386a67045a1da3d"
  },
  {
    "url": "keras/group1-shard6of9.bin",
    "revision": "35452ed09ab1db9772222ee1d6e8a573"
  },
  {
    "url": "keras/group1-shard7of9.bin",
    "revision": "f245af90344962bf862ba6e1f9dc6203"
  },
  {
    "url": "keras/group1-shard8of9.bin",
    "revision": "d3c64da69d0cff43e0408b04376c48cd"
  },
  {
    "url": "keras/group1-shard9of9.bin",
    "revision": "34498dba999aecb71daf8ae3683d256b"
  },
  {
    "url": "keras/model.json",
    "revision": "72caea77abd0539011a3050a1d799c4f"
  },
  {
    "url": "sw-config.js",
    "revision": "328a66060e5569ae39443ae9a38d2ebc"
  },
  {
    "url": "test.js",
    "revision": "5e09bd04a105864f014793f3035a653e"
  },
  {
    "url": "tf.min.js",
    "revision": "462584fef5cef066a164d0fcc66357f1"
  },
  {
    "url": "util.js",
    "revision": "22755d471f429b8004ab6b3a760e86dd"
  },
  {
    "url": "yolov3.js",
    "revision": "6a576bc18da6b9557017a460970ad87c"
  }
].concat(self.__precacheManifest || []);
workbox.precaching.precacheAndRoute(self.__precacheManifest, {});
