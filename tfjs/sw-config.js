module.exports = {
  "importWorkboxFrom":"local",
  "globDirectory": "tfjs/",
  "globPatterns": [
    "**/*.{txt,jpeg,html,bin,json,js}"
  ],
  "maximumFileSizeToCacheInBytes":1024*1024*10,
  "swDest": "tfjs/sw.js"
};