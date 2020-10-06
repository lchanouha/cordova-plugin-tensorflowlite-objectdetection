# cordova-plugin-tensorflowlite-objectdetection

Integrate the TensorFlow inference library into your PhoneGap/Cordova application!

This plugin is based on heigo's work (https://github.com/heigeo/cordova-plugin-tensorflow) on Android, adds on it iOS and custom quantized 300x300 model support. The code needs to be cleaned up and error handled in a better way but this plugin is actually working well.

## Usage
This plugins does not have an JS wrapper yet, feel free to add one (PRs are welcome). It handle only local files, so they must me downloaded first, with cordova-plugin-file-transfer for example:

### Loading model
```javascript
let id = 1;
let url = 'https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite';
$cordovafiletransfer.download(url, localPath, {}, true).then(function(){
	cordova.exec(function() {
		console.log('Classifier loaded')
	}, function(e){
		console.log("Could not load classifier', e);
	},"CDVTensorFlowLite", "loadModel", [ id, localPath, localPath]);
}, function(e){
   console.log("Could not load file", e);
});
```

### Inference

```javascript
//let imgData = new Image();
var  canvas = document.createElement('canvas'),
        ctx = canvas.getContext("2d");
canvas.width = imgData.width;
canvas.height = imgData.height;
ctx.drawImage(imgData, 0, 0);
let  uri = canvas.toDataURL('image/png');
toSendData = uri.replace(/^data:image.+;base64,/, '');

cordova.exec(function(results) {
	console.log(results)
}, function(e){
	console.log("Error loading inference from tensorflow", e);
}, "CDVTensorFlowLite", "classify", [ id, toSendData ]);
```
sample outputs:
```json
[ {title: 1, confidence: 0.857545, top: 102.415, left: 85.125, bottom: 140.45, right:120.15} ]
```

## Installation

### Cordova
TensorflowLite needs variable SWIFT_VERSION set. I did not find a way to set it cleanly. So the plugin needs to be added after the platform is added, variable set and platform built on time.
Add on config.xml, on section `<platform name="ios">`
```xml
<preference name="UseSwiftLanguageVersion" value="4.2" />
```
Then:
```bash
cordova plugin add cordova-custom-config -save
cordova build
cordova plugin add https://github.com/lchanouha/cordova-plugin-tensorflowlite-objectdetection -save
```

## Supported Platforms

* Android - working on 10, 11, should work on earlier versions
* iOS (use https://github.com/cordova-rtc/cordova-plugin-iosrtc for AR) - tested on iPhone 5S / iOS12

## API

### Check camera permission

```javascript
cordova.exec(function() {
	console.log("loaded model");
}, function(e){
	console.log("not granted permission", e);
}, "CDVTensorFlowLite", "checkPermission", [ ]);
```

### Loading model

```javascript
cordova.exec(function() {
	console.log('Classifier loaded')
}, function(e){
	console.log("Could not load classifier', e);
}, "CDVTensorFlowLite", "loadModel", [ clientNumber, internalURI ]);
```

clientNumber is a static client to call the appropriate model

internalUri is the model path (file://)

### Call inference

```javascript
cordova.exec(function(r) {
	console.log(output)
}, function(e){
	console.log("Error loading inference from tensorflow", e);
}, "CDVTensorFlowLite", "classify", [ clientNumber, base64Image ]);

clientNumber is the same value passed to loadModel
base64Image is an base64 encoded image (header "data:image.+;base64," must me stripped)
