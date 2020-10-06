import TensorFlowLite

@objc(CDVTensorFlowLite) class CDVTensorFlowLite : CDVPlugin {
  private var currModel: ModelDataHandler?
    
    @objc(checkPermission:)
    func checkPermission(command: CDVInvokedUrlCommand){
        switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: // The user has previously granted access to the camera.
                let pluginResult = CDVPluginResult(status: CDVCommandStatus_OK, messageAs: "The plugin succeeded (camera already autorizhed)");
                self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
                return ;
            
            case .notDetermined: // The user has not yet been asked for camera access.
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    if granted {
                        let pluginResult = CDVPluginResult(status: CDVCommandStatus_OK, messageAs: "now granded");
                        self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
                    } else {
                        let pluginResult = CDVPluginResult (status: CDVCommandStatus_ERROR, messageAs: "Not autorizhed");
                        self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
                    }
                }
                return ;
            
            case .denied: // The user has previously denied access.
                let pluginResult = CDVPluginResult (status: CDVCommandStatus_ERROR, messageAs: "Not autorizhed");
                self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
                return

            case .restricted: // The user can't grant access due to restrictions.
                let pluginResult = CDVPluginResult (status: CDVCommandStatus_ERROR, messageAs: "Not autorizhed");
                self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
                return
        }

    }
  @objc(loadModel:) // Declare your function name.
  func loadModel(command: CDVInvokedUrlCommand) { // write the function code.
    /* 
     * Always assume that the plugin will fail.
     * Even if in this example, it can't.
     */
    
    var path = command.arguments[1] as? String ?? ""
    path = path.replacingOccurrences(of: "file://", with: "")
    currModel = ModelDataHandler(modelPath: path)
    if currModel != nil {
        let pluginResult = CDVPluginResult(status: CDVCommandStatus_OK, messageAs: "The plugin succeeded");
        self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
    } else {
        let pluginResult = CDVPluginResult (status: CDVCommandStatus_ERROR, messageAs: "The Plugin Failed");
        self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
    }
    // Send the function result back to Cordova.
  }
    
    func fromBase64(d: String) -> Data? {
            guard let data = Data(base64Encoded: d) else {
                    return nil
            }
            return data
    }
    
    @objc (classify:)
    func classify(command: CDVInvokedUrlCommand) {
        let base64 = command.arguments[1] as? String ?? ""
        guard let decodedData = fromBase64(d: base64) else {
            let pluginResult = CDVPluginResult (status: CDVCommandStatus_ERROR, messageAs: "base64 invalide");
            self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
            return ;
        }
        guard let img = UIImage(data: decodedData) else {
            let pluginResult = CDVPluginResult (status: CDVCommandStatus_ERROR, messageAs: "Image invalide");
            self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
            return ;
        }

        guard let f = buffer(from: img) else {
            let pluginResult = CDVPluginResult (status: CDVCommandStatus_ERROR, messageAs: "Impossible de convertir en pixelbuffer");
            self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
            return ;
        }
        let result = currModel?.runModel(onFrame: f)
        //print ("result is \(result)")
        let pluginResult = CDVPluginResult(status: CDVCommandStatus_OK, messageAs: result?.inferences);
        self.commandDelegate!.send(pluginResult, callbackId: command.callbackId);
    }
    
    func buffer(from image: UIImage) -> CVPixelBuffer? {
      let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
      var pixelBuffer : CVPixelBuffer?
      let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
      guard (status == kCVReturnSuccess) else {
        return nil
      }

      CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
      let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

      let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
      let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

      context?.translateBy(x: 0, y: image.size.height)
      context?.scaleBy(x: 1.0, y: -1.0)

      UIGraphicsPushContext(context!)
      image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
      UIGraphicsPopContext()
      CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

      return pixelBuffer
    } 
}
