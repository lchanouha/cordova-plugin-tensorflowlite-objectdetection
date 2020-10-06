// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

/// A result from invoking the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let inferences: [[AnyHashable : Any]]
}

/// An inference from invoking the `Interpreter`.
struct Inference {
  let confidence: Float
  let label: Int
  let boxes: [Float]
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet model.
enum MobileNet {
  static let modelInfo: FileInfo = (name: "mobilenet_quant_v1_224", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labels", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler {

  // MARK: - Internal Properties

  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int

  let resultCount = 3
  let threadCountLimit = 10

  // MARK: - Model Parameters

  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 300
  let inputHeight = 300

  // MARK: - Private Properties

  /// List of labels from the given labels file.
  private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  /// Information about the alpha component in RGBA data.
  private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelPath: String, threadCount: Int = 1) {

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    var options = InterpreterOptions()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }
  }

  // MARK: - Internal Methods

  /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
    
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)


    let imageChannels = 4
    assert(imageChannels >= inputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize) else {
      return nil
    }

    let interval: TimeInterval
    let outputTensorBoxes: Tensor
    let outputTensorClasses: Tensor
    let outputTensorScores: Tensor
    let outputTensorNumD: Tensor
    
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        thumbnailPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }

      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      //print("output \(interpreter.outputTensorCount)");
      interval = Date().timeIntervalSince(startDate) * 1000

      // Get the output `Tensor` to process the inference results.
      outputTensorBoxes   = try interpreter.output(at: 0)
      outputTensorClasses = try interpreter.output(at: 1)
      outputTensorScores  = try interpreter.output(at: 2)
      outputTensorNumD    = try interpreter.output(at: 3)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }



    let quantizedResultsBoxes   = [Float32](unsafeData: outputTensorBoxes.data)
    let quantizedResultsClasses = [Float32](unsafeData: outputTensorClasses.data)
    let quantizedResultsScores  = [Float32](unsafeData: outputTensorScores.data)
    let quantizedResultsNumD    = [Float32](unsafeData: outputTensorNumD.data)
      
    //dump(quantizedResultsBoxes)
    //dump(quantizedResultsClasses)
    //dump(quantizedResultsScores)
    //dump(quantizedResultsNumD)
    
    var results: [[AnyHashable : Any]] = []
    let r = Int(quantizedResultsNumD![0])

    for i in 0...(r-1) {
        results.append([
            "confidence": quantizedResultsScores![i],
            "title"     : 1 + Int (quantizedResultsClasses![i]),
            "boxes"     : [quantizedResultsBoxes![i] * 300, quantizedResultsBoxes![i+1] * 300, quantizedResultsBoxes![i+2] * 300, quantizedResultsBoxes![i+3] * 300]
        ])
        
    }
    //dump(results);

    // Return the inference time and inference results.
    return Result(inferenceTime: interval, inferences: results)
  }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width
    
    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      print("Error: out of memory")
      return nil
    }
    
    defer {
        free(destinationData)
    }

    var destinationBuffer = vImage_Buffer(data: destinationData,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: destinationBytesPerRow)

    let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)

    switch (pixelBufferFormat) {
    case kCVPixelFormatType_32BGRA:
        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32ARGB:
        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32RGBA:
        vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    default:
        // Unknown pixel format.
        return nil
    }

    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
    if isModelQuantized {
        return byteData
    }

    // Not quantized, convert to floats
    let bytes = Array<UInt8>(unsafeData: byteData)!
    var floats = [Float]()
    for i in 0..<bytes.count {
        floats.append(Float(bytes[i]) / 255.0)
    }
    return Data(copyingBufferOf: floats)
  }
}

// MARK: - Extensions

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}
