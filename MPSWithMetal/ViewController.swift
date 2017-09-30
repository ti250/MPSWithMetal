//
//  ViewController.swift
//  MPSWithMetal
//
//  Created by Taketomo Isazawa on 21/9/17.
//  Copyright © 2017 Taketomo Isazawa. All rights reserved.
//

import UIKit
import Metal
import simd

func createInput(fromArray array: [UInt8])->[[Float32]]{
    var output = [[Float32]]()
    var i = 0
    let maxValue = Float32(array.max()!)
    while i < array.count{
        var value = Float32(array[i])
        if value >= maxValue/2{
            value = 1.0
        }
        else{
            value = 0.0
        }
        output.append([1.0, Float32(array[i])/maxValue])
        i += 1
    }
    i = 0
    print(output.count)
    return output
}

func getCoM(forArray array: [UInt8]) -> (Int, Int){
    // Expects input to be a square image layed out.
    let width = Int(sqrt(Double(array.count)))
    let height = width
    var i = 0
    var total: Int = 0
    var totalMoments: [Int] = [0, 0]
    for y in 0..<height {
        for x in 0..<width {
            let pixelIndex = (width * y + x)
            let value = Int(array[pixelIndex])
            total += value
            totalMoments[0] += x * value
            totalMoments[1] += y * value
        }
    }
    let xCoM = Int(Float(totalMoments[0]) / Float(total))
    let yCoM = Int(Float(totalMoments[1]) / Float(total))
    print("CoMs:", xCoM, yCoM)
    return (xCoM, yCoM)
}

func CoMPad(array: [UInt8], CoM: (x: Int, y: Int)) -> [UInt8]{
    var rowsBefore = 2 + (5 - CoM.y)
    var rowsAfter = 4 - rowsBefore
    if rowsBefore < 0{
        rowsBefore = 0
        rowsAfter = 4
    }
    else if rowsAfter < 0{
        rowsBefore = 4
        rowsAfter = 0
    }
    var columnsBefore = 2 + (5 - CoM.x)
    var columnsAfter = 4 - columnsBefore
    print(columnsAfter, columnsBefore)
    if columnsBefore < 0{
        columnsBefore = 0
        columnsAfter = 4
    }
    else if columnsAfter < 0{
        columnsBefore = 4
        columnsAfter = 0
    }
    var output = [UInt8]()
    let width = Int(sqrt(Double(array.count)))
    let height = width
    for _ in 0..<rowsBefore{
        for _ in 0..<(width+4){
            output.append(0)
        }
    }
    for y in 0..<height {
        for x in 0..<width {
            let pixelIndex = (width * y + x)
            let value = UInt8(array[pixelIndex])
            if x == 0{
                for _ in 0..<columnsBefore{
                    output.append(0)
                }
            }
            output.append(value)
            if x == width-1{
                for _ in 0..<columnsAfter{
                    output.append(0)
                }
            }
        }
    }
    for _ in 0..<rowsAfter{
        for _ in 0..<(width+4){
            output.append(0)
        }
    }
    return output
}

class ViewController: UIViewController {
    
    @IBOutlet weak var drawView: DrawView!
    @IBOutlet weak var predictLabel: UILabel!
    var device: MTLDevice! = nil
    var defaultLibrary: MTLLibrary! = nil
    var commandQueue: MTLCommandQueue! = nil
    var diagPartFunction: MTLFunction! = nil
    var textureDescriptor = MTLTextureDescriptor()
    var mps: MPS! = nil
    
    @IBAction func tappedClear(_ sender: Any) {
        drawView.lines = []
        drawView.setNeedsDisplay()
        predictLabel.isHidden = true
    }
    
    
    @IBAction func tappedDetect(_ sender: Any) {
        var context = drawView.getViewContext()
        var inputImage = context?.makeImage()?.cropImageByBlack()
        let colorSpace:CGColorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        let imageSize = 10
        context = CGContext(data: nil, width: imageSize, height: imageSize, bitsPerComponent: 8, bytesPerRow: imageSize, space: colorSpace, bitmapInfo: bitmapInfo)
        context?.draw(inputImage!, in: CGRect(x:0, y:0, width: imageSize, height: imageSize))
        inputImage = context?.makeImage()
        var inputArray = pixelValues(fromCGImage: inputImage)
        let CoM = getCoM(forArray: inputArray)
        print(CoM)
        print(inputArray)
        inputArray = CoMPad(array: inputArray, CoM: CoM)
        print(inputArray)
        let values = createInput(fromArray: inputArray)
        let mpsData = MPSData(withArray: values, batchsize: 1)
        let prediction = self.mps.predict(forData: mpsData)
        let maxElement = prediction.max()!
        print(prediction)
        let predictionText = String(describing: prediction.index(of: maxElement)!)
        self.predictLabel.isHidden = false
        self.predictLabel.text = predictionText
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        predictLabel.isHidden = true
        device = MTLCreateSystemDefaultDevice()
        defaultLibrary = device.makeDefaultLibrary()
        commandQueue = device.makeCommandQueue()
        textureDescriptor.textureType = .type1D
        textureDescriptor.pixelFormat = .r32Float
        textureDescriptor.usage = .shaderRead
        
        let path = Bundle.main.path(forResource: "MPSconfig", ofType: nil)
        let mpsBinaryData = NSData(contentsOfFile: path!)!
        self.mps = MPS(withData: mpsBinaryData)

//        path = Bundle.main.path(forResource: "test_data_bin", ofType: nil)
//        let binaryData = NSData(contentsOfFile: path!)!
//        let mpsData = MPSData(withData: binaryData)
//        print("Loaded")
//        let prediction = mps.predict(forData: mpsData)
//        print(prediction)
//        print("done")
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

func pixelValues(fromCGImage imageRef: CGImage?) -> [UInt8]
{
    var width = 0
    var height = 0
    var pixelValues: [UInt8]?
    if let imageRef = imageRef {
        width = imageRef.width
        height = imageRef.height
        let bitsPerComponent = imageRef.bitsPerComponent
        let bytesPerRow = imageRef.bytesPerRow
        let totalBytes = height * bytesPerRow
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var intensities = [UInt8](repeating: 0, count: totalBytes)
        
        let contextRef = CGContext(data: &intensities, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: 0)
        contextRef?.draw(imageRef, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))
        
        pixelValues = intensities
    }
    
    return pixelValues!
}

