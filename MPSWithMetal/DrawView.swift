//
//  DrawView.swift
//  MPSWithMetal
//
//  Created by Taketomo Isazawa on 27/9/17.
//  Copyright © 2017 Taketomo Isazawa. All rights reserved.
//

// Code taken from https://github.com/r4ghu/iOS-CoreML-MNIST/blob/master/iOS-CoreML-MNIST/DrawView.swift
// Code taken with inspiration from Apple's Metal-2 sample MPSCNNHelloWorld
import UIKit

/**
 This class is used to handle the drawing in the DigitView so we can get user input digit,
 This class doesn't really have an MPS or Metal going in it, it is just used to get user input
 */
class DrawView: UIView {
    
    // some parameters of how thick a line to draw 15 seems to work
    // and we have white drawings on black background just like MNIST needs its input
    var linewidth = CGFloat(15) { didSet { setNeedsDisplay() } }
    var color = UIColor.white { didSet { setNeedsDisplay() } }
    
    // we will keep touches made by user in view in these as a record so we can draw them.
    var lines: [Line] = []
    var lastPoint: CGPoint!
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        lastPoint = touches.first!.location(in: self)
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        let newPoint = touches.first!.location(in: self)
        // keep all lines drawn by user as touch in record so we can draw them in view
        lines.append(Line(start: lastPoint, end: newPoint))
        lastPoint = newPoint
        // make a draw call
        setNeedsDisplay()
    }
    
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        
        let drawPath = UIBezierPath()
        drawPath.lineCapStyle = .round
        
        for line in lines{
            drawPath.move(to: line.start)
            drawPath.addLine(to: line.end)
        }
        
        drawPath.lineWidth = linewidth
        color.set()
        drawPath.stroke()
    }
    
    
    /**
     This function gets the pixel data of the view so we can put it in MTLTexture
     
     - Returns:
     Void
     */
    func getViewContext() -> CGContext? {
        // our network takes in only grayscale images as input
        let colorSpace:CGColorSpace = CGColorSpaceCreateDeviceGray()
        
        // we have 3 channels no alpha value put in the network
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        let imageSize = 100
        // this is where our view pixel data will go in once we make the render call
        let context = CGContext(data: nil, width: imageSize, height: imageSize, bitsPerComponent: 8, bytesPerRow: imageSize, space: colorSpace, bitmapInfo: bitmapInfo)
        
        // scale and translate so we have the full digit and in MNIST standard size 28x28
        context!.translateBy(x: 0 , y: CGFloat(imageSize))
        context!.scaleBy(x: CGFloat(imageSize)/self.frame.size.width, y: -CGFloat(imageSize)/self.frame.size.height)
        
        // put view pixel data in context
        self.layer.render(in: context!)
        
        return context
    }
    
    func getImageAndContextFromView() -> (image: UIImage, context: CGContext) {
        let context = self.getViewContext()
        let cgImage: CGImage! = context?.makeImage()
        return (UIImage(cgImage: cgImage), context!)
    }
}

/*
 Code adapted from https://gist.github.com/krooked/9c4c81557fc85bc61e51c0b4f3301e6e
 */

extension CGImage {
    func cropImageByBlack() -> CGImage {
        let cgImage = self
        var height = cgImage.height
        var width = cgImage.width
        
        var rect: CGRect = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))
        
        let data = pixelValues(fromCGImage: cgImage)
        
        var minX = width
        var minY = height
        var maxX: Int = 0
        var maxY: Int = 0
        
        //Filter through data and look for non-transparent pixels.
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (width * y + x)
                
                if data[Int(pixelIndex)] != 0 {
                    if (x < minX) {
                        minX = x
                    }
                    if (x > maxX) {
                        maxX = x
                    }
                    if (y < minY) {
                        minY = y
                    }
                    if (y > maxY) {
                        maxY = y
                    }
                }
            }
        }
        width = maxX-minX
        height = maxY-minY
        if width < height{
            width = height
        }
        else{
            height = width
        }
        print(width)
        print(height)
        rect = CGRect(x: CGFloat(minX), y: CGFloat(minY), width: CGFloat(width), height: CGFloat(height))
        let cgiImage = cgImage.cropping(to: rect)
        return cgiImage!
    }
}

/**
 2 points can give a line and this class is just for that purpose, it keeps a record of a line
 */
class Line{
    var start, end: CGPoint
    
    init(start: CGPoint, end: CGPoint) {
        self.start = start
        self.end   = end
    }
}
