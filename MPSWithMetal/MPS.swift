//
//  MPS.swift
//  MPSWithMetal
//
//  Created by Taketomo Isazawa on 23/9/17.
//  Copyright © 2017 Taketomo Isazawa. All rights reserved.
//

import Foundation
import Metal
import simd

public struct tensorOpsConstants{
    // Constants for this app.
    public static let maxTensorRank = 10
    public static let sizeOfTensorDescriptor = 88
}

struct tensorDescriptor{
    var rank: Int = 0
    var multipliers:(uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    var dimensions:(uint, uint, uint, uint, uint, uint, uint, uint, uint, uint) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    init(withDimensions dimensions: [uint]){
        self.rank = dimensions.count
        let _dimensions_lookup = withUnsafeMutablePointer(to: &self.dimensions) { $0.withMemoryRebound(to: uint.self, capacity: tensorOpsConstants.maxTensorRank) { $0 } }
        let _multipliers_lookup = withUnsafeMutablePointer(to: &self.multipliers) { $0.withMemoryRebound(to: uint.self, capacity: tensorOpsConstants.maxTensorRank+1) { $0 } }
        var i = 0
        var multiplier: uint = 1
        while i < self.rank{
            _dimensions_lookup[i] = dimensions[i]
            i += 1
        }
        i = self.rank
        while i > 0{
            _multipliers_lookup[i] = multiplier
            multiplier *= dimensions[i-1]
            i -= 1
        }
        i = 0
        //        while i < tensorOpsConstants.maxTensorRank{
        //            self.multipliers[i] = uint(Float(self.multipliers[i]))
        //            self.dimensions[i] = uint(Float(self.dimensions[i]))
        //            i += 1
        //        }
        //        self.multipliers[i] = self.multipliers[i]
        _multipliers_lookup[0] = multiplier
    }
//    init(withDimensions dimensions: [Int]) {
//        self.init(withDimensions: [uint](dimensions))
//    }
}

struct Tensor{
    var value: [Float32]
    var descriptor: tensorDescriptor
    
    
    func makeBuffer(onDevice device: MTLDevice) -> MTLBuffer{
        var value = self.value
        let length = MemoryLayout<Float32>.stride * value.count
        return device.makeBuffer(bytes: &value, length: length)!
    }
}

class MPSData{
    var labelTensors: [Tensor]! = nil
    var dataTensors: [Tensor]! = nil
    
    init(withArray array: [[Float32]], batchsize: uint){
        let descriptor = tensorDescriptor(withDimensions: [batchsize, uint(array[0].count)])
        print(descriptor)
        dataTensors = [Tensor]()
        for element in array{
            let tensor = Tensor(value: element, descriptor: descriptor)
            dataTensors.append(tensor)
        }
    }
    
    init(withData data: NSData){
        var version:Int = 0
        data.getBytes(&version, range: NSMakeRange(0, 2))
        print("MPSData version:", version)
        
        var inputSize:Int = 0
        data.getBytes(&inputSize, range: NSMakeRange(2, 2))
        
        var localDimension:Int = 0
        data.getBytes(&localDimension, range: NSMakeRange(4, 2))
        
        var numClasses:Int = 0
        data.getBytes(&numClasses, range: NSMakeRange(6, 2))
        
        var batchSize:Int = 0
        data.getBytes(&batchSize, range: NSMakeRange(8, 2))
        batchSize = 10000
        // Fixed this way for testing.
        // TODO: Add a MNISTData file that doesn't have the wrong batchSize so this isn't neccesary
        
        var batchSizeuint = uint(batchSize)
        let localDimensionuint = uint(localDimension)
        
        let actualBatchSize = 10
        // The actualBatchSize variable is used so that less data is loaded in than is actually saved in the file, as large batch sizes can crash older devices
        var i = 0
        var startIndex = 10
        var length = 0
        let dataDescriptor = tensorDescriptor(withDimensions: [uint(actualBatchSize), localDimensionuint])
        print(dataDescriptor.dimensions)
        dataTensors = [Tensor]()
        print(inputSize)
        
        // Make tensors for the data
        while i < inputSize{
            length = localDimension * batchSize
            var value = [Float32](repeatElement(0.0, count: actualBatchSize*localDimension))
            var j = 0
            while j < length{
                if j < actualBatchSize*localDimension{
                    data.getBytes(&value[j], range: NSMakeRange(startIndex, 4))
                }
                startIndex += 4
                j += 1
            }
            dataTensors.append(Tensor(value: value, descriptor: dataDescriptor))
            i += 1
        }
        print("Batch size:", batchSize)
        i = 0
        // TODO: import the classifications.
        while i < inputSize{
            i += 1
        }
    }
}

class MPS{
    var tensors: [Tensor]
    var startTensor: Tensor
    var endTensor: Tensor
    var specialNodeLocation:Int
    
    var device: MTLDevice! = nil
    var defaultLibrary: MTLLibrary! = nil
    var commandQueue: MTLCommandQueue! = nil
    var tensorDotFunction: MTLFunction! = nil
    var tensorDotDiagFunction: MTLFunction! = nil
    var tensorDotPipelineState: MTLComputePipelineState
    var tensorDotDiagPipelineState: MTLComputePipelineState
    var textureDescriptor = MTLTextureDescriptor()
    
    func predict(forData mpsData: MPSData) -> [Float]{
        let start = Date()
        var commandBuffer:MTLCommandBuffer! = nil
        var commandEncoder:MTLComputeCommandEncoder! = nil
        var contractedNodes = calculateContractedNodes(forData: mpsData, commandBuffer: &commandBuffer, commandEncoder: &commandEncoder)
        print("contracted")
//         Calculate the C1 and C2 matrices
        let C1 = calculateC1(withContractedNodes: contractedNodes, commandBuffer: &commandBuffer, commandEncoder: &commandEncoder)
        print("C1 found")
        let C2 = calculateC2(withContractedNodes: contractedNodes, commandBuffer: &commandBuffer, commandEncoder: &commandEncoder)
        commandBuffer.waitUntilCompleted()
//        print("C2 found")
        let results = calculatePrediction(fromC1: C1, C2: C2, contractedNode: (contractedNodes.0[specialNodeLocation], contractedNodes.1[specialNodeLocation]))
//        print(contractedNodes.0.count)
//        var results = (contractedNodes.0[195], contractedNodes.1[195])
//        print(mpsData.dataTensors[0].value.count)
//        print(self.tensors[0].descriptor.dimensions)
//        print(self.tensors[0].value.count)
//        print(results.1.dimensions)
        
        let float32Pointer = results.0.contents().bindMemory(to: Float.self, capacity: Int(results.1.multipliers.0))
        let float32Buffer = UnsafeBufferPointer(start:float32Pointer, count: Int(results.1.multipliers.0))
        let outputBytes = Array(float32Buffer)
        let end = Date()
        print("Time taken: ", end.timeIntervalSince(start))
        return outputBytes
//        print(outputBytes)
        
    }
    
    func calculatePrediction(fromC1 C1: (MTLBuffer, tensorDescriptor), C2: (MTLBuffer, tensorDescriptor), contractedNode: (MTLBuffer, tensorDescriptor)) -> (MTLBuffer, tensorDescriptor){
        let w = tensorDotDiagPipelineState.threadExecutionWidth
        var axes = int2(1, 2)
        var axesBuffer: MTLBuffer = device.makeBuffer(bytes: &axes, length: MemoryLayout<int2>.stride)!
        var collectedAxes = int2(0, 0)
        let collectedAxesBuffer = device.makeBuffer(bytes: &collectedAxes, length: MemoryLayout<int2>.stride)!
        var commandBuffer = commandQueue.makeCommandBuffer()!
        var commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        commandEncoder.setComputePipelineState(self.tensorDotDiagPipelineState)
        //tensordot C1 with the collected special node
        var C1WithSpecialNodeDescriptor = tensorDescriptor(withDimensions: [C1.1.dimensions.0, contractedNode.1.dimensions.1, contractedNode.1.dimensions.3])
        var C1Descriptor = C1.1
        let C1DescriptorBuffer = device.makeBuffer(bytes: &C1Descriptor, length: MemoryLayout<tensorDescriptor>.stride)!
        var contractedNodeDescriptor = contractedNode.1
        let contractedNodeDescriptorBuffer = device.makeBuffer(bytes: &contractedNodeDescriptor, length: MemoryLayout<tensorDescriptor>.stride)!
        let C1WithSpecialNodeDescriptorBuffer = device.makeBuffer(bytes: &C1WithSpecialNodeDescriptor, length: MemoryLayout<tensorDescriptor>.stride)!
        var length = Int(C1WithSpecialNodeDescriptor.multipliers.0) * MemoryLayout<Float32>.stride
        let C1WithSpecialNode = device.makeBuffer(length: length)
        
        commandEncoder.setBuffer(C1DescriptorBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(contractedNodeDescriptorBuffer, offset: 0, index: 1)
        commandEncoder.setBuffer(C1WithSpecialNodeDescriptorBuffer, offset: 0, index: 2)
        commandEncoder.setBuffer(axesBuffer, offset: 0, index: 3)
        commandEncoder.setBuffer(collectedAxesBuffer, offset: 0, index: 4)
        commandEncoder.setBuffer(C1.0, offset: 0, index: 5)
        commandEncoder.setBuffer(contractedNode.0, offset:0, index: 6)
        commandEncoder.setBuffer(C1WithSpecialNode, offset:0, index:7)
        
        var totalThreads = Int(C1WithSpecialNodeDescriptor.multipliers.0)
        var threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
        var threadgroupsPerGrid = MTLSizeMake((totalThreads + w - 1)/w, 1, 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        commandEncoder.endEncoding()
        commandBuffer.commit()
        
        //tensordot C2 with the C1 dotted with collected special Node
        
        var C2Descriptor = C2.1
        let C2DescriptorBuffer = device.makeBuffer(bytes: &C2Descriptor, length: MemoryLayout<tensorDescriptor>.stride)!
        axes = int2(2, 1)
        axesBuffer = device.makeBuffer(bytes: &axes, length: MemoryLayout<int2>.stride)!
        var outputDescriptor = tensorDescriptor(withDimensions: [C1Descriptor.dimensions.0, contractedNodeDescriptor.dimensions.1])
        let outputDescriptorBuffer = device.makeBuffer(bytes: &outputDescriptor, length: MemoryLayout<tensorDescriptor>.stride)!
        length = Int(outputDescriptor.multipliers.0) * MemoryLayout<Float32>.stride
        let outputBuffer = device.makeBuffer(length: length)
        
        commandBuffer.waitUntilCompleted()
        commandBuffer = commandQueue.makeCommandBuffer()!
        commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        commandEncoder.setComputePipelineState(self.tensorDotDiagPipelineState)
        commandEncoder.setBuffer(C1WithSpecialNodeDescriptorBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(C2DescriptorBuffer, offset: 0, index: 1)
        commandEncoder.setBuffer(outputDescriptorBuffer, offset: 0, index: 2)
        commandEncoder.setBuffer(axesBuffer, offset: 0, index: 3)
        commandEncoder.setBuffer(collectedAxesBuffer, offset: 0, index: 4)
        commandEncoder.setBuffer(C1WithSpecialNode, offset: 0, index: 5)
        commandEncoder.setBuffer(C2.0, offset:0, index: 6)
        commandEncoder.setBuffer(outputBuffer, offset:0, index:7)
        totalThreads = Int(outputDescriptor.multipliers.0)
        threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
        threadgroupsPerGrid = MTLSizeMake((totalThreads + w - 1)/w, 1, 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return (outputBuffer!, outputDescriptor)
    }
    
    func calculateC1(withContractedNodes contractedNodes: ([MTLBuffer], [tensorDescriptor]), commandBuffer: inout MTLCommandBuffer!, commandEncoder: inout MTLComputeCommandEncoder!) -> (MTLBuffer, tensorDescriptor){
        var C1 = self.startTensor.makeBuffer(onDevice: self.device)
        var C1Descriptor: tensorDescriptor! = self.startTensor.descriptor
        var C1DescriptorBuffer: MTLBuffer = device.makeBuffer(bytes: &C1Descriptor, length: MemoryLayout<tensorDescriptor>.stride)!
        var i = 0
        var w = self.tensorDotPipelineState.threadExecutionWidth
        var axes = int2(0, 1)
        var axesBuffer: MTLBuffer = device.makeBuffer(bytes: &axes, length: MemoryLayout<int2>.stride)!
        var collectedAxes = int2(0, 0)
        let collectedAxesBuffer = device.makeBuffer(bytes: &collectedAxes, length: MemoryLayout<int2>.stride)!
        while i < self.specialNodeLocation{
            let contractedNode = contractedNodes.0[i]
            var contractedNodeDescriptor = contractedNodes.1[i]
            var outputDescriptor = tensorDescriptor(withDimensions: [contractedNodeDescriptor.dimensions.0, contractedNodeDescriptor.dimensions.2])
            let length = Int(outputDescriptor.multipliers.0) * MemoryLayout<Float32>.stride
            let outputBuffer = device.makeBuffer(length: length)
            
            let outputDescriptorBuffer = device.makeBuffer(bytes: &outputDescriptor, length: MemoryLayout<tensorDescriptor>.stride)!
            let contractedNodeDescriptorBuffer = device.makeBuffer(bytes: &contractedNodeDescriptor, length: MemoryLayout<tensorDescriptor>.stride)!
            commandBuffer.waitUntilCompleted()
            commandBuffer = commandQueue.makeCommandBuffer()!
            commandEncoder = commandBuffer.makeComputeCommandEncoder()!
            if i == 0{
                commandEncoder.setComputePipelineState(self.tensorDotPipelineState)
            }
            else{
                commandEncoder.setComputePipelineState(self.tensorDotDiagPipelineState)
            }
            
            commandEncoder.setBuffer(C1DescriptorBuffer, offset: 0, index: 0)
            commandEncoder.setBuffer(contractedNodeDescriptorBuffer, offset: 0, index: 1)
            commandEncoder.setBuffer(outputDescriptorBuffer, offset: 0, index: 2)
            commandEncoder.setBuffer(axesBuffer, offset: 0, index: 3)
            if i != 0{
                commandEncoder.setBuffer(collectedAxesBuffer, offset: 0, index: 4)
                commandEncoder.setBuffer(C1, offset: 0, index: 5)
                commandEncoder.setBuffer(contractedNode, offset:0, index: 6)
                commandEncoder.setBuffer(outputBuffer, offset:0, index:7)
            }
            else {
                commandEncoder.setBuffer(C1, offset: 0, index: 4)
                commandEncoder.setBuffer(contractedNode, offset:0, index: 5)
                commandEncoder.setBuffer(outputBuffer, offset:0, index:6)
            }
            
            let totalThreads = Int(outputDescriptor.multipliers.0)
            let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
            let threadgroupsPerGrid = MTLSizeMake((totalThreads + w - 1)/w, 1, 1)
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            
            commandEncoder.endEncoding()
            commandBuffer.commit()
            if i == 0{
                axes = int2(1, 1)
                axesBuffer = device.makeBuffer(bytes: &axes, length: MemoryLayout<int2>.stride)!
                w = self.tensorDotDiagPipelineState.threadExecutionWidth
            }
            C1 = outputBuffer!
            C1Descriptor = outputDescriptor
            C1DescriptorBuffer = outputDescriptorBuffer
            i += 1
        }
        return(C1, C1Descriptor)
    }
    
    func calculateC2(withContractedNodes contractedNodes: ([MTLBuffer], [tensorDescriptor]), commandBuffer: inout MTLCommandBuffer!, commandEncoder: inout MTLComputeCommandEncoder!) -> (MTLBuffer, tensorDescriptor){
        var C2 = self.endTensor.makeBuffer(onDevice: self.device)
        var C2Descriptor: tensorDescriptor! = self.endTensor.descriptor
        var C2DescriptorBuffer: MTLBuffer = device.makeBuffer(bytes: &C2Descriptor, length: MemoryLayout<tensorDescriptor>.stride)!
        var i = self.tensors.count - 1
        var w = self.tensorDotPipelineState.threadExecutionWidth
        var axes = int2(0, 2)
        var axesBuffer: MTLBuffer = device.makeBuffer(bytes: &axes, length: MemoryLayout<int2>.stride)!
        var collectedAxes = int2(0, 0)
        let collectedAxesBuffer = device.makeBuffer(bytes: &collectedAxes, length: MemoryLayout<int2>.stride)!
        while i > self.specialNodeLocation {
            let contractedNode = contractedNodes.0[i]
            var contractedNodeDescriptor = contractedNodes.1[i]
            var outputDescriptor = tensorDescriptor(withDimensions: [contractedNodeDescriptor.dimensions.0, contractedNodeDescriptor.dimensions.2])
            let length = Int(outputDescriptor.multipliers.0) * MemoryLayout<Float32>.stride
            let outputBuffer = device.makeBuffer(length: length)
            let outputDescriptorBuffer = device.makeBuffer(bytes: &outputDescriptor, length: MemoryLayout<tensorDescriptor>.stride)!
            let contractedNodeDescriptorBuffer = device.makeBuffer(bytes: &contractedNodeDescriptor, length: MemoryLayout<tensorDescriptor>.stride)!
            commandBuffer.waitUntilCompleted()
            commandBuffer = commandQueue.makeCommandBuffer()!
            commandEncoder = commandBuffer.makeComputeCommandEncoder()!
            if i == (self.tensors.count - 1){
                commandEncoder.setComputePipelineState(self.tensorDotPipelineState)
            }
            else{
                commandEncoder.setComputePipelineState(self.tensorDotDiagPipelineState)
            }
            
            commandEncoder.setBuffer(C2DescriptorBuffer, offset: 0, index: 0)
            commandEncoder.setBuffer(contractedNodeDescriptorBuffer, offset: 0, index: 1)
            commandEncoder.setBuffer(outputDescriptorBuffer, offset: 0, index: 2)
            commandEncoder.setBuffer(axesBuffer, offset: 0, index: 3)
            if i == (self.tensors.count - 1){
                commandEncoder.setBuffer(C2, offset: 0, index: 4)
                commandEncoder.setBuffer(contractedNode, offset:0, index: 5)
                commandEncoder.setBuffer(outputBuffer, offset:0, index:6)
            }
            else{
                commandEncoder.setBuffer(collectedAxesBuffer, offset: 0, index: 4)
                commandEncoder.setBuffer(C2, offset: 0, index: 5)
                commandEncoder.setBuffer(contractedNode, offset:0, index: 6)
                commandEncoder.setBuffer(outputBuffer, offset:0, index:7)
            }
            let totalThreads = Int(outputDescriptor.multipliers.0)
            let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
            let threadgroupsPerGrid = MTLSizeMake((totalThreads + w - 1)/w, 1, 1)
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            
            commandEncoder.endEncoding()
            commandBuffer.commit()
            if i == (self.tensors.count - 1){
                axes = int2(1, 2)
                axesBuffer = device.makeBuffer(bytes: &axes, length: MemoryLayout<int2>.stride)!
                w = self.tensorDotDiagPipelineState.threadExecutionWidth
            }
            C2 = outputBuffer!
            C2Descriptor = outputDescriptor
            C2DescriptorBuffer = outputDescriptorBuffer
            i -= 1
        }
        return(C2, C2Descriptor)
    }
    
    func calculateContractedNodes(forData mpsData: MPSData, commandBuffer: inout MTLCommandBuffer!, commandEncoder: inout MTLComputeCommandEncoder!) -> ([MTLBuffer], [tensorDescriptor]){
        var contractedNodes = [MTLBuffer]()
        var contractedNodesDescriptors = [tensorDescriptor]()
        var i = 0
        let w = self.tensorDotPipelineState.threadExecutionWidth
        // Calculate the contracted nodes
        while i < tensors.count{
            commandBuffer = commandQueue.makeCommandBuffer()!
            commandEncoder = commandBuffer.makeComputeCommandEncoder()!
            commandEncoder.setComputePipelineState(self.tensorDotPipelineState)
            let input = mpsData.dataTensors[i]
            let tensor = self.tensors[i]
            
            var outputDescriptor = tensorDescriptor(withDimensions: [input.descriptor.dimensions.0, tensor.descriptor.dimensions.1, tensor.descriptor.dimensions.2])
            var axes = int2(1, 0)
            var axesBuffer: MTLBuffer = device.makeBuffer(bytes: &axes, length: MemoryLayout<int2>.stride)!
            if i == self.specialNodeLocation{
                axes = int2(1, 1)
                outputDescriptor = tensorDescriptor(withDimensions: [input.descriptor.dimensions.0, tensor.descriptor.dimensions.0, tensor.descriptor.dimensions.2, tensor.descriptor.dimensions.3])
                axesBuffer = device.makeBuffer(bytes: &axes, length: MemoryLayout<int2>.stride)!
            }
            
            let inputBuffer = input.makeBuffer(onDevice: self.device)
            let tensorBuffer = tensor.makeBuffer(onDevice: self.device)
            
            let length = Int(outputDescriptor.multipliers.0) * MemoryLayout<Float32>.stride
            let outputBuffer = device.makeBuffer(length: length)
            
            var descriptorForTensor = tensor.descriptor
            var inputDescriptor = input.descriptor
            let tensorDescriptorBuffer: MTLBuffer = device.makeBuffer(bytes: &descriptorForTensor, length: MemoryLayout<tensorDescriptor>.stride)!
            let inputDescriptorBuffer: MTLBuffer = device.makeBuffer(bytes: &inputDescriptor, length: MemoryLayout<tensorDescriptor>.stride)!
            let outputDescriptorBuffer:MTLBuffer = device.makeBuffer(bytes: &outputDescriptor, length: MemoryLayout<tensorDescriptor>.stride)!

            commandEncoder.setBuffer(inputDescriptorBuffer, offset: 0, index: 0)
            commandEncoder.setBuffer(tensorDescriptorBuffer, offset: 0, index: 1)
            commandEncoder.setBuffer(outputDescriptorBuffer, offset: 0, index: 2)
            commandEncoder.setBuffer(axesBuffer, offset: 0, index: 3)
            commandEncoder.setBuffer(inputBuffer, offset: 0, index: 4)
            commandEncoder.setBuffer(tensorBuffer, offset: 0, index: 5)
            commandEncoder.setBuffer(outputBuffer, offset: 0, index: 6)
            
            let totalThreads = Int(outputDescriptor.multipliers.0)
            let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
            let threadgroupsPerGrid = MTLSizeMake(Int(ceil(Float(totalThreads)/Float(w))), 1, 1)
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            
            commandEncoder.endEncoding()
            commandBuffer.commit()
            contractedNodes.append(outputBuffer!)
            contractedNodesDescriptors.append(outputDescriptor)
            i+=1
        }
        return (contractedNodes, contractedNodesDescriptors)
    }
    
    init(withData data: NSData) {
        var containsWeights: Bool = false
        data.getBytes(&containsWeights, range: NSMakeRange(0, 2))
        
        if containsWeights == false{
            fatalError("MPS configuration file does not contain weights!")
        }
        self.tensors = [Tensor]()
        var inputSizeInt:Int = 0
        var specialNodeLocation:Int = 0
        var dFeatureInt:Int = 0
        var dOutputInt:Int = 0
        var intermediate: Int16 = 0
        data.getBytes(&intermediate, range: NSMakeRange(1, 2))
        inputSizeInt = Int(Int16(bigEndian: intermediate))
        data.getBytes(&intermediate, range: NSMakeRange(3, 2))
        specialNodeLocation = Int(Int16(bigEndian: intermediate))
        data.getBytes(&intermediate, range: NSMakeRange(5, 2))
        dFeatureInt = Int(Int16(bigEndian: intermediate))
        data.getBytes(&intermediate, range: NSMakeRange(7, 2))
        dOutputInt = Int(Int16(bigEndian: intermediate))
        let inputSize = uint(inputSizeInt)
        let dFeature = uint(dFeatureInt)
        let dOutput = uint(dOutputInt)
        self.specialNodeLocation = specialNodeLocation
        print(inputSize, specialNodeLocation, dFeature, dOutput)
        var dimensions = [Int]()
        
        var i = 10
        while i < (10 + (inputSize - 1) * 2){
            var dimension:Int = 0
            data.getBytes(&dimension, range: NSMakeRange(i, 2))
            dimensions.append(dimension)
            i += 2
        }
        var startIndex = i + 1
        i = 0
        var leftDim:uint = uint(dOutput * 2)
        var rightDim = uint(dOutput * 2)
        var length = 0
        var descriptor = tensorDescriptor(withDimensions: [0])
        while i < inputSizeInt{
            leftDim = rightDim
            if i != inputSizeInt - 1{
                rightDim = uint(dimensions[i])
            }
            else{
                rightDim = uint(dOutput * 2)
            }
            if i == specialNodeLocation{
                length = Int(leftDim * rightDim * dFeature * dOutput)
                descriptor = tensorDescriptor(withDimensions: [dOutput, dFeature, leftDim, rightDim])
            }
            else{
                length = Int(leftDim * rightDim * dFeature)
                descriptor = tensorDescriptor(withDimensions: [dFeature, leftDim, rightDim])
            }
            var value = [Float32](repeatElement(0.0, count: Int(length)))
            var j = 0
            while j < length{
                data.getBytes(&value[j], range: NSMakeRange(startIndex, 4))
                startIndex += 4
                j += 1
            }
            tensors.append(Tensor(value: value, descriptor: descriptor))
            i+=1
        }
        
        device = MTLCreateSystemDefaultDevice()
        defaultLibrary = device.makeDefaultLibrary()
        commandQueue = device.makeCommandQueue()
        textureDescriptor.textureType = .type1D
        textureDescriptor.pixelFormat = .r32Float
        textureDescriptor.usage = .shaderRead
        tensorDotFunction = defaultLibrary.makeFunction(name: "tensor_dot")!
        tensorDotDiagFunction = defaultLibrary.makeFunction(name: "tensor_dot_with_diag")
        tensorDotPipelineState = try! device.makeComputePipelineState(function: tensorDotFunction)
        tensorDotDiagPipelineState = try! device.makeComputePipelineState(function: tensorDotDiagFunction)
        
        let startAndEndDescriptor = tensorDescriptor(withDimensions: [dOutput * 2])
        var startVectorValue = [Float32](repeatElement(0.0, count: dOutputInt * 2))
        i = dOutputInt
        while i < dOutputInt * 2{
            startVectorValue[i] = 1.0
            i += 1
        }
        self.startTensor = Tensor(value: startVectorValue, descriptor: startAndEndDescriptor)
        
        var endVectorValue = [Float32](repeatElement(0.0, count: dOutputInt * 2))
        i = 0
        while i < dOutputInt{
            endVectorValue[i] = 1.0
            i += 1
        }
        self.endTensor = Tensor(value: endVectorValue, descriptor: startAndEndDescriptor)
    }
    
    convenience init(withURL URL: URL) {
        try! self.init(withData: NSData(contentsOf: URL))
    }
}



