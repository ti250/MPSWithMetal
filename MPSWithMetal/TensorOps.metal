//
//  TensorOps.metal
//  TensorOps
//
//  Created by Taketomo Isazawa on 21/9/17.
//  Copyright © 2017 Taketomo Isazawa. All rights reserved.
//

//ISSUE: For some reason the indices for tensorDescriptor seem to be 1-indexed????? Perhaps it's do with some sort of error of where it reads from when it reads in a buffer....

#include <metal_stdlib>
using namespace metal;

#define tensor_max_rank 10
#define multipliers_max_rank 11
#define max_uint 4294967295
// multipliers_max_rank = tensor_max_rank + 1

struct tensorDescriptor{
    int rank;
    array<uint, multipliers_max_rank> multipliers;
    array<uint, tensor_max_rank> dimensions;
};

array<uint, tensor_max_rank> indices_for(uint gid, constant tensorDescriptor &descriptor);
uint gid_for(array<uint, tensor_max_rank> indices, constant tensorDescriptor &descriptor);

kernel void tensor_dot(constant tensorDescriptor &A_descriptor[[buffer(0)]],
                       constant tensorDescriptor &B_descriptor[[buffer(1)]],
                       constant tensorDescriptor &out_descriptor[[buffer(2)]],
                       constant int2 &axes[[buffer(3)]],
                       device float* tensor_A [[buffer(4)]],
                       device float* tensor_B[[buffer(5)]],
                       device float* out_tensor[[buffer(6)]],
                       uint gid [[thread_position_in_grid]]
                       ){
    if (gid >= out_descriptor.multipliers[1]){
        return;
    }
    thread array<uint, tensor_max_rank> out_indices = indices_for(gid, out_descriptor);
    int i = 0;
    thread array<uint, tensor_max_rank> tensor_A_indices;
    thread array<uint, tensor_max_rank> tensor_B_indices;
    while (i < A_descriptor.rank){
        if (i >= axes.x){
            tensor_A_indices[i] = out_indices[i-1];
        }
        else{
            tensor_A_indices[i] = out_indices[i];
        }
        i += 1;
    }
    i = 0;
    while (i < B_descriptor.rank){
        if (i >= axes.y){
            tensor_B_indices[i] = out_indices[i+A_descriptor.rank-2];
        }
        else{
            tensor_B_indices[i] = out_indices[i+A_descriptor.rank-1];
        }
        i += 1;
    }
    uint j = 0;
    thread float result = 0.0;
    thread uint A_gid = 0;
    thread uint B_gid = 0;
    while (j < A_descriptor.dimensions[axes.x+1]){
        tensor_A_indices[axes.x] = j;
        tensor_B_indices[axes.y] = j;
        A_gid = gid_for(tensor_A_indices, A_descriptor);
        B_gid = gid_for(tensor_B_indices, B_descriptor);
        result += tensor_A[A_gid] * tensor_B[B_gid];
        j += 1;
    }
    out_tensor[gid] = result;
}

kernel void tensor_dot_with_diag(constant tensorDescriptor &A_descriptor[[buffer(0)]],
                                 constant tensorDescriptor &B_descriptor[[buffer(1)]],
                                 constant tensorDescriptor &out_descriptor[[buffer(2)]],
                                 constant int2 &axes[[buffer(3)]],
                                 constant int2 &collected_axes[[buffer(4)]],
                                 device float* tensor_A [[buffer(5)]],
                                 device float* tensor_B[[buffer(6)]],
                                 device float* out_tensor[[buffer(7)]],
                                 uint gid [[thread_position_in_grid]]){
    // Collected axes refers to the dimensions of A and B (one from each) along which the "Diagonal part" is taken
    // Currently, collected axes[1] needs to be 0, or won't work
    // TODO: Fix that collected axes[1] needs to be 0, or won't work
    if (gid >= out_descriptor.multipliers[1]){
        return;
    }
    thread array<uint, tensor_max_rank> out_indices = indices_for(gid, out_descriptor);
    int i = 0;
    thread array<uint, tensor_max_rank> tensor_A_indices;
    thread array<uint, tensor_max_rank> tensor_B_indices;
    while (i < A_descriptor.rank){
        if (i >= axes.x){
            tensor_A_indices[i] = out_indices[i-1];
        }
        else{
            tensor_A_indices[i] = out_indices[i];
        }
        i += 1;
    }
    i = 0;
    while (i < B_descriptor.rank){
        if (i >= axes.y){
            tensor_B_indices[i] = out_indices[i+A_descriptor.rank-3];
        }
        else if (i == collected_axes.y){
            tensor_B_indices[i] = tensor_A_indices[collected_axes.x];
        }
        else{
            tensor_B_indices[i] = out_indices[i+A_descriptor.rank-2];
        }
        i += 1;
    }
    uint j = 0;
    thread float result = 0.0;
    thread uint A_gid = 0;
    thread uint B_gid = 0;
    while (j < A_descriptor.dimensions[axes.x+1]){
        tensor_A_indices[axes.x] = j;
        tensor_B_indices[axes.y] = j;
        A_gid = gid_for(tensor_A_indices, A_descriptor);
        B_gid = gid_for(tensor_B_indices, B_descriptor);
        result += tensor_A[A_gid] * tensor_B[B_gid];
        j += 1;
    }
    out_tensor[gid] = result;
}

thread array<uint, tensor_max_rank> indices_for(uint gid,
                            constant tensorDescriptor &descriptor){
    array<uint, tensor_max_rank> indices;
    thread float remaining_gid = float(gid);
    int i = 0;
    while (i < descriptor.rank){
        float divided = remaining_gid/(descriptor.multipliers[i+2]);
        float floored_divided = floor(divided);
        if (floored_divided >= descriptor.dimensions[i+1]){
            floored_divided = 0.0;
        }
        remaining_gid = round(remaining_gid - floored_divided* descriptor.multipliers[i+2]);
        indices[i] = uint(floored_divided);
        i += 1;
    }
    return indices;
}

thread uint gid_for(array<uint, tensor_max_rank> indices,
             constant tensorDescriptor &descriptor){
    int i = 0;
    int gid = 0;
    while (i < descriptor.rank){
        gid += indices[i] * descriptor.multipliers[i+2];
        i += 1;
    }
    return gid;
}
