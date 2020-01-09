#include <torch/types.h>
#include <stdio.h>
#include <cuda.h>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

#define _FLT_MAX 3.402823466e+38F

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N){
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void AlignFeatureKernel(
    const uint32 nthreads, 
    const torch::PackedTensorAccessor<Dtype, 4, torch::RestrictPtrTraits, size_t> feature_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 H,
    const uint8 W,
    const uint8 nOrientation,
    torch::PackedTensorAccessor<uint8, 4, torch::RestrictPtrTraits, size_t> mainDirection_data,
    torch::PackedTensorAccessor<Dtype, 4, torch::RestrictPtrTraits, size_t> aligned_data)
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        uint8 l, h, w;
        
        const uint16 j = n % nFeature;
        const uint16 i = n / nFeature;
        
        for(h = 0; h < H; h++){
        	for(w = 0; w < W; w++){
        		Dtype maxVal = -_FLT_MAX;
				for (l = 0; l < nOrientation; l++) {
					Dtype val = feature_data[i][j * nOrientation + l][h][w];
					if (val > maxVal) {
						maxVal = val;
						mainDirection_data[i][j][h][w] = l;
					}
				}
				
		        for (l = 0; l < nOrientation; l++) {
		        	Dtype src = feature_data[i][j * nOrientation + l][h][w];
		            uint8 alignedIndex = (l - mainDirection_data[i][j][h][w] + nOrientation) % nOrientation;
		            aligned_data[i][j * nOrientation + alignedIndex][h][w] = src;
		        }
        	}
        }
    }
}

std::vector<torch::Tensor> RIE_AlignFeature_forward_cuda(
    const torch::Tensor feature,
    const uint8 nOrientation)
{
    AT_ASSERTM(feature.type().is_cuda(), "feature must be a CUDA tensor");
//    AT_ASSERTM((feature.size(2) == 1) and (feature.size(3) == 1), "feature must be 1-D tensor in dim=2, 3");
    
    const uint16 nBatch = feature.size(0);
    const uint16 nChannel = feature.size(1);
    const uint8 H = feature.size(2);
    const uint8 W = feature.size(3);
    const uint16 nFeature = nChannel / nOrientation;
    const uint32 count = nBatch * nFeature;
    
    const auto feature_data = feature;
    auto mainDirection_data = torch::zeros({nBatch, nFeature, H, W}, feature.options().dtype(at::kByte).device(at::kCUDA));
    auto aligned_data = torch::zeros({nBatch, nChannel, H, W}, feature.options().dtype(at::kFloat).device(at::kCUDA));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(feature.type(), "rie_cuda_forward", [&] {
    	AlignFeatureKernel<scalar_t> <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>(
                count,
                feature_data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                nBatch,
                nFeature,
                H,
                W,
                nOrientation,
                mainDirection_data.packed_accessor<uint8, 4, torch::RestrictPtrTraits, size_t>(),
    			aligned_data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
    });
        
	THCudaCheck(cudaGetLastError());
	return {aligned_data, mainDirection_data};
}

template <typename Dtype>
__global__ void UnAlignFeatureKernel(
    const uint32 nthreads,
    const torch::PackedTensorAccessor<Dtype, 4, torch::RestrictPtrTraits, size_t> feature_data,
    const uint16 nBatch,
    const uint16 nFeature,
    const uint8 H,
    const uint8 W,
    const uint8 nOrientation,
    torch::PackedTensorAccessor<uint8, 4, torch::RestrictPtrTraits, size_t> mainDirection_data,
    torch::PackedTensorAccessor<Dtype, 4, torch::RestrictPtrTraits, size_t> unaligned_data)
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        uint8 l, w, h;
        
        const uint16 j = n % nFeature;
        const uint16 i = n / nFeature;
        for(h = 0; h < H; h++){
        	for(w = 0; w < W; w++){
        		for (l = 0; l < nOrientation; l++) {
        		        	Dtype src = feature_data[i][j * nOrientation + l][h][w];
        		            uint8 alignedIndex = (l + mainDirection_data[i][j][h][w]) % nOrientation;
        		            unaligned_data[i][j * nOrientation + alignedIndex][h][w] = src;
				}
        	}
        }
        
    }
}

torch::Tensor RIE_AlignFeature_backward_cuda(
    const torch::Tensor feature,	//feature is the align output grad paras
    const torch::Tensor mainDirection,
    const uint8 nOrientation)
{
    AT_ASSERTM(feature.type().is_cuda(), "feature must be a CUDA tensor");
//    AT_ASSERTM((feature.size(2) == 1) and (feature.size(3) == 1), "feature must be 1-D tensor in dim=2, 3");

    const uint16 nBatch = feature.size(0);
    const uint16 nChannel = feature.size(1);
    const uint8 H = feature.size(2);
	const uint8 W = feature.size(3);
    const uint16 nFeature = nChannel / nOrientation;
    const uint32 count = nBatch * nFeature;

    const auto feature_data = feature;		
    const auto mainDirection_data = mainDirection;
    auto unaligned_data = torch::zeros({nBatch, nChannel, H, W}, feature.options().dtype(at::kFloat).device(at::kCUDA));
    //feature is the align output grad paras
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(feature.type(), "rie_cuda_backward", [&] {
    	UnAlignFeatureKernel<scalar_t> <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>(
                count,
                feature_data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                nBatch,
                nFeature,
                H,
                W,
                nOrientation,
                mainDirection_data.packed_accessor<uint8, 4, torch::RestrictPtrTraits, size_t>(),
    			unaligned_data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
    });

	THCudaCheck(cudaGetLastError());
	return unaligned_data;
}