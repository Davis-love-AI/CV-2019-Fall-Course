// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;
#define maxn 510
const float eps=1E-8;
int const threadsPerBlock = 512; //sizeof(unsigned long long) * 8;


__device__ inline int sig(float d){
    return(d>eps)-(d<-eps);
}

struct Point{
    float x,y;
    __device__ Point(){}
    __device__ Point(float x,float y):x(x),y(y){}
};

__device__ inline bool point_same(Point& a, Point& b){
    return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

__device__ inline void swap1(Point* a, Point* b){
    Point temp;
    temp.x = a->x;
    temp.y = a->y;

    a->x = b->x;
    a->y = b->y;

    b->x = temp.x;
    b->y = temp.y;
}

__device__ inline void reverse1(Point* a, const int n){
    Point temp[maxn];
    for(int i = 0; i < n; i++){
        temp[i].x = a[i].x;
        temp[i].y = a[i].y;
    }
    for(int i = 0; i < n; i++){
        a[i].x = temp[n - 1 - i].x;
        a[i].y = temp[n - 1 - i].y;
    }
}

__device__ inline float cross(Point o,Point a,Point b){  //叉积
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}

__device__ inline float area(Point* ps,int n){
    ps[n]=ps[0];
    float res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
__device__ inline int lineCross(Point a,Point b,Point c,Point d,Point&p){
    float s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}
//多边形切割
//用直线ab切割多边形p，切割后的在向量(a,b)的左侧，并原地保存切割结果
//如果退化为一个点，也会返回去,此时n为1
__device__ inline void polygon_cut(Point*p,int&n,Point a,Point b){
    Point pp[maxn];
    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
            lineCross(a,b,p[i],p[i+1],pp[m++]);
    }
    n=0;
    for(int i=0;i<m;i++)
          if(!i || !(point_same(pp[i], pp[i-1])))
    		p[n++]=pp[i];
    while(n > 1 && point_same(p[n-1], p[0]))n--;
}
//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
__device__ inline float intersectArea(Point a,Point b,Point c,Point d){
    Point o(0,0);
    int s1=sig(cross(o,a,b));
    int s2=sig(cross(o,c,d));
    if(s1==0||s2==0)return 0.0;//退化，面积为0
    if(s1==-1){
    	Point* i = &a;
    	Point* j = &b;
    	swap1(i, j);
    }
    if(s2==-1){
    	Point* i = &c;
    	Point* j = &d;
    	swap1(i, j);
    }
    Point p[10]={o,a,b};
    int n=3;
    polygon_cut(p,n,o,c);
    polygon_cut(p,n,c,d);
    polygon_cut(p,n,d,o);
    float res=fabs(area(p,n));
    if(s1*s2==-1) res=-res;return res;
}
//求两多边形的交面积
__device__ inline float intersectAreaO(Point*ps1,int n1,Point*ps2,int n2){
    if(area(ps1,n1)<0) reverse1(ps1,n1);
    if(area(ps2,n2)<0) reverse1(ps2,n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    float res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;
}


__device__ inline float devrIoU(float const * const p, float const * const q) {
    Point ps1[maxn],ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    float inter_area = intersectAreaO(ps1, n1, ps2, n2);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    float iou = inter_area / union_area;
    //printf("ex:%f, %f, %f, %f, %f, %f, %f, %f\n  gt:%f, %f, %f, %f, %f, %f, %f, %f\n  iou:%f\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[07], iou);
    return iou;

}

__global__ void riou_kernel(const int ex_n_boxes, const int gt_n_boxes, 
                            const float *ex_boxes, const float *gt_boxes,
                            float* iou) {
  const int ex_start = blockIdx.x;

  //const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int ex_size = min(ex_n_boxes - ex_start * threadsPerBlock, threadsPerBlock);
  //const int col_size =
  //      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

//  __shared__ float block_boxes[threadsPerBlock * 8];
// if (threadIdx.x < ex_size) {
//    block_boxes[threadIdx.x * 8 + 0] =
//        ex_boxes[(threadsPerBlock * ex_start + threadIdx.x) * 8 + 0];
//    block_boxes[threadIdx.x * 8 + 1] =
//        ex_boxes[(threadsPerBlock * ex_start + threadIdx.x) * 8 + 1];
//    block_boxes[threadIdx.x * 8 + 2] =
//        ex_boxes[(threadsPerBlock * ex_start + threadIdx.x) * 8 + 2];
//    block_boxes[threadIdx.x * 8 + 3] =
//        ex_boxes[(threadsPerBlock * ex_start + threadIdx.x) * 8 + 3];
//    block_boxes[threadIdx.x * 8 + 4] =
//        ex_boxes[(threadsPerBlock * ex_start + threadIdx.x) * 8 + 4];
//    block_boxes[threadIdx.x * 8 + 5] =
//        ex_boxes[(threadsPerBlock * ex_start + threadIdx.x) * 8 + 5];
//    block_boxes[threadIdx.x * 8 + 6] =
//        ex_boxes[(threadsPerBlock * ex_start + threadIdx.x) * 8 + 6];
//    block_boxes[threadIdx.x * 8 + 7] =
//      ex_boxes[(threadsPerBlock * ex_start + threadIdx.x) * 8 + 7];
//}
//  __syncthreads();

  if (threadIdx.x < ex_size) {
    const int cur_box_idx = threadsPerBlock * ex_start + threadIdx.x;
    const float *cur_box = ex_boxes + cur_box_idx * 8;
    for(int i = 0; i < gt_n_boxes; i++){
      iou[cur_box_idx * gt_n_boxes + i] = devrIoU(cur_box, gt_boxes + i * 8);
      //printf("iou: %f\n", iou[cur_box_idx * gt_n_boxes + i]);
      }
  }
//    int i = 0;
//    unsigned long long t = 0;
//    int start = 0;
//    if (row_start == col_start) {
//      start = threadIdx.x + 1;
//    }
//    for (i = start; i < col_size; i++) {
//      if (devrIoU(cur_box, block_boxes + i * 8) > nms_overlap_thresh) {
//        t |= 1ULL << i;
//      }
//    }
//    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
//    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  
}

// should be N x 8 tensor
at::Tensor riou_cuda(const at::Tensor ex_boxes, const at::Tensor gt_boxes) {
  using scalar_t = float;
  AT_ASSERTM(ex_boxes.type().is_cuda(), "ex_boxes must be a CUDA tensor");
  AT_ASSERTM(gt_boxes.type().is_cuda(), "gt_boxes must be a CUDA tensor");
  int ex_boxes_num = ex_boxes.size(0);
  int gt_boxes_num = gt_boxes.size(0);
  //cout << "ex_num:" << ex_boxes_num << endl;
  //cout << "gt_num:" << gt_boxes_num << endl;
  const int ex_blocks = THCCeilDiv(ex_boxes_num, threadsPerBlock);
  //cout << "ex_blocks:" << ex_blocks << endl;
  scalar_t* ex_boxes_dev = ex_boxes.data<scalar_t>();
  scalar_t* gt_boxes_dev = gt_boxes.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  float* iou_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  //iou_dev = (float*) THCudaMalloc(state, (ex_boxes_num * gt_boxes_num) * ex_blocks * sizeof(float));
  iou_dev = (float*) THCudaMalloc(state, (ex_boxes_num * gt_boxes_num) * sizeof(float));
  
  dim3 blocks(THCCeilDiv(ex_boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  riou_kernel<<<blocks, threads>>>(ex_boxes_num,
                                  gt_boxes_num,
                                  ex_boxes_dev,
                                  gt_boxes_dev,
                                  iou_dev);
  //cudaError_t error = cudaGetLastError();
  //printf("CUDA error: %s\n", cudaGetErrorString(error));

  //std::vector<float> iou_host((ex_boxes_num * gt_boxes_num) * ex_blocks);
  std::vector<float> iou_host((ex_boxes_num * gt_boxes_num));
  THCudaCheck(cudaMemcpy(&iou_host[0],
                        iou_dev,
                        sizeof(float) * (ex_boxes_num * gt_boxes_num),
                        cudaMemcpyDeviceToHost));
  //for(int i = 0; i < (ex_boxes_num * gt_boxes_num); i++){
  //  cout << "iou_host:" << iou_host[i] << endl;
  //}
  at::Tensor overlaps = at::empty({gt_boxes_num * ex_boxes_num}, ex_boxes.options().dtype(at::kFloat).device(at::kCPU));

  float* overlaps_out = overlaps.data<float>();

  //cudaError_t error = cudaGetLastError();
  //printf("CUDA error: %s\n", cudaGetErrorString(error));

  for(int i = 0; i < (ex_boxes_num * gt_boxes_num); i++){
    overlaps_out[i] = iou_host[i];
  }
  //for(int i = ex_boxes_num * gt_boxes_num - 10; i < (ex_boxes_num * gt_boxes_num); i++){
  //  cout << "output:" << overlaps_out[i] << endl;
  //}
  THCudaFree(state, iou_dev);
  // TODO improve this part
  return overlaps.to(ex_boxes.device()); 
  //return std::get<0>(order_t.index({
  //                     keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
  //                       order_t.device(), keep.scalar_type())
  //                   }).sort(0, false));
}
