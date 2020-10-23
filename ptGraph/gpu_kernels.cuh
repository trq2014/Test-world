
#include "range.hpp"
#include "globals.cuh"

using namespace util::lang;


// type alias to simplify typing...
template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template<typename T>
__device__
step_range<T> grid_stride_range(T begin, T end) {
    begin += blockDim.x * blockIdx.x + threadIdx.x;
    return range(begin, end).step(gridDim.x * blockDim.x);
}

template<typename Predicate>
__device__ void streamVertices(int vertices, Predicate p) {
    for (auto i : grid_stride_range(0, vertices)) {
        p(i);
    }
}

__global__ void
bfs_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
           bool *labelD);

__global__ void
cc_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
          bool *labelD);

__global__ void
sssp_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
            uint *valueD,
            bool *labelD);

__global__ void
bfs_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
              bool *labelD, uint overloadNode, uint *overloadEdgeListD,
              uint *nodePointersOverloadD);

__global__ void
bfs_kernelStatic(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
                 uint *labelD);

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, bool *isInD);

__global__ void
bfs_kernelDynamic(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                  uint *labelD, uint overloadNode, uint *overloadEdgeListD,
                  uint *nodePointersOverloadD);

__global__ void
bfs_kernelDynamicSwap(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                      uint *labelD, uint *overloadEdgeListD,
                      uint *nodePointersOverloadD);

__global__ void
sssp_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
               uint *valueD,
               bool *labelD, uint overloadNode, EdgeWithWeight *overloadEdgeListD, uint *nodePointersOverloadD);

__global__ void
cc_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
             bool *labelD, uint overloadNode, uint *overloadEdgeListD, uint *nodePointersOverloadD);

__global__ void
bfs_kernelOptOfSorted(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                      uint *edgeListOverload, uint *valueD, bool *labelD, bool *isInListD, uint *nodePointersOverloadD);

__global__ void
bfs_kernelShareOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                   uint *edgeListShare, uint *valueD, bool *labelD, uint overloadNode);

__global__ void
setLabelDefault(uint activeNum, uint *activeNodes, bool *labelD);

__global__ void
setLabelDefaultOpt(uint activeNum, uint *activeNodes, uint *labelD);

__global__ void
setLabeling(uint vertexNum, bool *labelD, uint *labelingD);

__global__ void
setActiveNodeArray(uint vertexNum, uint *activeNodes, bool *activeLabel, uint *activeLabelPrefix);

__global__ void
setActiveNodeArrayAndNodePointer(uint vertexNum, uint *activeNodes, uint *activeNodePointers, bool *activeLabel,
                                 uint *activeLabelPrefix, uint overloadVertex, uint *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerBySortOpt(uint vertexNum, uint *activeNodes, uint *activeOverloadDegree,
                                          bool *activeLabel, uint *activeLabelPrefix, bool *isInList, uint *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerOpt(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                                    uint *activeLabelPrefix, uint overloadVertex, uint *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerSwap(uint vertexNum, uint *activeNodes, uint *activeLabel,
                                     uint *activeLabelPrefix, bool *isInD);

__global__ void
setOverloadNodePointerSwap(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                           uint *activeLabelPrefix, uint *degreeD);

__global__ void
setFragmentData(uint activeNodeNum, uint *activeNodeList, uint *staticNodePointers, uint *staticFragmentData,
                uint staticFragmentNum, uint fragmentSize, bool* isInStatic);
__global__ void
setStaticFragmentData(uint staticFragmentNum, uint *canSwapFragmentD, uint *canSwapFragmentPrefixD,
                uint *staticFragmentDataD);