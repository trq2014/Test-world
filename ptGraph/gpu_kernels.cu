#include "gpu_kernels.cuh"

__global__ void
setLabeling(uint vertexNum, bool *labelD, uint *labelingD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (labelD[vertexId]) {
            labelingD[vertexId] = 1;
            //printf("vertex[%d] set 1\n", vertexId);
        } else {
            labelingD[vertexId] = 0;
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointerOpt(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                                    uint *activeLabelPrefix, uint overloadVertex, uint *degreeD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (vertexId > overloadVertex) {
                activeNodePointers[activeLabelPrefix[vertexId]] = degreeD[vertexId];
                activeLabel[vertexId] = 1;
            } else {
                activeNodePointers[activeLabelPrefix[vertexId]] = 0;
                activeLabel[vertexId] = 0;
            }
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointerSwap(uint vertexNum, uint *activeNodes, uint *activeLabel,
                                     uint *activeLabelPrefix, bool *isInD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (!isInD[vertexId]) {
                activeLabel[vertexId] = 1;
            } else {
                activeLabel[vertexId] = 0;
            }
        }
    });
}

__global__ void
setOverloadNodePointerSwap(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                           uint *activeLabelPrefix, uint *degreeD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            activeNodePointers[activeLabelPrefix[vertexId]] = degreeD[vertexId];
        }
    });
}

__global__ void
setActiveNodeArray(uint vertexNum, uint *activeNodes, bool *activeLabel,
                   uint *activeLabelPrefix) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointer(uint vertexNum, uint *activeNodes, uint *activeNodePointers, bool *activeLabel,
                                 uint *activeLabelPrefix, uint overloadVertex, uint *degreeD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (vertexId > overloadVertex) {
                activeNodePointers[activeLabelPrefix[vertexId]] = degreeD[vertexId];
            } else {
                activeNodePointers[activeLabelPrefix[vertexId]] = 0;
            }
        }
    });
}

__global__ void
setActiveNodeArrayAndNodePointerBySortOpt(uint vertexNum, uint *activeNodes, uint *activeOverloadDegree,
                                          bool *activeLabel, uint *activeLabelPrefix, bool *isInList, uint *degreeD) {
    streamVertices(vertexNum, [&](uint vertexId) {
        if (activeLabel[vertexId]) {
            activeNodes[activeLabelPrefix[vertexId]] = vertexId;
            if (!isInList[vertexId]) {
                activeOverloadDegree[activeLabelPrefix[vertexId]] = degreeD[vertexId];
            } else {
                activeOverloadDegree[activeLabelPrefix[vertexId]] = 0;
            }
        }
    });
}

__global__ void
setLabelDefault(uint activeNum, uint *activeNodes, bool *labelD) {
    streamVertices(activeNum, [&](uint vertexId) {
        if (labelD[activeNodes[vertexId]]) {
            labelD[activeNodes[vertexId]] = 0;
            //printf("vertex%d index %d true to %d \n", vertexId, activeNodes[vertexId], labelD[activeNodes[vertexId]]);
        }
    });
}

__global__ void
setLabelDefaultOpt(uint activeNum, uint *activeNodes, uint *labelD) {
    streamVertices(activeNum, [&](uint vertexId) {
        if (labelD[activeNodes[vertexId]]) {
            labelD[activeNodes[vertexId]] = 0;
            //printf("vertex%d index %d true to %d \n", vertexId, activeNodes[vertexId], labelD[activeNodes[vertexId]]);
        }
    });
}

__global__ void
bfs_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
           bool *labelD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            if (finalValue < valueD[edgeListD[i]]) {
                atomicMin(&valueD[edgeListD[i]], finalValue);
                labelD[edgeListD[i]] = true;
            }
        }
    });
}

__global__ void
cc_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
          bool *labelD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            if (sourceValue < valueD[edgeListD[i]]) {
                atomicMin(&valueD[edgeListD[i]], sourceValue);
                labelD[edgeListD[i]] = true;
            }
        }
    });
}

__global__ void
sssp_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
            uint *valueD,
            bool *labelD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
            finalValue = sourceValue + edgeListD[i].weight;
            uint vertexId = edgeListD[i].toNode;

            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
                //printf("source vertex %d, toNode is %d \n", id, vertexId);
            }
        }
    });
}

__global__ void
bfs_kernelShareOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                   uint *edgeListShare, uint *valueD, bool *labelD, uint overloadNode) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        if (id >= overloadNode) {
            uint edgeIndex = nodePointersD[id];
            uint sourceValue = valueD[id];
            uint finalValue;
            for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                if (finalValue < valueD[edgeListShare[i]]) {
                    atomicMin(&valueD[edgeListShare[i]], finalValue);
                    labelD[edgeListShare[i]] = true;
                    //printf("vertext[%d](edge[%d]) set 1\n", edgeListD[i], i);
                }
            }
        } else {
            uint edgeIndex = nodePointersD[id];
            uint sourceValue = valueD[id];
            uint finalValue;
            for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                if (finalValue < valueD[edgeListD[i]]) {
                    atomicMin(&valueD[edgeListD[i]], finalValue);
                    labelD[edgeListD[i]] = true;
                    //printf("vertext[%d](edge[%d]) set 1\n", edgeListD[i], i);
                }
            }
        }

        //printf("index %d vertex %d edgeIndex %d degree %d sourcevalue %d \n", index, id, edgeIndex, degreeD[id], sourceValue);
    });
}

__global__ void
bfs_kernelStatic(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
                 uint *labelD) {
    streamVertices(nodeNum, [&](uint index) {
        uint id = activeNodesD[index];

        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId;
            vertexId = edgeListD[edgeIndex + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, bool *isInD) {
    streamVertices(nodeNum, [&](uint index) {
        uint id = activeNodesD[index];
        if (isInD[id]) {
            uint edgeIndex = nodePointersD[id];
            uint sourceValue = valueD[id];
            uint finalValue;
            for (uint i = 0; i < degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                uint vertexId;
                vertexId = edgeListD[edgeIndex + i];
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    labelD[vertexId] = 1;
                }
            }
        }
    });
}

/*__global__ void
bfs_kernelDynamic(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                  uint *labelD, uint overloadNode, uint *overloadEdgeListD,
                  uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint sourceValue = valueD[id];
        uint finalValue;
        if (id > overloadNode) {
            for (uint i = 0; i < degreeD[id]; i++) {
                finalValue = sourceValue + 1;
                uint vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
                if (finalValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], finalValue);
                    labelD[vertexId] = 1;
                }
            }
        }
    });
}*/

__global__ void
bfs_kernelDynamic(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                  uint *labelD, uint overloadNode, uint *overloadEdgeListD,
                  uint *nodePointersOverloadD) {
    streamVertices(overloadNode, [&](uint index) {
        uint theIndex = activeNum - overloadNode + index;
        uint id = activeNodesD[theIndex];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId = overloadEdgeListD[nodePointersOverloadD[theIndex] + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

__global__ void
bfs_kernelDynamicSwap(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                      uint *labelD, uint *overloadEdgeListD,
                      uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = 1;
            }
        }
    });
}

__global__ void
bfs_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
              uint *valueD,
              bool *labelD, uint overloadNode, uint *overloadEdgeListD, uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];

        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId;
            if (id > overloadNode) {
                vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
            } else {
                vertexId = edgeListD[edgeIndex + i];
            }
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
            }
        }
    });
}

__global__ void
cc_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
             bool *labelD, uint overloadNode, uint *overloadEdgeListD, uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        for (uint i = 0; i < degreeD[id]; i++) {
            uint vertexId;
            if (id > overloadNode) {
                vertexId = overloadEdgeListD[nodePointersOverloadD[index] + i];
            } else {
                vertexId = edgeListD[edgeIndex + i];
            }
            if (sourceValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], sourceValue);
                labelD[vertexId] = true;
            }
        }
    });
}

__global__ void
sssp_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
               uint *valueD,
               bool *labelD, uint overloadNode, EdgeWithWeight *overloadEdgeListD, uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];

        uint edgeIndex = nodePointersD[id];
        uint sourceValue = valueD[id];
        uint finalValue;
        for (uint i = 0; i < degreeD[id]; i++) {
            EdgeWithWeight checkNode;
            if (id > overloadNode) {
                checkNode = overloadEdgeListD[nodePointersOverloadD[index] + i];
            } else {
                checkNode = edgeListD[edgeIndex + i];
            }
            finalValue = sourceValue + checkNode.weight;
            uint vertexId = checkNode.toNode;
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
                //printf("source vertex %d, toNode is %d \n", id, vertexId);
            }
        }
    });
}

__global__ void
bfs_kernelOptOfSorted(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                      uint *edgeListOverload, uint *valueD, bool *labelD, bool *isInListD,
                      uint *nodePointersOverloadD) {
    streamVertices(activeNum, [&](uint index) {
        uint id = activeNodesD[index];
        uint sourceValue = valueD[id];
        uint finalValue;
        uint edgeIndex;
        uint *edgeList;
        if (!isInListD[id]) {
            edgeIndex = nodePointersOverloadD[index];
            edgeList = edgeListOverload;
        } else {
            edgeIndex = nodePointersD[id];
            edgeList = edgeListD;
        }

        for (uint i = 0; i < degreeD[id]; i++) {
            finalValue = sourceValue + 1;
            uint vertexId = edgeList[edgeIndex + i];
            if (finalValue < valueD[vertexId]) {
                atomicMin(&valueD[vertexId], finalValue);
                labelD[vertexId] = true;
                //printf("vertext[%d](edge[%d]) set 1\n", edgeListD[i], i);
            }
        }
    });
}

__global__ void
setFragmentData(uint activeNodeNum, uint *activeNodeList, uint *staticNodePointers, uint *staticFragmentData,
                uint staticFragmentNum, uint fragmentSize, bool* isInStatic) {
    streamVertices(activeNodeNum, [&](uint index) {
        uint vertexId = activeNodeList[index];
        if (isInStatic[vertexId]) {
            uint staticFragmentIndex = staticNodePointers[vertexId] / fragmentSize;
            if (staticFragmentIndex < staticFragmentNum) {
                staticFragmentData[staticFragmentIndex] = 1;
            }
        }
    });
}

__global__ void
setStaticFragmentData(uint staticFragmentNum, uint *canSwapFragmentD, uint *canSwapFragmentPrefixD,
                uint *staticFragmentDataD) {
    streamVertices(staticFragmentNum, [&](uint index) {
        if (canSwapFragmentD[index] > 0) {
            staticFragmentDataD[canSwapFragmentPrefixD[index]] = index;
            canSwapFragmentD[index] = 0;
        }
    });
}
