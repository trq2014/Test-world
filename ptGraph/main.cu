#include <iostream>
#include <chrono>
#include <fstream>
#include <math.h>
#include "gpu_kernels.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <algorithm>
#include <thread>

using namespace std;
#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void convertBncr(uint vertexNum, ulong edgeNum, uint *nodePointer, uint *edgeList);

void
readGraphFromJava(string filePath, uint &testNumNodes, ulong &testNumEdge, ulong *nodePointersUL, uint *nodePointersI,
                  uint *edgeList, bool isNeedConvert);

void caculateInCommon(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

long bfsCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long bfsCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long bfsCaculateInAsync(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long bfsCaculateInAsyncSwap(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode);

long ssspCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                         uint sourceNode);

long
ssspCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList, uint sourceNode);

long ccCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

long ccCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

long prCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

long prCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

void caculateInShareOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

void caculateInOptChooseByDegree(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList);

void testBFS();

void testSSSP();

void testCC();

void testPagerank();

uint fragment_size = 4096;
//string testGraphPath = "/home/gxl/labproject/subway/uk-2007-04/output.txt";
string converPath = "/home/gxl/labproject/subway/uk-2007-04/uk-2007-04.bcsr";
//string testGraphPath = "/home/gxl/labproject/subway/uk-2007-04/uk-2007-04.bcsr";
//string testGraphPath = "/home/gxl/labproject/subway/friendster.bcsr";
string testGraphPath = "/home/gxl/labproject/subway/friendster.bcsr";
string testWeightGraphPath = "/home/gxl/labproject/subway/sk-2005.bwcsr";

uint DIST_INFINITY = std::numeric_limits<unsigned int>::max() - 1;

void convertBwcsr();

int main() {
    cudaFree(0);
    //convertBwcsr();
    testBFS();
    //testSSSP();
    //testCC();
    //testPagerank();
    return 0;
}

void testPagerank() {
    cout << "===============PR==============" << endl;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    ulong *nodePointersUL;
    uint *nodePointersI;
    uint *edgeList;
    bool isUseShare = true;

    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(testGraphPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;

    if (isUseShare) {
        gpuErrorcheck(cudaMallocManaged(&nodePointersI, (testNumNodes + 1) * sizeof(uint)));
        infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
        gpuErrorcheck(cudaMallocManaged(&edgeList, (numEdge) * sizeof(uint)));
        infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
        infile.close();
    } else {
        nodePointersI = new uint[testNumNodes];
        infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
        edgeList = new uint[testNumEdge];
        infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
        infile.close();
    }
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    uint sourceNode = rand() % testNumNodes;
    cout << "sourceNode " << sourceNode << endl;
    if (isUseShare) {
        ccCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList);
        //caculateInShareOpt(testNumNodes, testNumEdge, nodePointersI, edgeList);
    } else {
        //caculateInOptChooseByDegree(testNumNodes, testNumEdge, nodePointersI, edgeList);
        ccCaculateInOpt(testNumNodes, testNumEdge, nodePointersI, edgeList);
    }
}

void testCC() {
    cout << "===============CC==============" << endl;
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    ulong *nodePointersUL;
    uint *nodePointersI;
    uint *edgeList;
    bool isUseShare = true;

    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(testGraphPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;

    if (isUseShare) {
        gpuErrorcheck(cudaMallocManaged(&nodePointersI, (testNumNodes + 1) * sizeof(uint)));
        infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
        gpuErrorcheck(cudaMallocManaged(&edgeList, (numEdge) * sizeof(uint)));
        infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
        infile.close();
    } else {
        nodePointersI = new uint[testNumNodes];
        infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
        edgeList = new uint[testNumEdge];
        infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
        infile.close();
    }
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    int testTimes = 64;
    long timeSum = 0;
    for (int i = 0; i < testTimes; i++) {
        uint sourceNode = rand() % testNumNodes;
        cout << "sourceNode " << sourceNode << endl;
        if (isUseShare) {
            timeSum += ccCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList);
            break;
            //caculateInShareOpt(testNumNodes, testNumEdge, nodePointersI, edgeList);
        } else {
            //caculateInOptChooseByDegree(testNumNodes, testNumEdge, nodePointersI, edgeList);
            timeSum += ccCaculateInOpt(testNumNodes, testNumEdge, nodePointersI, edgeList);
            break;
        }
        cout << i << "========================================" << endl;
    }

    cout << "mean time is " << timeSum / testTimes << endl;
}

//std::thread
void ccDynamic(int tId,
               int numThreads,
               unsigned int overloadNodeBegin,
               unsigned int numActiveNodes,
               unsigned int *outDegree,
               unsigned int *activeNodesPointer,
               unsigned int *nodePointer,
               unsigned int *activeNodes,
               uint *edgeListOverload,
               uint *edgeList) {

    unsigned int chunkSize = ceil((numActiveNodes - overloadNodeBegin) / numThreads);
    unsigned int left, right;
    left = tId * chunkSize + overloadNodeBegin;
    right = min(left + chunkSize, numActiveNodes);

    unsigned int thisNode;
    unsigned int thisDegree;
    unsigned int fromHere;
    unsigned int fromThere;

    for (unsigned int i = left; i < right; i++) {
        thisNode = activeNodes[i];
        thisDegree = outDegree[thisNode];
        fromHere = activeNodesPointer[i];
        fromThere = nodePointer[thisNode];

        for (unsigned int j = 0; j < thisDegree; j++) {
            edgeListOverload[fromHere + j] = edgeList[fromThere + j];
        }
    }

}

long ccCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList) {

    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    degree = new uint[testNumNodes];
    value = new uint[testNumNodes];
    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate max memory
    int deviceID;
    cudaDeviceProp dev;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);


    unsigned long max_partition_size = 0.7 * (dev.totalGlobalMem - 10 * 4 * testNumNodes) / sizeof(uint);

    uint *edgeListD;
    gpuErrorcheck(cudaMalloc(&edgeListD, max_partition_size * sizeof(uint)));

    gpuErrorcheck(cudaMemcpy(edgeListD, edgeList, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));

    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    printf("size of edge is %d, max memory is %ld, most edge size is %ld\n multiprocessors %d", sizeof(uint),
           dev.totalGlobalMem, max_partition_size, dev.multiProcessorCount);
    if (max_partition_size > DIST_INFINITY) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = DIST_INFINITY;
    }

    //caculate degree
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];

    int maxPartitionNode = testNumNodes - 1;
    bool hasMaxPartitionNode = false;
    if (max_partition_size < testNumEdge) {
        for (uint i = testNumNodes - 1; i >= 0; i--) {
            if (nodePointersI[i] < max_partition_size) {
                maxPartitionNode = i - 1;
                break;
            }
        }
    }
    max_partition_size = nodePointersI[maxPartitionNode + 1];
    cout << "max_partition_size: " << max_partition_size << "  testNumEdge: " << testNumEdge << endl;

    cout << "maxPartitionNode: " << maxPartitionNode << endl;

    bool *label;
    label = new bool[testNumNodes];
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = true;
        value[i] = i;
    }
    uint *nodePointerD;
    uint *degreeD;
    bool *labelD;
    uint *valueD;
    uint *activeNodeList;
    uint *activeOverloadNodePointers;
    uint *activeNodeListD;
    uint *activeNodeLabelingD;
    uint *activeNodeLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegree;
    activeNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];

    uint *edgeListOverload;
    gpuErrorcheck(cudaMallocManaged(&edgeListOverload, (testNumEdge - max_partition_size) * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&nodePointerD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&labelD, testNumNodes * sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegree, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(nodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(labelD, label, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegree);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(activeOverloadNodePointersD);

    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = 1;
    uint overloadEdgeSum = 0;
    auto startProcessing = std::chrono::steady_clock::now();
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArrayAndNodePointer<<<grid, block>>>(testNumNodes, activeNodeListD, activeOverloadDegree,
                                                          labelD, activeNodeLabelingPrefixD, maxPartitionNode, degreeD);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        uint overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                              ptrOverloadDegree + activeNodesNum, 0);
        overloadEdgeSum += overloadEdgeNum;
        thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + activeNodesNum, activeOverloadNodePointersD);
        startCpu = std::chrono::steady_clock::now();
        if (overloadEdgeNum > 0) {
            cudaMemcpy(activeNodeList, activeNodeListD, activeNodesNum * sizeof(uint), cudaMemcpyDeviceToHost);
            cudaMemcpy(activeOverloadNodePointers, activeOverloadNodePointersD, activeNodesNum * sizeof(uint),
                       cudaMemcpyDeviceToHost);

            cudaDeviceSynchronize();
            int overloadNodeBegin = -1;
            int overloadNodeNum = 0;
            for (int i = 0; i < activeNodesNum; i++) {
                if (activeNodeList[i] > maxPartitionNode) {
                    overloadNodeBegin = i;
                    overloadNodeNum = activeNodesNum - i;
                    break;
                }
            }
            if (overloadNodeBegin >= 0) {
                cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseSetAccessedBy,
                              cudaCpuDeviceId);
                int threadNum = 20;
                if (overloadNodeNum < 50) {
                    threadNum = 1;
                }
                thread runThreads[threadNum];
                for (int i = 0; i < threadNum; i++) {
                    runThreads[i] = thread(ccDynamic,
                                           i,
                                           threadNum,
                                           overloadNodeBegin,
                                           activeNodesNum,
                                           degree,
                                           activeOverloadNodePointers,
                                           nodePointersI,
                                           activeNodeList,
                                           edgeListOverload,
                                           edgeList);
                }
                for (unsigned int t = 0; t < threadNum; t++) {
                    runThreads[t].join();
                }
                cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseUnsetAccessedBy,
                              cudaCpuDeviceId);
            }
        }
        endReadCpu = std::chrono::steady_clock::now();
        durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
        startGpuProcessing = std::chrono::steady_clock::now();
        cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeListD, labelD);
        cc_kernelOpt<<<grid, block>>>(activeNodesNum, activeNodeListD, nodePointerD, degreeD, edgeListD, valueD,
                                      labelD, maxPartitionNode, edgeListOverload, activeOverloadNodePointersD);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseUnsetReadMostly, 0);
        endGpuProcessing = std::chrono::steady_clock::now();
        durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endGpuProcessing - startGpuProcessing).count();
        setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;

        cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
    }
    cudaDeviceSynchronize();
    cout << "nodeSum: " << nodeSum << endl;
    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
    cudaFree(nodePointerD);
    cudaFree(edgeListD);
    cudaFree(edgeListOverload);
    cudaFree(degreeD);
    cudaFree(labelD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegree);

    delete[] label;
    delete[] degree;
    delete[] value;
    delete[] activeNodeList;
    delete[] activeOverloadNodePointers;
    return durationRead;
}

void testSSSP() {
    // convert the webgraph output dataset to data of can be executed
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    ulong *nodePointersUL;
    uint *nodePointersI;
    EdgeWithWeight *edgeList;
    bool isUseShare = true;

    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(testWeightGraphPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;

    if (isUseShare) {
        gpuErrorcheck(cudaMallocManaged(&nodePointersI, (testNumNodes + 1) * sizeof(uint)));
        infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
        gpuErrorcheck(cudaMallocManaged(&edgeList, (testNumEdge) * sizeof(EdgeWithWeight)));
        infile.read((char *) edgeList, sizeof(EdgeWithWeight) * testNumEdge);
        infile.close();
    } else {
        nodePointersI = new uint[testNumNodes + 1];
        infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
        edgeList = new EdgeWithWeight[testNumEdge + 1];
        infile.read((char *) edgeList, sizeof(EdgeWithWeight) * testNumEdge);
        infile.close();
    }
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    int testTimes = 64;
    long timeSum = 0;

    for (int i = 0; i < testTimes; i++) {
        uint sourceNode = rand() % testNumNodes;
        cout << "sourceNode " << sourceNode << endl;
        if (isUseShare) {
            timeSum += ssspCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList, 26737294);
            break;
        } else {
            timeSum += ssspCaculateInOpt(testNumNodes, testNumEdge, nodePointersI, edgeList, 26737294);
            break;
        }
        cout << i << "========================================" << endl;
    }

    cout << "mean time is " << timeSum / testTimes << endl;
}

int needCpu = 0;
int notNeedCpu = 0;

long processingTimeSum = 0;
long cpuTimeSum = 0;
long allTimeSum = 0;

void testBFS() {
    uint testNumNodes = 0;
    ulong testNumEdge = 0;
    ulong *nodePointersUL;
    uint *nodePointersI;
    uint *edgeList;
    bool isUseShare = false;

    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(testGraphPath, ios::in | ios::binary);
    infile.read((char *) &testNumNodes, sizeof(uint));
    uint numEdge = 0;
    infile.read((char *) &numEdge, sizeof(uint));
    testNumEdge = numEdge;
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;

    if (isUseShare) {
        gpuErrorcheck(cudaMallocManaged(&nodePointersI, (testNumNodes + 1) * sizeof(uint)));
        infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
        gpuErrorcheck(cudaMallocManaged(&edgeList, (numEdge) * sizeof(uint)));
        cudaMemAdvise(nodePointersI, (testNumNodes + 1) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(edgeList, (numEdge) * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
        infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
        infile.close();
    } else {
        nodePointersI = new uint[testNumNodes + 1];
        infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
        edgeList = new uint[testNumEdge + 1];
        infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
        infile.close();
    }
    auto endReadGraph = std::chrono::steady_clock::now();
    long durationReadGraph = std::chrono::duration_cast<std::chrono::milliseconds>(
            endReadGraph - startReadGraph).count();
    cout << "read graph time : " << durationReadGraph << "ms" << endl;
    int testTimes = 64;
    long timeSum = 0;
    for (int i = 0; i < testTimes; i++) {
        uint sourceNode = rand() % testNumNodes;
        if (isUseShare) {
            //timeSum += bfsCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);
            timeSum += bfsCaculateInShare(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);
            break;
            //caculateInShareOpt(testNumNodes, testNumEdge, nodePointersI, edgeList);
        } else {
            //timeSum += bfsCaculateInOpt(testNumNodes, testNumEdge, nodePointersI, edgeList, sourceNode);

            //caculateInOptChooseByDegree(testNumNodes, testNumEdge, nodePointersI, edgeList);
            timeSum += bfsCaculateInAsyncSwap(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);
            timeSum += bfsCaculateInAsync(testNumNodes, testNumEdge, nodePointersI, edgeList, 25838548);

            break;
        }
        cout << i << "========================================" << endl;
    }
    cout << "need cpu " << needCpu << " not need cpu " << notNeedCpu << endl;
    cout << "processingTime " << processingTimeSum / 64 << " cpu time " << cpuTimeSum / 64 << " all Time "
         << allTimeSum / 64 << endl;
    cout << "mean time is " << timeSum / testTimes << endl;
}

long ssspCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList,
                         uint sourceNode) {
    cout << "==================ssspshare==============" << endl;
    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    uint sourceCode = 0;
    gpuErrorcheck(cudaMallocManaged(&degree, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMallocManaged(&value, testNumNodes * sizeof(uint)));

    auto startPreCaculate = std::chrono::steady_clock::now();
    for (uint i = 0; i < testNumNodes - 1; i++) {
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }

    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    sourceCode = sourceNode;
    bool *label;
    gpuErrorcheck(cudaMallocManaged(&label, testNumNodes * sizeof(bool)));
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX - 1;
    }

    label[sourceCode] = true;
    value[sourceCode] = 1;
    uint *activeNodeList;
    cudaMallocManaged(&activeNodeList, testNumNodes * sizeof(uint));
    //cacaulate the active node And make active node array
    uint *activeNodeLabelingD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    uint *activeNodeLabelingPrefixD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeList, label, activeNodeLabelingPrefixD);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeList, label);

        sssp_kernel<<<grid, block>>>(activeNodesNum, activeNodeList, nodePointersI, degree, edgeList, value, label);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
    }
    cudaDeviceSynchronize();

    cout << "nodeSum: " << nodeSum << endl;

    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - start).count();
    cout << "iter sum is " << iter << " finish time : " << durationRead << " ms" << endl;
    cudaFree(degree);
    cudaFree(label);
    cudaFree(value);
    cudaFree(activeNodeList);
    cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    return durationRead;
}


//std::thread
void ssspDynamic(int tId,
                 int numThreads,
                 unsigned int overloadNodeBegin,
                 unsigned int numActiveNodes,
                 unsigned int *outDegree,
                 unsigned int *activeNodesPointer,
                 unsigned int *nodePointer,
                 unsigned int *activeNodes,
                 EdgeWithWeight *edgeListOverload,
                 EdgeWithWeight *edgeList) {

    unsigned int chunkSize = ceil((numActiveNodes - overloadNodeBegin) / numThreads) + 1;
    unsigned int left, right;
    left = tId * chunkSize + overloadNodeBegin;
    right = min(left + chunkSize, numActiveNodes);
    unsigned int thisNode;
    unsigned int thisDegree;
    unsigned int fromHere;
    unsigned int fromThere;

    for (unsigned int i = left; i < right; i++) {
        thisNode = activeNodes[i];
        thisDegree = outDegree[thisNode];
        fromHere = activeNodesPointer[i];
        fromThere = nodePointer[thisNode];
        for (unsigned int j = 0; j < thisDegree; j++) {
            edgeListOverload[fromHere + j].toNode = edgeList[fromThere + j].toNode;
            edgeListOverload[fromHere + j].weight = edgeList[fromThere + j].weight;
            if (edgeList[fromThere + j].toNode == 0) {
                cout << thisNode << " " << nodePointer[thisNode] << endl;
            }
        }
    }

}

long
ssspCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, EdgeWithWeight *edgeList, uint sourceNode) {

    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    degree = new uint[testNumNodes];
    value = new uint[testNumNodes];
    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate max memory
    int deviceID;
    cudaDeviceProp dev;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);


    unsigned long max_partition_size = 0.5 * (dev.totalGlobalMem - 10 * 4 * testNumNodes) / sizeof(EdgeWithWeight);

    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    printf("size of edge is %d, max memory is %ld, most edge size is %ld\n multiprocessors %d", sizeof(EdgeWithWeight),
           dev.totalGlobalMem, max_partition_size, dev.multiProcessorCount);
    if (max_partition_size > DIST_INFINITY) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = DIST_INFINITY;
    }

    //caculate degree
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];


    int maxPartitionNode = testNumNodes - 1;
    bool hasMaxPartitionNode = false;
    if (max_partition_size < testNumEdge) {
        for (uint i = testNumNodes - 1; i >= 0; i--) {
            if (nodePointersI[i] < max_partition_size) {
                maxPartitionNode = i - 1;
                break;
            }
        }
    }

    cout << "maxPartitionNode: " << maxPartitionNode << endl;
    max_partition_size = nodePointersI[maxPartitionNode + 1];

    EdgeWithWeight *edgeListD;
    gpuErrorcheck(cudaMalloc(&edgeListD, max_partition_size * sizeof(EdgeWithWeight)));

    gpuErrorcheck(cudaMemcpy(edgeListD, edgeList, max_partition_size * sizeof(EdgeWithWeight), cudaMemcpyHostToDevice));

    bool *label;
    label = new bool[testNumNodes];
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX;
    }

    label[sourceNode] = true;
    value[sourceNode] = 1;

    uint *nodePointerD;
    uint *degreeD;
    bool *labelD;
    uint *valueD;
    uint *activeNodeList;
    uint *activeOverloadNodePointers;
    uint *activeNodeListD;
    uint *activeNodeLabelingD;
    uint *activeNodeLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegree;
    EdgeWithWeight *edgeListOverload;
    activeNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];

    gpuErrorcheck(cudaMallocManaged(&edgeListOverload, (testNumEdge - max_partition_size) * sizeof(EdgeWithWeight)));

    gpuErrorcheck(cudaMalloc(&nodePointerD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&labelD, testNumNodes * sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegree, testNumNodes * sizeof(uint)));

    gpuErrorcheck(cudaMemcpy(nodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(labelD, label, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegree);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(activeOverloadNodePointersD);

    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = 1;
    uint overloadEdgeSum = 0;
    auto startProcessing = std::chrono::steady_clock::now();
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;
    //cout << "source node " << sourceNode << "  degree " << degree[sourceNode] << endl;
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArrayAndNodePointer<<<grid, block>>>(testNumNodes, activeNodeListD, activeOverloadDegree,
                                                          labelD, activeNodeLabelingPrefixD, maxPartitionNode, degreeD);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        uint overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                              ptrOverloadDegree + activeNodesNum, 0);
        overloadEdgeSum += overloadEdgeNum;
        thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + activeNodesNum, activeOverloadNodePointersD);
        startCpu = std::chrono::steady_clock::now();
        if (overloadEdgeNum > 0) {
            cudaMemcpy(activeNodeList, activeNodeListD, activeNodesNum * sizeof(uint), cudaMemcpyDeviceToHost);
            cudaMemcpy(activeOverloadNodePointers, activeOverloadNodePointersD, activeNodesNum * sizeof(uint),
                       cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            int overloadNodeBegin = -1;
            int overloadNodeNum = 0;
            for (int i = 0; i < activeNodesNum; i++) {
                if (activeNodeList[i] > maxPartitionNode) {
                    overloadNodeBegin = i;
                    overloadNodeNum = activeNodesNum - i;
                    break;
                }
            }
            if (overloadNodeBegin >= 0) {
                int threadNum = 20;
                if (overloadNodeNum < 50) {
                    threadNum = 1;
                }
                thread runThreads[threadNum];

                for (int i = 0; i < threadNum; i++) {
                    runThreads[i] = thread(ssspDynamic,
                                           i,
                                           threadNum,
                                           overloadNodeBegin,
                                           activeNodesNum,
                                           degree,
                                           activeOverloadNodePointers,
                                           nodePointersI,
                                           activeNodeList,
                                           edgeListOverload,
                                           edgeList);
                }
                for (unsigned int t = 0; t < threadNum; t++) {
                    runThreads[t].join();
                }
            }
        }
        endReadCpu = std::chrono::steady_clock::now();
        durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeListD, labelD);
        sssp_kernelOpt<<<grid, block>>>(activeNodesNum, activeNodeListD, nodePointerD, degreeD, edgeListD, valueD,
                                        labelD, maxPartitionNode, edgeListOverload, activeOverloadNodePointersD);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;

        cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;

    }
    cudaDeviceSynchronize();
    cout << "nodeSum: " << nodeSum << endl;
    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
    cudaFree(nodePointerD);
    cudaFree(edgeListD);
    cudaFree(edgeListOverload);
    cudaFree(degreeD);
    cudaFree(labelD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegree);

    delete[] label;
    delete[] degree;
    delete[] value;
    delete[] activeNodeList;
    delete[] activeOverloadNodePointers;
    return durationRead;
}

long prCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList) {
    auto start = std::chrono::steady_clock::now();
    uint *degree;
    float *value;
    gpuErrorcheck(cudaMallocManaged(&degree, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMallocManaged(&value, testNumNodes * sizeof(float)));

    auto startPreCaculate = std::chrono::steady_clock::now();
    for (uint i = 0; i < testNumNodes - 1; i++) {
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }

    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    bool *label;
    gpuErrorcheck(cudaMallocManaged(&label, testNumNodes * sizeof(bool)));
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = true;
        value[i] = i;
    }
    uint *activeNodeList;
    cudaMallocManaged(&activeNodeList, testNumNodes * sizeof(uint));
    //cacaulate the active node And make active node array
    uint *activeNodeLabelingD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    uint *activeNodeLabelingPrefixD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;

    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeList, label, activeNodeLabelingPrefixD);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeList, label);
        //cc_kernel<<<grid, block>>>(activeNodesNum, activeNodeList, nodePointersI, degree, edgeList, value, label);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
    }
    cudaDeviceSynchronize();

    cout << "nodeSum: " << nodeSum << endl;

    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - start).count();
    cout << "iter sum is " << iter << " finish time : " << durationRead << " ms" << endl;
    cudaFree(degree);
    cudaFree(label);
    cudaFree(value);
    cudaFree(activeNodeList);
    cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    return durationRead;
}

long ccCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList) {
    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    uint *recordActiveNodes = new uint[testNumNodes];
    gpuErrorcheck(cudaMallocManaged(&degree, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMallocManaged(&value, testNumNodes * sizeof(uint)));

    auto startPreCaculate = std::chrono::steady_clock::now();
    for (uint i = 0; i < testNumNodes - 1; i++) {
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }

    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    bool *label;
    gpuErrorcheck(cudaMallocManaged(&label, testNumNodes * sizeof(bool)));
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = true;
        value[i] = i;
    }
    uint *activeNodeList;
    cudaMallocManaged(&activeNodeList, testNumNodes * sizeof(uint));
    //cacaulate the active node And make active node array
    uint *activeNodeLabelingD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    uint *activeNodeLabelingPrefixD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;

    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;
    auto startProcessing = std::chrono::steady_clock::now();
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeList, label, activeNodeLabelingPrefixD);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeList, label);
        for (uint i = 0; i < activeNodesNum; i++) {
            uint activeNode = activeNodeList[i];
            recordActiveNodes[activeNode]++;
        }
        cc_kernel<<<grid, block>>>(activeNodesNum, activeNodeList, nodePointersI, degree, edgeList, value, label);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
    }
    cudaDeviceSynchronize();

    cout << "nodeSum: " << nodeSum << endl;

    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "iter sum is " << iter << " finish time : " << durationRead << " ms" << endl;

    cudaFree(degree);
    cudaFree(label);
    cudaFree(value);
    cudaFree(activeNodeList);
    cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    return durationRead;
}

uint rangeMin = UINT_MAX - 1;
uint rangeMax = 0;
uint rangeSum = 0;

long bfsCaculateInShare(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode) {
    rangeMin = UINT_MAX - 1;
    rangeMax = 0;
    rangeSum = 0;
    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    uint sourceCode = 0;
    gpuErrorcheck(cudaMallocManaged(&degree, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMallocManaged(&value, testNumNodes * sizeof(uint)));

    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate degree
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    sourceCode = sourceNode;
    cout << "sourceNode " << sourceNode << " degree " << degree[sourceNode] << endl;
    bool *label;
    gpuErrorcheck(cudaMallocManaged(&label, testNumNodes * sizeof(bool)));
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX;
    }
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    label[sourceCode] = true;
    value[sourceCode] = 1;
    uint *activeNodeList;
    cudaMallocManaged(&activeNodeList, testNumNodes * sizeof(uint));
    //cacaulate the active node And make active node array
    uint *activeNodeLabelingD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    uint *activeNodeLabelingPrefixD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;
    auto startProcessing = std::chrono::steady_clock::now();
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeList, label, activeNodeLabelingPrefixD);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeList, label);
        bfs_kernel<<<grid, block>>>(activeNodesNum, activeNodeList, nodePointersI, degree, edgeList, value, label);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        //cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
    }
    cudaDeviceSynchronize();

    cout << "nodeSum: " << nodeSum << endl;

    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "iter sum is " << iter << " finish time : " << durationRead << " ms" << endl;
    //cout << "range min " << rangeMin << " range max " << rangeMax << " range sum " << rangeSum << endl;
    cout << "source node pointer  " << nodePointersI[sourceNode] << endl;
    cudaFree(degree);
    cudaFree(label);
    cudaFree(value);
    cudaFree(activeNodeList);
    cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    return durationRead;
}

//std::thread
void bfsDynamic(int tId,
                int numThreads,
                unsigned int overloadNodeBegin,
                unsigned int numActiveNodes,
                unsigned int *outDegree,
                unsigned int *activeNodesPointer,
                unsigned int *nodePointer,
                unsigned int *activeNodes,
                uint *edgeListOverload,
                uint *edgeList) {
    float waitToHandleNum = numActiveNodes - overloadNodeBegin;
    float numThreadsF = numThreads;
    unsigned int chunkSize = ceil(waitToHandleNum / numThreadsF);
    unsigned int left, right;
    left = tId * chunkSize + overloadNodeBegin;
    right = min(left + chunkSize, numActiveNodes);
    unsigned int thisNode;
    unsigned int thisDegree;
    unsigned int fromHere;
    unsigned int fromThere;

    for (unsigned int i = left; i < right; i++) {
        thisNode = activeNodes[i];
        thisDegree = outDegree[thisNode];
        fromHere = activeNodesPointer[i];
        fromThere = nodePointer[thisNode];
        for (unsigned int j = 0; j < thisDegree; j++) {
            edgeListOverload[fromHere + j] = edgeList[fromThere + j];
        }
    }
}

void setVisited(int tId,
                int numThreads,
                unsigned int numActiveNodes,
                unsigned int *activeNodes,
                bool *isVisited) {
    float waitToHandleNum = numActiveNodes;
    float numThreadsF = numThreads;
    unsigned int chunkSize = ceil(waitToHandleNum / numThreadsF);
    unsigned int left, right;
    left = tId * chunkSize;
    right = min(left + chunkSize, numActiveNodes);

    unsigned int thisNode;
    unsigned int fromHere;
    unsigned int fromThere;

    for (unsigned int i = left; i < right; i++) {
        isVisited[activeNodes[i]] = true;
    }

}

long bfsCaculateInOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode) {
    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    degree = new uint[testNumNodes];
    value = new uint[testNumNodes];
    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate max memory
    int deviceID;
    cudaDeviceProp dev;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);

    unsigned long max_partition_size = 0.8 * (dev.totalGlobalMem - 10 * 4 * testNumNodes) / sizeof(uint);

    uint *edgeListD;
    gpuErrorcheck(cudaMalloc(&edgeListD, max_partition_size * sizeof(uint)));

    gpuErrorcheck(cudaMemcpy(edgeListD, edgeList, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));

    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    printf("size of edge is %d, max memory is %ld, most edge size is %ld\n multiprocessors %d", sizeof(uint),
           dev.totalGlobalMem, max_partition_size, dev.multiProcessorCount);
    if (max_partition_size > DIST_INFINITY) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = DIST_INFINITY;
    }

    //caculate degree
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];

    int maxPartitionNode = testNumNodes - 1;
    bool hasMaxPartitionNode = false;
    if (max_partition_size < testNumEdge) {
        for (uint i = testNumNodes - 1; i >= 0; i--) {
            if (nodePointersI[i] < max_partition_size) {
                maxPartitionNode = i - 1;
                break;
            }
        }
    }
    max_partition_size = nodePointersI[maxPartitionNode + 1];
    cout << "max_partition_size: " << max_partition_size << "  testNumEdge: " << testNumEdge << endl;

    cout << "maxPartitionNode: " << maxPartitionNode << endl;

    bool *label;
    label = new bool[testNumNodes];
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX;
    }

    label[sourceNode] = true;
    value[sourceNode] = 1;

    uint *nodePointerD;
    uint *degreeD;
    bool *labelD;
    uint *valueD;
    uint *activeNodeList;
    uint *activeOverloadNodePointers;
    uint *activeNodeListD;
    uint *activeNodeLabelingD;
    uint *activeNodeLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegree;
    uint *edgeListOverload;
    activeNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];

    gpuErrorcheck(cudaMallocManaged(&edgeListOverload, (testNumEdge - max_partition_size) * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&nodePointerD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&labelD, testNumNodes * sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegree, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(nodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(labelD, label, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegree);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(activeOverloadNodePointersD);

    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = 1;
    uint overloadEdgeSum = 0;
    auto startProcessing = std::chrono::steady_clock::now();
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;

    cout << "source node " << sourceNode << "  degree " << degree[sourceNode] << endl;
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArrayAndNodePointer<<<grid, block>>>(testNumNodes, activeNodeListD, activeOverloadDegree,
                                                          labelD, activeNodeLabelingPrefixD, maxPartitionNode,
                                                          degreeD);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        uint overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                              ptrOverloadDegree + activeNodesNum, 0);
        overloadEdgeSum += overloadEdgeNum;
        thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + activeNodesNum,
                               activeOverloadNodePointersD);
        startCpu = std::chrono::steady_clock::now();
        if (overloadEdgeNum > 0) {
            needCpu++;
            cudaMemcpy(activeNodeList, activeNodeListD, activeNodesNum * sizeof(uint), cudaMemcpyDeviceToHost);

            cudaMemcpy(activeOverloadNodePointers, activeOverloadNodePointersD, activeNodesNum * sizeof(uint),
                       cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            int overloadNodeBegin = -1;
            int overloadNodeNum = 0;
            for (int i = 0; i < activeNodesNum; i++) {
                if (activeNodeList[i] > maxPartitionNode) {
                    overloadNodeBegin = i;
                    overloadNodeNum = activeNodesNum - i;
                    break;
                }
            }
            if (overloadNodeBegin >= 0) {
                cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseSetAccessedBy,
                              cudaCpuDeviceId);
                int threadNum = 20;
                if (overloadNodeNum < 50) {
                    threadNum = 1;
                }
                thread runThreads[threadNum];

                for (int i = 0; i < threadNum; i++) {
                    runThreads[i] = thread(bfsDynamic,
                                           i,
                                           threadNum,
                                           overloadNodeBegin,
                                           activeNodesNum,
                                           degree,
                                           activeOverloadNodePointers,
                                           nodePointersI,
                                           activeNodeList,
                                           edgeListOverload,
                                           edgeList);
                }
                for (unsigned int t = 0; t < threadNum; t++) {
                    runThreads[t].join();
                }
                cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseUnsetAccessedBy,
                              cudaCpuDeviceId);
            }

        } else {
            notNeedCpu++;
        }

        endReadCpu = std::chrono::steady_clock::now();
        durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
        startGpuProcessing = std::chrono::steady_clock::now();
        cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeListD, labelD);
        bfs_kernelOpt<<<grid, block>>>(activeNodesNum, activeNodeListD, nodePointerD, degreeD, edgeListD,
                                       valueD,
                                       labelD, maxPartitionNode, edgeListOverload, activeOverloadNodePointersD);
        cudaDeviceSynchronize();
        //gpuErrorcheck(cudaPeekAtLastError());
        cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseUnsetReadMostly, 0);
        auto endGpuProcessing = std::chrono::steady_clock::now();
        durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endGpuProcessing - startGpuProcessing).count();
        setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
    }
    cudaDeviceSynchronize();
    cout << "nodeSum: " << nodeSum << endl;
    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;

    cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
    cout << "need Cpu " << needCpu << " not need cpu " << notNeedCpu << endl;
    processingTimeSum += durationGpuProcessing;
    allTimeSum += durationRead;
    cpuTimeSum += durationReadCpu;
    cudaFree(nodePointerD);
    cudaFree(edgeListD);
    cudaFree(edgeListOverload);
    cudaFree(degreeD);
    cudaFree(labelD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegree);

    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    return durationRead;
}

long bfsCaculateInAsync(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode) {
    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    degree = new uint[testNumNodes];
    value = new uint[testNumNodes];
    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate max memory
    int deviceID;
    cudaDeviceProp dev;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);

    unsigned long max_partition_size = 0.6 * (dev.totalGlobalMem - 10 * 4 * testNumNodes) / sizeof(uint);

    printf("size of edge is %d, max memory is %ld, most edge size is %ld\n multiprocessors %d", sizeof(uint),
           dev.totalGlobalMem, max_partition_size, dev.multiProcessorCount);
    if (max_partition_size > DIST_INFINITY) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = DIST_INFINITY;
    }

    //caculate degree
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];

    int maxPartitionNode = testNumNodes - 1;
    bool hasMaxPartitionNode = false;
    if (max_partition_size < testNumEdge) {
        for (uint i = testNumNodes - 1; i >= 0; i--) {
            if (nodePointersI[i] < max_partition_size) {
                maxPartitionNode = i - 1;
                break;
            }
        }
    }
    max_partition_size = nodePointersI[maxPartitionNode + 1];
    cout << "max_partition_size: " << max_partition_size << "  testNumEdge: " << testNumEdge << endl;
    cout << "maxPartitionNode: " << maxPartitionNode << endl;
    uint *edgeListD;
    gpuErrorcheck(cudaMalloc(&edgeListD, max_partition_size * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(edgeListD, edgeList, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));

    uint *label;
    label = new uint[testNumNodes];
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = 0;
        value[i] = UINT_MAX;
    }

    label[sourceNode] = 1;
    value[sourceNode] = 1;

    uint *nodePointerD;
    uint *degreeD;
    uint *isActiveD;
    uint *valueD;
    uint *activeNodeList;
    uint *activeOverloadNodePointers;
    uint *activeNodeListD;
    //uint *activeNodeLabelingD;
    uint *activeNodeLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegree;
    uint *edgeListOverload;
    activeNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];

    gpuErrorcheck(cudaMallocManaged(&edgeListOverload, (testNumEdge - max_partition_size) * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&nodePointerD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isActiveD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(uint)));
    //gpuErrorcheck(cudaMalloc(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegree, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(isActiveD, label, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(nodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegree);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(activeOverloadNodePointersD);

    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;
    uint overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto startProcessing = std::chrono::steady_clock::now();

    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    cout << "source node " << sourceNode << "  degree " << degree[sourceNode] << endl;
    cudaStream_t steamStatic, streamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);
    while (activeNodesNum > 0) {
        startPreGpuProcessing = std::chrono::steady_clock::now();
        iter++;
        cout << "1iter " << iter << " activeNodesNum " << activeNodesNum << endl;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArrayAndNodePointerOpt<<<grid, block>>>(testNumNodes, activeNodeListD, activeOverloadDegree,
                                                             isActiveD, activeNodeLabelingPrefixD, maxPartitionNode,
                                                             degreeD);
        uint overloadNodeNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);

        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
        uint overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                              ptrOverloadDegree + activeNodesNum, 0);
        overloadEdgeSum += overloadEdgeNum;
        thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + activeNodesNum, activeOverloadNodePointersD);

        startGpuProcessing = std::chrono::steady_clock::now();
        setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(activeNodesNum, activeNodeListD, isActiveD);
        bfs_kernelStatic<<<grid, block, 0, steamStatic>>>(activeNodesNum - overloadNodeNum, activeNodeListD,
                                                          nodePointerD, degreeD,
                                                          edgeListD, valueD, isActiveD);
        if (overloadEdgeNum > 0) {
            startCpu = std::chrono::steady_clock::now();
            needCpu++;
            cudaMemcpyAsync(activeNodeList, activeNodeListD, activeNodesNum * sizeof(uint), cudaMemcpyDeviceToHost,
                            streamDynamic);
            cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, activeNodesNum * sizeof(uint),
                            cudaMemcpyDeviceToHost, streamDynamic);
            int overloadNodeBegin = -1;
            int overloadNodeNum = 0;
            for (int i = 0; i < activeNodesNum; i++) {
                if (activeNodeList[i] > maxPartitionNode) {
                    overloadNodeBegin = i;
                    overloadNodeNum = activeNodesNum - i;
                    break;
                }
            }

            if (overloadNodeBegin >= 0) {
                cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseSetAccessedBy,
                              cudaCpuDeviceId);
                int threadNum = 10;
                if (overloadNodeNum < 50) {
                    threadNum = 1;
                }
                thread runThreads[threadNum];

                for (int i = 0; i < threadNum; i++) {
                    runThreads[i] = thread(bfsDynamic,
                                           i,
                                           threadNum,
                                           overloadNodeBegin,
                                           activeNodesNum,
                                           degree,
                                           activeOverloadNodePointers,
                                           nodePointersI,
                                           activeNodeList,
                                           edgeListOverload,
                                           edgeList);
                }

                if (iter == 6) {
                    cout << "2iter " << iter << " activeNodesNum " << activeNodesNum << endl;
                }
                for (unsigned int t = 0; t < threadNum; t++) {
                    runThreads[t].join();
                }


                cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseUnsetAccessedBy,
                              cudaCpuDeviceId);
            }
            endReadCpu = std::chrono::steady_clock::now();
            durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();

            //cudaDeviceSynchronize();
            cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
            bfs_kernelDynamic<<<grid, block>>>(activeNodesNum, activeNodeListD, degreeD, valueD,
                                               isActiveD, overloadNodeNum, edgeListOverload,
                                               activeOverloadNodePointersD);
            cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseUnsetReadMostly, 0);

        } else {
            notNeedCpu++;
        }

        cudaDeviceSynchronize();
        //gpuErrorcheck(cudaPeekAtLastError());
        auto endGpuProcessing = std::chrono::steady_clock::now();
        durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endGpuProcessing - startGpuProcessing).count();
        //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
        startPreGpuProcessing = std::chrono::steady_clock::now();
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
    }
    cudaDeviceSynchronize();
    cout << "nodeSum: " << nodeSum << endl;
    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;

    cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
    cout << "needCpu " << needCpu << " not need cpu " << notNeedCpu << endl;
    processingTimeSum += durationGpuProcessing;
    allTimeSum += durationRead;
    cpuTimeSum += durationReadCpu;
    cudaFree(nodePointerD);
    cudaFree(edgeListD);
    cudaFree(edgeListOverload);
    cudaFree(degreeD);
    cudaFree(isActiveD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    //cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegree);

    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    return durationRead;
}


long bfsCaculateInAsyncSwap(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList, uint sourceNode) {
    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    degree = new uint[testNumNodes];
    value = new uint[testNumNodes];
    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate max memory
    int deviceID;
    cudaDeviceProp dev;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);
    unsigned long max_partition_size = 0.6 * (dev.totalGlobalMem - 10 * 4 * testNumNodes) / sizeof(uint);
    printf("size of edge is %d, max memory is %ld, most edge size is %ld\n multiprocessors %d", sizeof(uint),
           dev.totalGlobalMem, max_partition_size, dev.multiProcessorCount);
    if (max_partition_size > DIST_INFINITY) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = DIST_INFINITY;
    }
    uint temp = max_partition_size % fragment_size;
    max_partition_size = max_partition_size - temp;
    //caculate degree
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];

    uint *edgeListD;
    gpuErrorcheck(cudaMalloc(&edgeListD, max_partition_size * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(edgeListD, edgeList, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));
    /*
     * add swap
     * */
    bool *isInStatic = new bool[testNumNodes];
    bool *isInStaticD;
    bool *isInStaticManaged;
    gpuErrorcheck(cudaMalloc(&isInStaticD, testNumNodes * sizeof(bool)))
    gpuErrorcheck(cudaMallocManaged(&isInStaticManaged, testNumNodes * sizeof(bool)))
    uint *overloadNodeListD;
    gpuErrorcheck(cudaMalloc(&overloadNodeListD, testNumNodes * sizeof(uint)));
    uint *overloadNodeList = new uint[testNumNodes];
    bool *isVisited = new bool[testNumNodes];
    FragmentData *fragmentData;
    uint *staticNodePointer = new uint[testNumNodes];
    uint *staticNodePointerD;
    uint *staticNodePointerManaged;
    memcpy(staticNodePointer, nodePointersI, testNumNodes * sizeof(uint));
    gpuErrorcheck(cudaMalloc(&staticNodePointerD, testNumNodes * sizeof(uint)))
    gpuErrorcheck(cudaMallocManaged(&staticNodePointerManaged, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(staticNodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    uint *nodePointerD;
    gpuErrorcheck(cudaMalloc(&nodePointerD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(nodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    uint *label;
    uint maxStaticNode = 0;
    label = new uint[testNumNodes];
    uint fragmentNum = testNumEdge / fragment_size;
    gpuErrorcheck(cudaMallocManaged(&fragmentData, fragmentNum * sizeof(FragmentData)));
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = 0;
        value[i] = UINT_MAX;

        uint pointStartFragmentIndex = nodePointersI[i] / fragment_size;
        uint pointEndFragmentIndex =
                degree[i] == 0 ? pointStartFragmentIndex : (nodePointersI[i] + degree[i] - 1) / fragment_size;
        if (pointStartFragmentIndex == pointEndFragmentIndex) {
            if (fragmentData[pointStartFragmentIndex].vertexNum == 0) {
                fragmentData[pointStartFragmentIndex].startVertex = i;
            } else if (fragmentData[pointStartFragmentIndex].startVertex > i) {
                fragmentData[pointStartFragmentIndex].startVertex = i;
            }
            fragmentData[pointStartFragmentIndex].vertexNum++;
        }

        if (nodePointersI[i] < max_partition_size && (nodePointersI[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > maxStaticNode) maxStaticNode = i;
        } else {
            isInStatic[i] = false;
        }
    }
    cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
    cout << "max_partition_size: " << max_partition_size << "  maxStaticNode: " << maxStaticNode << endl;

    uint staticFragmentNum = max_partition_size / fragment_size;
    cout << "fragmentNum " << fragmentNum << " staticFragmentNum " << staticFragmentNum << endl;
    for (int i = 0; i < fragmentNum; i++) {
        fragmentData[i].isIn = false;
        fragmentData[i].isVisit = false;
    }
    for (int i = 0; i < staticFragmentNum; i++) {
        fragmentData[i].isIn = true;
    }

    uint *staticFragmentToNormalMap = new uint[staticFragmentNum];
    for (uint i = 0; i < staticFragmentNum; i++) {
        staticFragmentToNormalMap[i] = i;
    }
    uint *staticFragmentData = new uint[staticFragmentNum];
    uint *staticFragmentDataD;
    gpuErrorcheck(cudaMalloc(&staticFragmentDataD, staticFragmentNum * sizeof(uint)));
    uint *canSwapStaticFragmentDataD;
    gpuErrorcheck(cudaMalloc(&canSwapStaticFragmentDataD, staticFragmentNum * sizeof(uint)));
    uint *canSwapFragmentPrefixSumD;
    gpuErrorcheck(cudaMalloc(&canSwapFragmentPrefixSumD, staticFragmentNum * sizeof(uint)));
    thrust::device_ptr<unsigned int> ptr_canSwapFragment(canSwapStaticFragmentDataD);
    thrust::device_ptr<unsigned int> ptr_canSwapFragmentPrefixSum(canSwapFragmentPrefixSumD);

    label[sourceNode] = 1;
    value[sourceNode] = 1;

    uint *degreeD;
    uint *isActiveD;
    uint *valueD;
    uint *activeNodeList;
    uint *activeOverloadNodePointers;
    uint *activeNodeListD;
    uint *activeNodeLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegree;
    uint *edgeListOverload;
    activeNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];

    gpuErrorcheck(cudaMallocManaged(&edgeListOverload, (testNumEdge - max_partition_size) * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&isActiveD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegree, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(isActiveD, label, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    //setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(isActiveD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegree);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(activeOverloadNodePointersD);

    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;
    uint overloadEdgeSum = 0;
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;

    auto startSwap = std::chrono::steady_clock::now();
    auto endSwap = std::chrono::steady_clock::now();
    long durationSwap = 0;

    auto startGpuProcessing = std::chrono::steady_clock::now();
    auto endGpuProcessing = std::chrono::steady_clock::now();
    long durationGpuProcessing = 0;

    auto startPreGpuProcessing = std::chrono::steady_clock::now();
    auto endPreGpuProcessing = std::chrono::steady_clock::now();
    long durationPreGpuProcessing = 0;
    auto startProcessing = std::chrono::steady_clock::now();

    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    cout << "source node " << sourceNode << "  degree " << degree[sourceNode] << endl;
    cudaStream_t steamStatic, streamDynamic;
    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);
    while (activeNodesNum > 0) {

        startPreGpuProcessing = std::chrono::steady_clock::now();
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArrayAndNodePointerSwap<<<grid, block>>>(testNumNodes, activeNodeListD,
                                                              isActiveD, activeNodeLabelingPrefixD, isInStaticD);
        cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;


        setFragmentData<<<grid, block>>>(activeNodesNum, activeNodeListD, staticNodePointerD,
                                         canSwapStaticFragmentDataD, staticFragmentNum, fragment_size,
                                         isInStaticD);
        uint canSwapFragmentNum = thrust::reduce(ptr_canSwapFragment, ptr_canSwapFragment + staticFragmentNum);
        if (canSwapFragmentNum > 0) {
            cout << "iter : " << iter << " canSwapFragmentNum " << canSwapFragmentNum << endl;
            thrust::exclusive_scan(ptr_canSwapFragment, ptr_canSwapFragment + staticFragmentNum,
                                   ptr_canSwapFragmentPrefixSum);
            setStaticFragmentData<<<grid, block>>>(staticFragmentNum, canSwapStaticFragmentDataD,
                                                   canSwapFragmentPrefixSumD, staticFragmentDataD);
            cudaMemcpy(staticFragmentData, staticFragmentDataD, canSwapFragmentNum * sizeof(uint),
                       cudaMemcpyDeviceToHost);
        }

        uint overloadNodeNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        uint overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                              ptrOverloadDegree + activeNodesNum, 0);
        overloadEdgeSum += overloadEdgeNum;

        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setOverloadNodePointerSwap<<<grid, block>>>(testNumNodes, overloadNodeListD, activeOverloadDegree, isActiveD,
                                                    activeNodeLabelingPrefixD, degreeD);

        thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + overloadNodeNum, activeOverloadNodePointersD);

        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
        startGpuProcessing = std::chrono::steady_clock::now();
        setLabelDefaultOpt<<<grid, block, 0, steamStatic>>>(activeNodesNum, activeNodeListD, isActiveD);
        bfs_kernelStaticSwap<<<grid, block, 0, steamStatic>>>(activeNodesNum, activeNodeListD,
                                                              staticNodePointerD, degreeD,
                                                              edgeListD, valueD, isActiveD, isInStaticD);
        if (overloadNodeNum > 0) {
            startCpu = std::chrono::steady_clock::now();
            needCpu++;
            cudaMemcpyAsync(activeNodeList, activeNodeListD, activeNodesNum * sizeof(uint), cudaMemcpyDeviceToHost,
                            streamDynamic);
            cudaMemcpyAsync(overloadNodeList, overloadNodeListD, overloadNodeNum * sizeof(uint), cudaMemcpyDeviceToHost,
                            streamDynamic);
            cudaMemcpyAsync(activeOverloadNodePointers, activeOverloadNodePointersD, overloadNodeNum * sizeof(uint),
                            cudaMemcpyDeviceToHost, streamDynamic);
            cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseSetAccessedBy,
                          cudaCpuDeviceId);
            int threadNum = 20;
            if (overloadNodeNum < 50) {
                threadNum = 1;
            }
            thread runThreads[threadNum];

            for (int i = 0; i < threadNum; i++) {
                runThreads[i] = thread(bfsDynamic,
                                       i,
                                       threadNum,
                                       0,
                                       overloadNodeNum,
                                       degree,
                                       activeOverloadNodePointers,
                                       nodePointersI,
                                       overloadNodeList,
                                       edgeListOverload,
                                       edgeList);
            }


            for (unsigned int t = 0; t < threadNum; t++) {
                runThreads[t].join();
            }

            cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseUnsetAccessedBy,
                          cudaCpuDeviceId);
            endReadCpu = std::chrono::steady_clock::now();
            durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
            //cudaDeviceSynchronize();
            cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseSetReadMostly, 0);
            bfs_kernelDynamicSwap<<<grid, block, 0, streamDynamic>>>(overloadNodeNum, overloadNodeListD, degreeD,
                                                                     valueD,
                                                                     isActiveD, edgeListOverload,
                                                                     activeOverloadNodePointersD);

        } else {
            notNeedCpu++;
        }

        /*startSwap = std::chrono::steady_clock::now();
        uint canSwapStaticFragmentIndex = 0;
        bool needSwap = false;
        uint swapSum = 0;
        for (uint i = fragmentNum - 1; i > 0; i--) {
            if (cudaSuccess == cudaStreamQuery(streamDynamic) || canSwapStaticFragmentIndex >= canSwapFragmentNum) {
                //if (needSwap) cudaMemcpy(isInStaticD, isInStatic, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice);
                endSwap = std::chrono::steady_clock::now();
                durationSwap += std::chrono::duration_cast<std::chrono::milliseconds>(
                        endSwap - startSwap).count();
                //cout << iter << "  transfer==== " << swapSum << endl;

                break;
            }
            if (cudaErrorNotReady == cudaStreamQuery(streamDynamic)) {
                FragmentData swapFragmentData = fragmentData[i];
                if (!swapFragmentData.isVisit && !swapFragmentData.isIn) {
                    uint swapStaticFragmentIndex = staticFragmentData[canSwapStaticFragmentIndex++];
                    uint beSwappedFragmentIndex = staticFragmentToNormalMap[swapStaticFragmentIndex];
                    FragmentData beSwappedFragment = fragmentData[beSwappedFragmentIndex];
                    if (beSwappedFragmentIndex > 0 && beSwappedFragmentIndex < fragmentNum) {
                        for (uint j = beSwappedFragment.startVertex - 1;
                             j < beSwappedFragment.startVertex + beSwappedFragment.vertexNum + 1 &&
                             j < testNumNodes; j++) {
                            isInStaticManaged[j] = false;
                            if (j == 68312114) {
                                cout << "false iter " << iter << "  swap  " << 68312114 << " " << isInStaticManaged[j]
                                     << endl;
                            }
                        }
                        for (uint j = swapFragmentData.startVertex;
                             j < swapFragmentData.startVertex + swapFragmentData.vertexNum; j++) {
                            isInStaticManaged[j] = true;
                            if (j == 68312114) {
                                cout << "true iter " << iter << "  swap  " << 68312114 << " " << isInStaticManaged[j]
                                     << endl;
                            }
                            staticNodePointerD[j] =
                                    nodePointersI[j] - i * fragment_size + swapStaticFragmentIndex * fragment_size;
                        }

                    }
                    staticFragmentToNormalMap[swapStaticFragmentIndex] = i;
                    fragmentData[i].isIn = true;
                    swapSum++;
                }
            }
        }
        */
        cudaDeviceSynchronize();
        cudaMemAdvise(edgeListOverload, overloadEdgeNum * sizeof(uint), cudaMemAdviseUnsetReadMostly, 0);
        auto endGpuProcessing = std::chrono::steady_clock::now();
        durationGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endGpuProcessing - startGpuProcessing).count();
        startPreGpuProcessing = std::chrono::steady_clock::now();
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        endPreGpuProcessing = std::chrono::steady_clock::now();
        durationPreGpuProcessing += std::chrono::duration_cast<std::chrono::milliseconds>(
                endPreGpuProcessing - startPreGpuProcessing).count();
    }
    cudaDeviceSynchronize();
    cout << "nodeSum: " << nodeSum << endl;
    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startProcessing).count();
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "fact processing time : " << durationGpuProcessing << " ms" << endl;

    cout << "gpu pre processing time : " << durationPreGpuProcessing << " ms" << endl;
    cout << "swap processing time : " << durationSwap << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " " << endl;
    cout << "needCpu " << needCpu << " not need cpu " << notNeedCpu << endl;
    processingTimeSum += durationGpuProcessing;
    allTimeSum += durationRead;
    cpuTimeSum += durationReadCpu;

    cudaFree(nodePointerD);
    cudaFree(edgeListD);
    cudaFree(edgeListOverload);
    cudaFree(degreeD);
    cudaFree(isActiveD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    //cudaFree(activeNodeLabelingD);
    cudaFree(activeNodeLabelingPrefixD);
    cudaFree(activeOverloadNodePointersD);
    cudaFree(activeOverloadDegree);
    cudaFree(isInStaticManaged);
    cudaFree(isInStaticD);
    cudaFree(staticNodePointerD);
    cudaFree(overloadNodeListD);
    cudaFree(fragmentData);

    delete[]            label;
    delete[]            degree;
    delete[]            value;
    delete[]            activeNodeList;
    delete[]            activeOverloadNodePointers;
    delete[] isInStatic;
    delete[] overloadNodeList;
    delete[] isVisited;
    delete[] staticNodePointer;
    return durationRead;
}

void caculateInShareOpt(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList) {

    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    uint sourceCode = 0;
    gpuErrorcheck(cudaMallocManaged(&degree, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMallocManaged(&value, testNumNodes * sizeof(uint)));

    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate max memory
    int deviceID;
    cudaDeviceProp dev;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);

    unsigned long max_partition_size = 0.9 * (dev.totalGlobalMem - 8 * 4 * testNumNodes) / sizeof(uint);

    printf("size of edge is %d, max memory is %ld, most edge size is %ld\n multiprocessors %d", sizeof(uint),
           dev.totalGlobalMem, max_partition_size, dev.multiProcessorCount);
    if (max_partition_size > DIST_INFINITY) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = DIST_INFINITY;
    }

    int maxPartitionNode = testNumNodes - 1;
    bool hasMaxPartitionNode = false;
    cout << "max_partition_size: " << max_partition_size << "  testNumEdge: " << testNumEdge << endl;
    if (max_partition_size < testNumEdge) {
        for (uint i = testNumNodes - 1; i >= 0; i--) {
            if (nodePointersI[i] < max_partition_size) {
                maxPartitionNode = i - 1;
                break;
            }
        }
        cout << "dayu" << endl;
    } else {
        cout << "xiaoyu" << endl;
    }

    cout << "maxPartitionNode: " << maxPartitionNode << endl;

    for (uint i = 0; i < testNumNodes - 1; i++) {
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
        if (sourceCode == 0 && degree[i] > 1000) {
            sourceCode = i;
            cout << "sourceCode " << sourceCode << " degree " << degree[i] << endl;
        }
    }

    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];

    uint *edgeListD;
    gpuErrorcheck(cudaMalloc(&edgeListD, max_partition_size * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(edgeListD, edgeList, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));

    bool *label;
    gpuErrorcheck(cudaMallocManaged(&label, testNumNodes * sizeof(bool)));
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX;
    }
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;

    label[sourceCode] = true;
    value[sourceCode] = 1;
    uint *activeNodeList;
    cudaMallocManaged(&activeNodeList, testNumNodes * sizeof(uint));
    //cacaulate the active node And make active node array
    uint *activeNodeLabelingD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    uint *activeNodeLabelingPrefixD;
    gpuErrorcheck(cudaMallocManaged(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = activeNodesNum;
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeList, label, activeNodeLabelingPrefixD);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeList, label);

        /*uint edgeNum = 0;
        for (int i = 0; i < activeNodesNum; i++) {
            edgeNum += degree[activeNodeList[i]];
        }
        float percent = (float) edgeNum / (float) testNumEdge;
        if (percent > 0.01) {
            cout << "iter: " << iter << " edgeNum: " << edgeNum << " percent: " << percent << endl;
        }*/

        bfs_kernelShareOpt<<<grid, block>>>(activeNodesNum, activeNodeList, nodePointersI, degree, edgeListD,
                                            edgeList,
                                            value, label, maxPartitionNode);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        setLabeling<<<grid, block>>>(testNumNodes, label, activeNodeLabelingD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        nodeSum += activeNodesNum;
        //cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
    }
    cudaDeviceSynchronize();

    cout << "nodeSum: " << nodeSum << endl;

    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - start).count();
    cout << "finish time : " << durationRead << " ms" << endl;
    cudaFree(nodePointersI);
    cudaFree(edgeList);
    cudaFree(degree);
    cudaFree(label);
    cudaFree(value);
    cudaFree(activeNodeList);
}

struct TempSortIndexDegree {
    uint index;
    int degree;
};

bool compare(TempSortIndexDegree a, TempSortIndexDegree b) {
    return a.degree > b.degree;
}

void
caculateInOptChooseByDegree(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList,
                            uint sourceNode) {

    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    degree = new uint[testNumNodes];
    value = new uint[testNumNodes];
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    auto startPreCaculate = std::chrono::steady_clock::now();
    //caculate max memory
    int deviceID;
    cudaDeviceProp dev;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);

    unsigned long max_partition_size = 0.9 * (dev.totalGlobalMem - 10 * 4 * testNumNodes) / sizeof(uint);

    printf("size of edge is %d, max memory is %ld, most edge size is %ld\n multiprocessors %d", sizeof(uint),
           dev.totalGlobalMem, max_partition_size, dev.multiProcessorCount);
    if (max_partition_size > DIST_INFINITY) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = DIST_INFINITY;
    }

    //sort by degree
    TempSortIndexDegree *sortedData = new TempSortIndexDegree[testNumNodes];
    //caculate degree
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
        sortedData[i].degree = degree[i];
        sortedData[i].index = i;
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    sortedData[testNumNodes - 1].degree = degree[testNumNodes - 1];
    sortedData[testNumNodes - 1].index = testNumNodes - 1;

    uint *edgeListGpu = new uint[max_partition_size];
    uint *ptrOfEdgeListGpu = new uint[testNumNodes];
    bool *isInList = new bool[testNumNodes];
    int edgeSum = 0;
    sort(sortedData, sortedData + testNumNodes, compare);
    int edgeListGpuTempIndex = 0;
    for (int i = 0; i < testNumNodes; i++) {
        edgeSum += sortedData[i].degree;
        if (edgeSum > max_partition_size) {
            break;
        }
        isInList[sortedData[i].index] = true;
        ptrOfEdgeListGpu[sortedData[i].index] = edgeListGpuTempIndex;
        for (int j = 0; j < sortedData[i].degree; j++) {
            edgeListGpu[edgeListGpuTempIndex + j] = edgeList[nodePointersI[sortedData[i].index] + j];
        }
        edgeListGpuTempIndex += sortedData[i].degree;
        //cout << "isInList[ " << sortedData[i].index << "] is true"<< " degree is " << sortedData[i].degree << endl;
    }
    cout << "edgeSum: " << edgeSum << endl;
    bool *isInListD;
    gpuErrorcheck(cudaMalloc(&isInListD, testNumNodes * sizeof(bool)));
    gpuErrorcheck(cudaMemcpy(isInListD, isInList, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice));

    bool *label;
    label = new bool[testNumNodes];
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX;
    }
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;

    label[sourceNode] = true;
    value[sourceNode] = 1;
    uint *edgeListD;
    gpuErrorcheck(cudaMalloc(&edgeListD, max_partition_size * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(edgeListD, edgeListGpu, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));
    uint *nodePointerD;
    uint *degreeD;
    bool *labelD;
    uint *valueD;

    uint *activeNodeList;
    uint *activeOverloadNodePointers;
    uint *activeNodeListD;
    uint *activeNodeLabelingD;
    uint *activeNodeLabelingPrefixD;
    uint *activeOverloadNodePointersD;
    uint *activeOverloadDegree;
    uint *edgeListOverload;
    activeNodeList = new uint[testNumNodes];
    activeOverloadNodePointers = new uint[testNumNodes];
    gpuErrorcheck(cudaMallocManaged(&edgeListOverload, (testNumEdge - max_partition_size) * sizeof(uint)));


    gpuErrorcheck(cudaMalloc(&nodePointerD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&labelD, testNumNodes * sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadNodePointersD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&activeOverloadDegree, testNumNodes * sizeof(uint)));

    gpuErrorcheck(cudaMemcpy(nodePointerD, ptrOfEdgeListGpu, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(labelD, label, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice));
    auto endPreCaculate = std::chrono::steady_clock::now();
    long durationPreCaculate = std::chrono::duration_cast<std::chrono::milliseconds>(
            endPreCaculate - startPreCaculate).count();
    cout << "durationPreCaculate time : " << durationPreCaculate << " ms" << endl;

    //cacaulate the active node And make active node array
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    thrust::device_ptr<unsigned int> ptrOverloadDegree(activeOverloadDegree);
    thrust::device_ptr<unsigned int> ptrOverloadPrefixsum(activeOverloadNodePointersD);
    auto startCpu = std::chrono::steady_clock::now();
    auto endReadCpu = std::chrono::steady_clock::now();
    long durationReadCpu = 0;
    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    uint nodeSum = 1;
    uint overloadEdgeSum = 0;
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArrayAndNodePointerBySortOpt<<<grid, block>>>(testNumNodes, activeNodeListD,
                                                                   activeOverloadDegree,
                                                                   labelD, activeNodeLabelingPrefixD, isInListD,
                                                                   degreeD);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        // cout << "=========================================================================================" << endl;
        uint overloadEdgeNum = thrust::reduce(thrust::device, ptrOverloadDegree,
                                              ptrOverloadDegree + activeNodesNum, 0);
        overloadEdgeSum += overloadEdgeNum;
        thrust::exclusive_scan(ptrOverloadDegree, ptrOverloadDegree + activeNodesNum, activeOverloadNodePointersD);
        cout << "iter: " << iter << " overloadEdgeNum: " << overloadEdgeNum << endl;
        startCpu = std::chrono::steady_clock::now();
        if (overloadEdgeNum > 0) {
            cudaMemcpy(activeNodeList, activeNodeListD, activeNodesNum * sizeof(uint), cudaMemcpyDeviceToHost);
            /*for (int i = 0; i < activeNodesNum; i++) {
                cout << "activeNodeList[" << i << "] " << activeNodeList[i] << endl;
            }*/
            cudaMemcpy(activeOverloadNodePointers, activeOverloadNodePointersD, activeNodesNum * sizeof(uint),
                       cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            int testNum = 0;
            for (int i = 0; i < activeNodesNum; i++) {
                if (!isInList[activeNodeList[i]]) {
                    uint dest = activeOverloadNodePointers[i];
                    uint src = nodePointersI[activeNodeList[i]];
                    for (int j = 0; j < degree[activeNodeList[i]]; j++) {
                        edgeListOverload[dest + j] = edgeList[src + j];
                    }

                }
            }
        }
        endReadCpu = std::chrono::steady_clock::now();
        durationReadCpu += std::chrono::duration_cast<std::chrono::milliseconds>(endReadCpu - startCpu).count();
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeListD, labelD);
        bfs_kernelOptOfSorted<<<grid, block>>>(activeNodesNum, activeNodeListD, nodePointerD, degreeD, edgeListD,
                                               edgeListOverload, valueD,
                                               labelD, isInListD, activeOverloadNodePointersD);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
        //thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
        //thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
        nodeSum += activeNodesNum;
    }
    cudaDeviceSynchronize();
    cout << "nodeSum: " << nodeSum << endl;
    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - start).count();
    cout << "finish time : " << durationRead << " ms" << endl;
    cout << "cpu time : " << durationReadCpu << " ms" << endl;
    cout << "overloadEdgeSum : " << overloadEdgeSum << " ms" << endl;

    cudaFree(nodePointerD);
    cudaFree(edgeListD);
    cudaFree(degreeD);
    cudaFree(labelD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
    cudaFree(isInListD);
}

void caculateInCommon(uint testNumNodes, uint testNumEdge, uint *nodePointersI, uint *edgeList) {

    auto start = std::chrono::steady_clock::now();
    uint *degree;
    uint *value;
    degree = new uint[testNumNodes];
    value = new uint[testNumNodes];
    for (uint i = 0; i < testNumNodes - 1; i++) {
        if (nodePointersI[i] > testNumEdge) {
            cout << i << "   " << nodePointersI[i] << endl;
            break;
        }
        degree[i] = nodePointersI[i + 1] - nodePointersI[i];
    }
    degree[testNumNodes - 1] = testNumEdge - nodePointersI[testNumNodes - 1];
    bool *label;
    label = new bool[testNumNodes];
    for (uint i = 0; i < testNumNodes; i++) {
        label[i] = false;
        value[i] = UINT_MAX;
    }
    //cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    uint sourceCode = 0;
    label[sourceCode] = true;
    value[sourceCode] = 1;
    uint *nodePointerD;
    uint *edgeListD;
    uint *degreeD;
    bool *labelD;
    uint *valueD;
    uint *activeNodeListD;
    gpuErrorcheck(cudaMalloc(&nodePointerD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&edgeListD, testNumEdge * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&degreeD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&labelD, testNumNodes * sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&activeNodeListD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&valueD, testNumNodes * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(nodePointerD, nodePointersI, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(edgeListD, edgeList, testNumEdge * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(degreeD, degree, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(valueD, value, testNumNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(labelD, label, testNumNodes * sizeof(bool), cudaMemcpyHostToDevice));

    //cacaulate the active node And make active node array
    uint *activeNodeLabelingD;
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingD, testNumNodes * sizeof(unsigned int)));
    uint *activeNodeLabelingPrefixD;
    gpuErrorcheck(cudaMalloc(&activeNodeLabelingPrefixD, testNumNodes * sizeof(unsigned int)));
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);

    setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
    thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
    uint activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
    int iter = 0;
    while (activeNodesNum > 0) {
        iter++;
        thrust::exclusive_scan(ptr_labeling, ptr_labeling + testNumNodes, ptr_labeling_prefixsum);
        setActiveNodeArray<<<grid, block>>>(testNumNodes, activeNodeListD, labelD, activeNodeLabelingPrefixD);
        setLabelDefault<<<grid, block>>>(activeNodesNum, activeNodeListD, labelD);
        bfs_kernel<<<grid, block>>>(activeNodesNum, activeNodeListD, nodePointerD, degreeD, edgeListD, valueD,
                                    labelD);
        cudaDeviceSynchronize();
        setLabeling<<<grid, block>>>(testNumNodes, labelD, activeNodeLabelingD);
        //thrust::device_ptr<unsigned int> ptr_labeling(activeNodeLabelingD);
        //thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(activeNodeLabelingPrefixD);
        activeNodesNum = thrust::reduce(ptr_labeling, ptr_labeling + testNumNodes);
        cout << "iter: " << iter << " activeNodes: " << activeNodesNum << endl;
    }
    cudaDeviceSynchronize();

    auto endRead = std::chrono::steady_clock::now();
    long durationRead = std::chrono::duration_cast<std::chrono::milliseconds>(endRead - start).count();
    cout << "finish time : " << durationRead << " ms" << endl;
    cudaFree(nodePointerD);
    cudaFree(edgeListD);
    cudaFree(degreeD);
    cudaFree(labelD);
    cudaFree(valueD);
    cudaFree(activeNodeListD);
}

void
readGraphFromJava(string filePath, uint &testNumNodes, ulong &testNumEdge, ulong *nodePointersUL,
                  uint *nodePointersI,
                  uint *edgeList, bool isNeedConvert) {
    char *testChars = (char *) malloc(8);
    ifstream testInfile(filePath, ios::in | ios::binary);
    testInfile.read(testChars, 4);
    testNumNodes = uint((unsigned char) (testChars[0]) << 24 |
                        (unsigned char) (testChars[1]) << 16 |
                        (unsigned char) (testChars[2]) << 8 |
                        (unsigned char) (testChars[3]));
    testInfile.read(testChars, sizeof(ulong));
    testNumEdge = ulong((unsigned long) (unsigned char) (testChars[0]) << 56 |
                        (unsigned long) (unsigned char) (testChars[1]) << 48 |
                        (unsigned long) (unsigned char) (testChars[2]) << 40 |
                        (unsigned long) (unsigned char) (testChars[3]) << 32 |
                        (unsigned long) (unsigned char) (testChars[4]) << 24 |
                        (unsigned long) (unsigned char) (testChars[5]) << 16 |
                        (unsigned long) (unsigned char) (testChars[6]) << 8 |
                        (unsigned long) (unsigned char) (testChars[7]));
    cout << "testNumNodes " << testNumNodes << " testNumEdge " << testNumEdge << endl;

    if (testNumEdge > UINT_MAX) {
        cout << "isUseLongType" << endl;
        nodePointersUL = (ulong *) malloc(testNumNodes * sizeof(unsigned long));
        for (int i = 0; i < testNumNodes; i++) {
            testInfile.read(testChars, sizeof(ulong));
            ulong index = ulong((unsigned long) (unsigned char) (testChars[0]) << 56 |
                                (unsigned long) (unsigned char) (testChars[1]) << 48 |
                                (unsigned long) (unsigned char) (testChars[2]) << 40 |
                                (unsigned long) (unsigned char) (testChars[3]) << 32 |
                                (unsigned long) (unsigned char) (testChars[4]) << 24 |
                                (unsigned long) (unsigned char) (testChars[5]) << 16 |
                                (unsigned long) (unsigned char) (testChars[6]) << 8 |
                                (unsigned char) (unsigned char) (testChars[7]));

            if (index > testNumEdge) {
                for (int i = 0; i < 8; i++) {
                    printf("%d ", (unsigned char) testChars[i]);
                }
                cout << endl;
                break;
            }

            nodePointersUL[i] = index;
            cout << "nodepointer " << i << "  is " << nodePointersUL[i] << endl;
        }
    } else {
        cout << "not isUseLongType" << endl;
        nodePointersI = (uint *) malloc(testNumNodes * sizeof(unsigned int));
        for (int i = 0; i < testNumNodes; i++) {
            testInfile.read(testChars, sizeof(ulong));
            ulong index = ulong((unsigned long) (unsigned char) (testChars[0]) << 56 |
                                (unsigned long) (unsigned char) (testChars[1]) << 48 |
                                (unsigned long) (unsigned char) (testChars[2]) << 40 |
                                (unsigned long) (unsigned char) (testChars[3]) << 32 |
                                (unsigned long) (unsigned char) (testChars[4]) << 24 |
                                (unsigned long) (unsigned char) (testChars[5]) << 16 |
                                (unsigned long) (unsigned char) (testChars[6]) << 8 |
                                (unsigned char) (unsigned char) (testChars[7]));

            if (index > testNumEdge) {
                for (int i = 0; i < 8; i++) {
                    printf("%d ", (unsigned char) testChars[i]);
                }
                cout << endl;
                break;
            }

            nodePointersI[i] = index;
        }
    }

    edgeList = new uint[testNumEdge];
    for (int i = 0; i < testNumEdge; i++) {
        testInfile.read(testChars, 4);
        edgeList[i] = uint((unsigned char) (testChars[0]) << 24 |
                           (unsigned char) (testChars[1]) << 16 |
                           (unsigned char) (testChars[2]) << 8 |
                           (unsigned char) (testChars[3]));
        if (edgeList[i] > testNumNodes) {
            cout << edgeList[i] << endl;
            for (int i = 0; i < 8; i++) {
                printf("%d ", (unsigned char) testChars[i]);
            }
            cout << endl;
            break;
        }
    }

    if (isNeedConvert) {
        convertBncr(testNumNodes, testNumEdge, nodePointersI, edgeList);
    }

}

void convertBwcsr() {
    string inputPath = "/home/gxl/labproject/subway/sk-2005.bcsr";
    auto startReadGraph = std::chrono::steady_clock::now();
    ifstream infile(inputPath, ios::in | ios::binary);
    uint testNumNodes = 0;
    uint testNumEdge = 0;
    infile.read((char *) &testNumNodes, sizeof(uint));
    infile.read((char *) &testNumEdge, sizeof(uint));
    cout << "vertex num: " << testNumNodes << " edge num: " << testNumEdge << endl;
    uint *nodePointersI = new uint[testNumNodes];
    infile.read((char *) nodePointersI, sizeof(uint) * testNumNodes);
    uint *edgeList = new uint[testNumEdge];
    infile.read((char *) edgeList, sizeof(uint) * testNumEdge);
    infile.close();
    std::ofstream outfile("/home/gxl/labproject/subway/sk-2005.bwcsr", std::ofstream::binary);

    outfile.write((char *) &testNumNodes, sizeof(unsigned int));
    outfile.write((char *) &testNumEdge, sizeof(unsigned int));
    outfile.write((char *) nodePointersI, sizeof(unsigned int) * testNumNodes);
    for (int i = 0; i < testNumEdge; i++) {
        EdgeWithWeight edgeWithWeight;
        edgeWithWeight.weight = rand() % 5;
        edgeWithWeight.toNode = edgeList[i];
        outfile.write((char *) &edgeWithWeight, sizeof(EdgeWithWeight));
    }
    outfile.close();
    delete[] edgeList;
}

void convertBncr(uint vertexNum, ulong edgeNum, uint *nodePointer, uint *edgeList) {
    std::ofstream outfile(converPath, std::ofstream::binary);

    outfile.write((char *) &vertexNum, sizeof(unsigned int));
    uint edgeNumInt = edgeNum;
    outfile.write((char *) &edgeNumInt, sizeof(unsigned int));
    outfile.write((char *) nodePointer, sizeof(unsigned int) * vertexNum);
    outfile.write((char *) edgeList, sizeof(uint) * edgeNumInt);

    outfile.close();
}
