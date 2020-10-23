
struct EdgeWithWeight {
    uint toNode;
    uint weight;
};
struct FragmentData {
    uint startVertex = UINT_MAX - 1;
    uint vertexNum = 0;
    bool isIn = false;
    bool isVisit = false;
};