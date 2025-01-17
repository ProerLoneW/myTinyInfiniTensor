#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================

    // 1. 找到两个形状中较长的长度
    size_t maxRank = std::max(A.size(), B.size());

    // 2. 补全两个形状，使得维度数量相等
    Shape paddedA(maxRank, 1);
    Shape paddedB(maxRank, 1);
    for (size_t i = 0; i < A.size(); ++i)
    {
        paddedA[maxRank - A.size() + i] = A[i];
    }
    for (size_t i = 0; i < B.size(); ++i)
    {
        paddedB[maxRank - B.size() + i] = B[i];
    }

    // 3. 按规则从后向前逐维比较，确定结果形状
    Shape broadcastedShape(maxRank);
    for (size_t i = 0; i < maxRank; ++i)
    {
        if (paddedA[i] == 1)
        {
            broadcastedShape[i] = paddedB[i];
        }
        else if (paddedB[i] == 1)
        {
            broadcastedShape[i] = paddedA[i];
        }
        else
        {
            // 如果两个维度都不为 1 且不相等，则广播无效
            IT_ASSERT(paddedA[i] == paddedB[i], "Shapes are not broadcastable");
            broadcastedShape[i] = paddedA[i];
        }
    }

    return broadcastedShape;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
