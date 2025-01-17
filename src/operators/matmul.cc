#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        Tensor A = inputs[0], B = inputs[1];

        // 获取输入张量的形状
        auto shapeA = A->getDims();
        auto shapeB = B->getDims();

        // 转置最后两个维度
        if (transA)
        {
            std::swap(shapeA[shapeA.size() - 1], shapeA[shapeA.size() - 2]);
        }
        if (transB)
        {
            std::swap(shapeB[shapeB.size() - 1], shapeB[shapeB.size() - 2]);
        }

        // 确保形状有效
        IT_ASSERT(shapeA.size() >= 2 && shapeB.size() >= 2, "MatMul requires tensors with at least 2 dimensions");
        size_t aLastDim = shapeA[shapeA.size() - 1];
        size_t bSecondLastDim = shapeB[shapeB.size() - 2];
        IT_ASSERT(aLastDim == bSecondLastDim, "Inner dimensions must match for MatMul");

        // 确定广播形状
        Shape batchShape;
        auto aBatchDims = shapeA.begin(), bBatchDims = shapeB.begin();
        while (aBatchDims != shapeA.end() - 2 && bBatchDims != shapeB.end() - 2)
        {
            if (*aBatchDims == *bBatchDims)
            {
                batchShape.push_back(*aBatchDims);
            }
            else if (*aBatchDims == 1)
            {
                batchShape.push_back(*bBatchDims);
            }
            else if (*bBatchDims == 1)
            {
                batchShape.push_back(*aBatchDims);
            }
            else
            {
                IT_ASSERT(false, "Incompatible batch dimensions for MatMul");
            }
            ++aBatchDims;
            ++bBatchDims;
        }

        // Append the final M and N dimensions
        batchShape.insert(batchShape.end(), {shapeA[shapeA.size() - 2], shapeB[shapeB.size() - 1]});

        return {{batchShape}};
    }

} // namespace infini