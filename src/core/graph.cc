#include "core/graph.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    bool InvTranspose(const TransposeObj &a, const TransposeObj &b)
    {
        auto permute_a = a.getPermute();
        auto permute_b = b.getPermute();
        for (size_t i = 0; i < permute_a.size(); i++)
            if (permute_a[permute_b[i]] != i)
                return false;
        return true;
    }
    bool isTransForMul(const TransposeObj &self)
    {
        const auto &permute = self.getPermute();
        size_t rank = permute.size();

        // 检查rank是否足够大，否则无法交换最后两维
        if (rank < 2)
            return false;

        // 确保除了最后两维，其他维度没有变化
        for (size_t i = 0; i < rank - 2; ++i)
            if (permute[i] != i)
                return false; // 如果中间维度发生了变动，就不符合条件

        // 确认最后两维是互换的
        if (permute[rank - 1] == (int)rank - 2 && permute[rank - 2] == (int)rank - 1)
            return true;
        return false;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        IT_ASSERT(topo_sort());

        std::vector<Operator> remove_ops;
        std::vector<Tensor> remove_tensors;
        std::vector<Tensor> wait_for_cut;

        // Step 1: Remove redundant Transpose operators
        for (const auto &op : ops)
        {
            if (op->getOpType() == OpType::Transpose && op->getPredecessors().size() == 1)
            {
                auto upstream_op = op->getPredecessors()[0];
                if (upstream_op->getOpType() == OpType::Transpose)
                {
                    if (InvTranspose(*(dynamic_cast<TransposeObj *>(op.get())),
                                     *(dynamic_cast<TransposeObj *>(upstream_op.get()))))
                    {
                        auto upstream_tensor = upstream_op->getInputs()[0];
                        // Remove connections from upstream_op and op
                        removeConnections(op, upstream_op);
                        // Reconnect downstream operators directly to upstream_op’s input tensor
                        reconnectDownstream(op, upstream_tensor);
                        // Mark current op and its outputs for removal
                        markForRemoval(op, remove_ops, remove_tensors, wait_for_cut);
                    }
                }
            }
        }
        // Step 2: Merge Transpose into Matmul
        for (const auto &op : ops)
        {
            if (op->getOpType() == OpType::MatMul)
            {
                for (size_t i = 0; i < 2; ++i)
                {
                    auto input = op->getInputs()[i];
                    if (auto upstream_op = input->getSource())
                    {
                        if (upstream_op->getOpType() == OpType::Transpose)
                        {
                            if (bool is_transpose = isTransForMul(*(dynamic_cast<TransposeObj *>(upstream_op.get()))))
                            {
                                // Adjust MatMul attributes based on Transpose
                                auto matmul_op = dynamic_cast<MatmulObj *>(op.get());
                                if (i == 0)
                                {
                                    matmul_op->setTransA(matmul_op->getTransA() != is_transpose);
                                }
                                else
                                {
                                    matmul_op->setTransB(matmul_op->getTransB() != is_transpose);
                                }

                                // Remove connections and replace inputs
                                removeConnections(op, upstream_op);
                                reconnectDownstream(op, upstream_op->getInputs()[0]);
                            }
                        }
                    }
                }
            }
        }
        // Step 3: Clean up isolated tensors
        cleanIsolatedTensors(wait_for_cut, remove_ops, remove_tensors);
        // Step 4: Remove marked operators and tensors from graph
        finalizeRemoval(remove_ops, remove_tensors);
    }

    void GraphObj::removeConnections(const Operator &op, const Operator &upstream_op)
    {
        upstream_op->removeSuccessors(op);
        op->removePredecessors(upstream_op);
        if (upstream_op->getPredecessors().size() == 1)
        {
            auto upstream_parent_op = upstream_op->getPredecessors()[0];
            upstream_parent_op->removeSuccessors(upstream_op);
            upstream_op->removePredecessors(upstream_parent_op);
            upstream_op->addSuccessors(op);
            op->addPredecessors(upstream_op);
        }
    }

    void GraphObj::reconnectDownstream(const Operator &op, const Tensor &upstream_tensor)
    {
        for (const auto &output : op->getOutputs())
        {
            for (const auto &target : output->getTargets())
            {
                upstream_tensor->addTarget(target);
                target->replaceInput(output, upstream_tensor);
            }
        }
    }

    void GraphObj::markForRemoval(const Operator &op,
                                  std::vector<Operator> &remove_ops,
                                  std::vector<Tensor> &remove_tensors,
                                  std::vector<Tensor> &wait_for_cut)
    {
        remove_ops.push_back(op);
        for (const auto &output : op->getOutputs())
        {
            remove_tensors.push_back(output);
            for (const auto &target : output->getTargets())
            {
                target->replaceInput(output, nullptr);
            }
        }
    }

    void GraphObj::cleanIsolatedTensors(std::vector<Tensor> &wait_for_cut,
                                        std::vector<Operator> &remove_ops,
                                        std::vector<Tensor> &remove_tensors)
    {
        while (!wait_for_cut.empty())
        {
            auto tensor = wait_for_cut.back();
            wait_for_cut.pop_back();
            if (auto op = tensor->getSource())
            {
                removeConnections(op, nullptr);
                markForRemoval(op, remove_ops, remove_tensors, wait_for_cut);
            }
        }
    }

    void GraphObj::finalizeRemoval(const std::vector<Operator> &remove_ops,
                                   const std::vector<Tensor> &remove_tensors)
    {
        for (const auto &op : remove_ops)
        {
            ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
        }
        for (const auto &tensor : remove_tensors)
        {
            tensors.erase(std::remove(tensors.begin(), tensors.end(), tensor), tensors.end());
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        size_t allocSize = 0;

        for (auto &tensor : tensors)
            allocSize += tensor->size() * tensor->getDType().getSize();
        size_t offset = allocator.alloc(allocSize);

        // 遍历所有张量，为每个张量绑定
        for (auto &tensor : tensors)
        {
            // std::cout << allocator.getPtr() << ' ';
            auto tensorPtr = static_cast<char *>(allocator.getPtr()) + offset; // 计算张量的内存地址
            tensor->setDataBlob(make_ref<BlobObj>(runtime, tensorPtr));
            offset += tensor->size() * tensor->getDType().getSize();
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini