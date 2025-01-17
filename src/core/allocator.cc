#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        for (auto it = free_blocks.begin(); it != free_blocks.end(); it++)
        {
            size_t startAddr = it->first;
            size_t blockSize = it->second;
            if (startAddr >= size)
            {
                free_blocks.erase(it);
                // 检查剩余部分是否还有内存
                if (blockSize > size)
                {
                    // 将剩余的内存添加回空闲块列表
                    size_t remainingAddr = startAddr + size; // 剩余块的起始地址
                    size_t remainingSize = blockSize - size; // 剩余块的大小
                    free_blocks[remainingAddr] = remainingSize;
                }
                return startAddr;
            }
        }
        // cout << "here!!!" << endl;
        // this->ptr = this->runtime->alloc(size);
        // cout << "ptr::::" << ptr << endl;
        this->used += size;
        if (this->used > this->peak)
        {
            this->peak = size;
        }
        return this->used - size;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        // 插入新的空闲块
        free_blocks[addr] = size;

        // 尝试合并相邻的空闲块
        auto current = free_blocks.find(addr);
        auto prev = current == free_blocks.begin() ? free_blocks.end() : std::prev(current);
        auto next = std::next(current);

        // 检查是否与后面相邻
        if (next != free_blocks.end() && (addr + size == next->first))
        {
            current->second += next->second; // 扩展当前块的大小
            free_blocks.erase(next);         // 删除后面的块
        }

        // 检查是否与前面相邻
        if (prev != free_blocks.end() && (prev->first + prev->second == addr))
        {
            prev->second += current->second; // 扩展前一个块的大小
            free_blocks.erase(current);      // 删除当前块
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
