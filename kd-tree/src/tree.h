#include <algorithm>
#include <vector>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <stack>
#include <queue>
#include "kdtree.h"


class KDTree
{

public:
	KDTree(){}

	KDTree(tree_node *root, const float *datas, size_t rows, size_t cols, float p);

	KDTree(const float *datas, const float *labels,
		size_t rows, size_t cols, float p, bool free_tree = true);

	~KDTree();

	tree_node *GetRoot() { return root; }

	std::vector<std::tuple<size_t, float>> FindKNearests(const float *coor, size_t k);

	std::tuple<size_t, float> FindNearest(const float *coor, size_t k) { return FindKNearests(coor, k)[0]; }

	void CFindKNearests(const float *coor, size_t k, size_t *args, float *dists);


private:
	// The sample with the largest distance from point `coor`
	// is always at the top of the heap.   堆（最大堆和最小堆）
	struct neighbor_heap_cmp {
		bool operator()(const std::tuple<size_t, float> &i,
			const std::tuple<size_t, float> &j) {
			return std::get<1>(i) < std::get<1>(j);
		}
	};

	typedef std::tuple<size_t, float> neighbor;
	typedef std::priority_queue<neighbor,
		std::vector<neighbor>, neighbor_heap_cmp> neighbor_heap;

	// 搜索 K-近邻时的堆（大顶堆），堆顶始终是 K-近邻中样本点最远的点
	neighbor_heap k_neighbor_heap_;
	// 求距离时的 p, dist(x, y) = pow((x^p + y^p), 1/p)
	float p;
	// 析构时是否释放树的内存
	bool free_tree_;
	// 树根结点
	tree_node *root;
	// 训练集
	const float *datas;
	// 训练集的样本数
	size_t n_samples;
	// 每个样本的维度
	size_t n_features;
	// 训练集的标签
	const float *labels;
	// 寻找中位数时用到的缓存池
	std::tuple<size_t, float> *get_mid_buf_;
	// 搜索 K 近邻时的缓存池，如果已经搜索过点 i，令 visited_buf[i] = True
	bool *visited_buf_;

#ifdef USE_INTEL_MKL
	// 使用 Intel MKL 库时的缓存
	float *mkl_buf_;
#endif


	// 初始化缓存
	void InitBuffer();

	// 建树
	tree_node *BuildTree(const std::vector<size_t> &points);

	// 求一组数的中位数
	std::tuple<size_t, float> MidElement(const std::vector<size_t> &points, size_t dim);

	// 入堆
	void HeapStackPush(std::stack<tree_node *> &paths, tree_node *node, const float *coor, size_t k);

	// 获取训练集中第 sample 个样本点第 dim 的值
	float GetDimVal(size_t sample, size_t dim) {
		return datas[sample * n_features + dim];
	}

	// 求点 coor 距离训练集第 i 个点的距离
	float GetDist(size_t i, const float *coor);

	// 寻找切分点
	size_t FindSplitDim(const std::vector<size_t> &points);

};
