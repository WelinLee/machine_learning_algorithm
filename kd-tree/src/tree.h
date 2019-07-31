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
	// is always at the top of the heap.   �ѣ����Ѻ���С�ѣ�
	struct neighbor_heap_cmp {
		bool operator()(const std::tuple<size_t, float> &i,
			const std::tuple<size_t, float> &j) {
			return std::get<1>(i) < std::get<1>(j);
		}
	};

	typedef std::tuple<size_t, float> neighbor;
	typedef std::priority_queue<neighbor,
		std::vector<neighbor>, neighbor_heap_cmp> neighbor_heap;

	// ���� K-����ʱ�Ķѣ��󶥶ѣ����Ѷ�ʼ���� K-��������������Զ�ĵ�
	neighbor_heap k_neighbor_heap_;
	// �����ʱ�� p, dist(x, y) = pow((x^p + y^p), 1/p)
	float p;
	// ����ʱ�Ƿ��ͷ������ڴ�
	bool free_tree_;
	// �������
	tree_node *root;
	// ѵ����
	const float *datas;
	// ѵ������������
	size_t n_samples;
	// ÿ��������ά��
	size_t n_features;
	// ѵ�����ı�ǩ
	const float *labels;
	// Ѱ����λ��ʱ�õ��Ļ����
	std::tuple<size_t, float> *get_mid_buf_;
	// ���� K ����ʱ�Ļ���أ�����Ѿ��������� i���� visited_buf[i] = True
	bool *visited_buf_;

#ifdef USE_INTEL_MKL
	// ʹ�� Intel MKL ��ʱ�Ļ���
	float *mkl_buf_;
#endif


	// ��ʼ������
	void InitBuffer();

	// ����
	tree_node *BuildTree(const std::vector<size_t> &points);

	// ��һ��������λ��
	std::tuple<size_t, float> MidElement(const std::vector<size_t> &points, size_t dim);

	// ���
	void HeapStackPush(std::stack<tree_node *> &paths, tree_node *node, const float *coor, size_t k);

	// ��ȡѵ�����е� sample ��������� dim ��ֵ
	float GetDimVal(size_t sample, size_t dim) {
		return datas[sample * n_features + dim];
	}

	// ��� coor ����ѵ������ i ����ľ���
	float GetDist(size_t i, const float *coor);

	// Ѱ���зֵ�
	size_t FindSplitDim(const std::vector<size_t> &points);

};
