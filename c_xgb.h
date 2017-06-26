#include <iostream>

class Node
{
public:
	int split_feature;
	double split_value;
	double leaf_value;
	int is_leaf;
	int child_left, child_right;

	Node();
	void set_node(int is_leaf = -1, float leaf_value = -1, int split_feature = -1, float split_value = -1, int child_left = -1, int child_right = -1);
	void print_node(int id);
};