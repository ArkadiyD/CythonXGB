#include <cmath>
#include <cstdio>
#include "json.hpp"
#include <cstring>
#include <climits>
using namespace std;
using json::JSON;

class Node
{
    public:
        int split_feature;
        float split_value;
        float leaf_value;
        int is_leaf;
        int child_left, child_right;

    Node(){};

    void set_node(int is_leaf_ = -1, float leaf_value_ = -1, int split_feature_ = -1, float split_value_ = -1, int child_left_ = -1, int child_right_ = -1)
    {
        is_leaf = is_leaf_;
        leaf_value = leaf_value_;
        split_feature = split_feature_;
        split_value = split_value_;
        child_left = child_left_;
        child_right = child_right_;
    }

    void print_node(int id)
    {
        printf("%d %d %f %d %f %d %d\n", id, is_leaf, leaf_value, split_feature, split_value, child_left, child_right);
    }
};

class Tree
{
    public:
        int depth;
        Node *nodes;
    
    Tree(){};

    void init(int depth_ = 0)
    {
        depth = depth_;
        nodes = new Node[1 << (depth + 1)];
    }

    void print_tree_util(int root, int level)
    {
        for (int i = 0; i < level; ++i)
            printf("\t");

        nodes[root].print_node(root);

        int child_left = nodes[root].child_left;
        int child_right = nodes[root].child_right;

        if (child_left >= 0)
            print_tree_util(child_left, level + 1);
        
        if (child_right >= 0)
            print_tree_util(child_right, level + 1);
    }

    void print_tree()
    {
        print_tree_util(0, 0);
    }

    float predict_util(float *features, int root)
    {
        Node node = nodes[root];
        int child;

        //printf ("leaf: %f", node.leaf_value);
        if (node.is_leaf)
            return node.leaf_value;

        if (features[node.split_feature] < node.split_value)
            child = node.child_left;
        else
            child = node.child_right;
        //printf("%d %d %f %d\n", node.is_leaf, node.split_feature, node.split_value, features[node.split_feature] < node.split_value);

        return predict_util(features, child);
    }

    float predict(float *features)
    {
        return predict_util(features, 0);
    }
};


class CXgboost
{   
    public:
        int n_trees;
        Tree *trees;
        float base_score;
        int objective;

    CXgboost(){};
    CXgboost(int depth, int n_features, int n_trees_ , int objective_, float base_score_)
    {
        base_score = base_score_;
        objective = objective_;
        if (objective == 1)
            base_score = -log(1.0 / base_score - 1.0);
        n_trees = n_trees_;
        
        char string[1000000];

        trees = new Tree[n_trees];
        for(int i = 0; i < n_trees; ++i)
            trees[i].init(depth);
        
        char filename[1000];
        for(int i = 0; i < n_trees; ++i)
        {
            sprintf(filename, "trees/tree_%d.json", i);
            FILE *f = fopen(filename, "r");
            size_t fsize = fread(string, sizeof(char), 1000000, f);
            string[fsize] = 0;
            fclose(f);

            json::JSON obj = JSON::Load(string);
            save_tree(i, obj);
        }
    }

    void save_tree(int tree_num, json::JSON &tree_data)
    {
        int node_id, split_feature, child_left_id, child_right_id;
        float split_condition, leaf;
        json::JSON child_left, child_right;

        node_id = tree_data["nodeid"].ToInt();
            
        if (tree_data.hasKey("split_condition"))
        {
            bool has_floating_point = tree_data["split_condition"].ToFloat(has_floating_point);
            if (!has_floating_point)
                split_condition = (float)tree_data["split_condition"].ToInt();
            else
                split_condition = tree_data["split_condition"].ToFloat();
        }

        if (tree_data.hasKey("split"))
        {
            string split_feature_str = tree_data["split"].ToString();
            split_feature_str = split_feature_str.substr(1);
            split_feature = stoi(split_feature_str);
        }

        if (tree_data.hasKey("leaf"))
        {
            bool has_floating_point = tree_data["leaf"].ToFloat(has_floating_point);
            if (!has_floating_point)
                leaf = (float)tree_data["leaf"].ToInt();
            else
                leaf = tree_data["leaf"].ToFloat();
        }

        if (tree_data.hasKey("children"))
        {
            child_left = tree_data["children"][0];
            child_right = tree_data["children"][1];
        }   
        

        if (!tree_data.hasKey("leaf"))
        {
            child_left_id = child_left["nodeid"].ToInt();
            child_right_id = child_right["nodeid"].ToInt();
                      
            trees[tree_num].nodes[node_id].set_node(0, -1, split_feature, split_condition, child_left_id, child_right_id);
            
            save_tree(tree_num, child_left);
            save_tree(tree_num, child_right);
        }
        else
            trees[tree_num].nodes[node_id].set_node(1, leaf);
    }
    

    float logistic(float x)
    {
        return 1 / (1 + exp(-x));
    }

    float predict_tree(int tree_num, float *features)
    {
        float tree_prediction = trees[tree_num].predict(features);
        //printf ("%d %f\n", tree_num, tree_prediction);
        return tree_prediction;
    }

    float predict(float *features, int ntree_limit)
    {
        float total_prediction = base_score;

        for (int i = 0; i < ntree_limit; i++)
        {   total_prediction += predict_tree(i, features);
            //printf("%d -- %.3f, ", i+1, total_prediction);
        }
        if (objective == 1)
            total_prediction = logistic(total_prediction);
        
        
        return total_prediction;
    }

};
