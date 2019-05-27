 //144. Binary Tree Preorder Traversal
 //https://leetcode.com/problems/binary-tree-preorder-traversal/
 /*
Tree traversal problem... Both recursive solution and iterative solution should be totally understood.
 */
//Recursive solution
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
private:
    void dfs(vector<int>& res, TreeNode* root){
        if(root == nullptr) return;
        res.push_back(root->val);
        dfs(res, root->left);
        dfs(res, root->right);
    }
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        dfs(res, root);
        return res;
    }
};

//Iterative solution
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        st.push(root);
        while(!st.empty()){
            TreeNode* node = st.top();
            st.pop();
            if(node != nullptr){
                res.push_back(node->val);
                st.push(node->right);
                st.push(node->left);
            }
        }
        return res;
    }
};

//94. Binary Tree Inorder Traversal
//https://leetcode.com/problems/binary-tree-inorder-traversal/
//Recursive Version
class Solution {
private:
    void dfs(vector<int>& res, TreeNode* root){
        if(root == nullptr) return;
        dfs(res, root->left);
        res.push_back(root->val);
        dfs(res, root->right);
    }
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        dfs(res, root);
        return res;
    }
};

//Iterative Version
//This implementation is elegant, however, too many small tricks there.
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        if(root == nullptr) return res;
        //st.push(root);
        //Maintain a variable for current value. For inorder traversal, we need to push left nodes to our stack, then add the value to res.
        TreeNode* cur = root;
        while(cur || !st.empty()){
            if(cur!= nullptr){
                st.push(cur);
                cur = cur->left;
            }
            else{
                cur = st.top();//We need to update cur. because currentlt it's nullptr
                res.push_back(cur->val);
                //We need to pop the node here because we already added this value to the final array
                st.pop();
                //We do not need to push cur to stack now, we will push it in if statement
                cur = cur->right;
            }
        }
        return res;
    }
};
//Slight modification
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        if(root == nullptr) return res;
        //Push root to stack to prevent stack from being empty earlier.
        st.push(root);
        //Maintain a variable for current value. For inorder traversal, we need to push left nodes to our stack, then add the value to res.
        TreeNode* cur = root;
        do {
            if(cur!= nullptr){
                st.push(cur);
                cur = cur->left;
            }
            else{
                cur = st.top();//We need to update cur. because currentlt it's nullptr
                res.push_back(cur->val);
                //We need to pop the node here because we already added this value to the final array
                st.pop();
                //We do not need to push cur to stack now, we will push it in if statement
                cur = cur->right;
            }
        } while(!st.empty());
        //Pop_back the element we choose to be the placeholder.
        res.pop_back();

        return res;
    }
};



