/**
 * @file Array.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2023-02-04
 * 
 * A quick second pass with the common 'tree' algorithm problems.
 * 
 */
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
 /*
    144. Binary Tree Preorder Traversal
    https://leetcode.com/problems/binary-tree-preorder-traversal/
    Given the root of a binary tree, return the preorder traversal of its nodes' values.

    Example 1:
    Input: root = [1,null,2,3]
    Output: [1,2,3]

    Example 2:
    Input: root = []
    Output: []

    Example 3:
    Input: root = [1]
    Output: [1]
    

    Constraints:
    The number of nodes in the tree is in the range [0, 100].
    -100 <= Node.val <= 100
    
    Follow up: Recursive solution is trivial, could you do it iteratively?
 */
// Recursive
class Solution {
private:
    void traverse(TreeNode* node, vector<int>& v) {
        if (node == nullptr) return;
        v.push_back(node->val);
        traverse(node->left, v);
        traverse(node->right, v);
    }
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        traverse(root, res);
        return res;
    }
};

// Iterative
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> res;
        st.push(root);
        while (!st.empty()) {
            TreeNode* cur = st.top();
            st.pop();
            if (cur != nullptr) {
                res.push_back(cur->val);
                // Note given preorder is middle -> left -> right and 
                // we are using stack to emulate it,  we need to make
                // sure we first push right, so we examine left before 
                // riht.
                st.push(cur->right);
                st.push(cur->left);
            }
        }
        return res;
    }
};

// Iterative alternative: more general format
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        TreeNode* cur = root;
        vector<int> res;
        if (!root) return res;
        while (cur || !st.empty()) {
            while (cur) {
                res.push_back(cur->val);
                st.push(cur);
                cur = cur->left;
            }
            // The node has been examined, and we will examnie right immediately.
            // Safe to pop now.
            cur = st.top();
            st.pop();
            // Already examined middle and left nodes, now go to the right.
            cur = cur->right;
        }
        return res;
    }
};

 /*
    145. Binary Tree Postorder Traversal
    https://leetcode.com/problems/binary-tree-postorder-traversal/
    Given the root of a binary tree, return the postorder traversal of its nodes' values.

    Example 1:
    Input: root = [1,null,2,3]
    Output: [3,2,1]

    Example 2:
    Input: root = []
    Output: []

    Example 3:
    Input: root = [1]
    Output: [1]
    

    Constraints:
    The number of the nodes in the tree is in the range [0, 100].
    -100 <= Node.val <= 100

    Follow up: Recursive solution is trivial, could you do it iteratively?
 */
// Recursive
class Solution {
private:
    void traversal(TreeNode* node, vector<int>& v) {
        if (node == nullptr) return;
        traversal(node->left, v);
        traversal(node->right, v);
        v.push_back(node->val);
    }
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        traversal(root, res);
        return res;
    }
};

// Iterative
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> res;
        st.push(root);
        while (!st.empty()) {
            TreeNode* cur = st.top();
            st.pop();

            if (cur != nullptr) {
                res.push_back(cur->val);
                // We will reverse the res at the end, make sure that
                // the order is middle -> right -> left in the res 
                // now!
                st.push(cur->left);
                st.push(cur->right);
            }
        }
        reverse(res.begin(), res.end());
        return res;
    }
};

 /*
    94. Binary Tree Inorder Traversal
    https://leetcode.com/problems/binary-tree-inorder-traversal/
    Given the root of a binary tree, return the inorder traversal of its nodes' values.
 

    Example 1:
    Input: root = [1,null,2,3]
    Output: [1,3,2]

    Example 2:
    Input: root = []
    Output: []

    Example 3:
    Input: root = [1]
    Output: [1]
    

    Constraints:

    The number of nodes in the tree is in the range [0, 100].
    -100 <= Node.val <= 100
    
    Follow up: Recursive solution is trivial, could you do it iteratively?
 */
// Recursive
class Solution {
private:
    void traversal(TreeNode* node, vector<int>& res) {
        if (node == nullptr) return;
        traversal(node->left, res);
        res.push_back(node->val);
        traversal(node->right, res);
    }
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        traversal(root, res);
        return res;
    }
};

// Iterative
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
       TreeNode* cur = root;
       stack<TreeNode*> st;
       vector<int> res;
       while (cur || !st.empty()) {
           // Examine left first.
           while (cur) {
               st.push(cur);
               cur = cur->left;
           }
           cur = st.top();
           st.pop();
           res.push_back(cur->val);
           // Now examine the right.
           cur = cur->right;
       } 
       return res;
    }
};

 /*
    102. Binary Tree Level Order Traversal
    https://leetcode.com/problems/binary-tree-level-order-traversal/
    Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).
 
    Example 1:
    Input: root = [3,9,20,null,null,15,7]
    Output: [[3],[9,20],[15,7]]

    Example 2:
    Input: root = [1]
    Output: [[1]]

    Example 3:
    Input: root = []
    Output: []
    

    Constraints:
    The number of nodes in the tree is in the range [0, 2000].
    -1000 <= Node.val <= 1000
 */
// Iterative
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> q;
        TreeNode* cur = root;
        q.push(cur);
        vector<vector<int>> res;
        if (!root) return res;
        int level = 1;
        while (!q.empty()) {
            int level = q.size();
            vector<int> temp;
            for (int i = 0; i < level; ++i) {
                cur = q.front();
                q.pop();
                temp.push_back(cur->val);
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
            res.emplace_back(temp);
        }
        return res;
    }
};
