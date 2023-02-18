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

// Iterative impl: more general format.
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> res;
        TreeNode* cur = root;
        TreeNode* pre = nullptr;
        if (!root) return res;

        while(cur || !st.empty()) {
            while (cur) {
                st.push(cur);
                cur = cur->left;
            }

            // We cannot pop up the node right now, given we still
            // need to investigate the right side :)
            cur = st.top();

            if (cur->right && cur->right != pre) {
                cur = cur->right;
                continue;
            }

            res.push_back(cur->val);
            st.pop();
            // Make sure that we record the previous right node and `reset` cur node.
            // The former to prevent we examine the right node multiple times;
            // The later to ensure that totally discard the information of the right node,
            // so we can skip the inner while loop at line 189.
            pre = cur;
            cur = nullptr;
        }

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

// Recursive
class Solution {
private:
    void traverse(TreeNode* node, vector<vector<int>>& v, int level) {
        if (node == nullptr) return;

        if (v.empty() || level > v.size() - 1) {
            v.push_back(vector<int>());
        }

        v[level].push_back(node->val);

        traverse(node->left, v, level + 1);
        traverse(node->right, v, level + 1);
    }

public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        traverse(root, res, 0);
        return res;
    }
};

 /*
    226. Invert Binary Tree
    https://leetcode.com/problems/invert-binary-tree/
    Given the root of a binary tree, invert the tree, and return its root.

    Example 1:
    Input: root = [4,2,7,1,3,6,9]
    Output: [4,7,2,9,6,3,1]

    Example 2:
    Input: root = [2,1,3]
    Output: [2,3,1]

    Example 3:
    Input: root = []
    Output: []
    

    Constraints:
    The number of nodes in the tree is in the range [0, 100].
    -100 <= Node.val <= 100
 */
// Recursive
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;

        TreeNode* left = root->left;
        TreeNode* right = root->right;
        root->right = invertTree(left);
        root->left = invertTree(right);
        return root;
    }
};

// Iterative:
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        queue<TreeNode*> q;
        if (!root) return root;
        q.push(root);

        while (!q.empty()) {
            int level = q.size();
            for (int i = 0; i < level; ++i) {
                TreeNode* cur = q.front();
                q.pop();
                TreeNode* temp = cur->left;
                cur->left = cur->right;
                cur->right = temp;
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
        }
        return root;
    }
};

 /*
    101. Symmetric Tree
    https://leetcode.com/problems/symmetric-tree/
    Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

    Example 1:
    Input: root = [1,2,2,3,4,4,3]
    Output: true

    Example 2:
    Input: root = [1,2,2,null,3,null,3]
    Output: false
    
    Constraints:
    The number of nodes in the tree is in the range [1, 1000].
    -100 <= Node.val <= 100
    
    Follow up: Could you solve it both recursively and iteratively?
 */
// Recursive
class Solution {
private:
    bool checkTrees(TreeNode* left, TreeNode* right) {
        if ((!left && right) || (!right && left)) return false;
        if (left == right && left == nullptr) return true;
        if (left->val != right->val) return false;

        // Note we need to check the left and right child trees and determine
        // whether they are symmetric!
        return checkTrees(left->left, right -> right) && checkTrees(left->right, right->left);

    }
public:
    bool isSymmetric(TreeNode* root) {
        if (!root) return true;
        return checkTrees(root->left, root -> right);
    }
};

// Iterative
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        queue<TreeNode*> q;
        if (!root) return true;

        q.push(root->left);
        q.push(root->right);

        // This is not a level order traversal.
        while (!q.empty()) {
            TreeNode* left = q.front();
            q.pop();
            TreeNode* right = q.front();
            q.pop();
            // Reaches the leaf!
            if (!left && !right) continue;
            if (left && !right) return false;
            if (!left && right) return false;
            if (left->val != right->val) return false;
            // Note how we orgnize the nodes in pairs!
            q.push(left->left);
            q.push(right->right);
            q.push(left->right);
            q.push(right->left);
        }

        return true;
    }
};

 /*
    572. Subtree of Another Tree
    https://leetcode.com/problems/subtree-of-another-tree/
    Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

    A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.


    Example 1:
    Input: root = [3,4,5,1,2], subRoot = [4,1,2]
    Output: true

    Example 2:
    Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
    Output: false
    

    Constraints:
    The number of nodes in the root tree is in the range [1, 2000].
    The number of nodes in the subRoot tree is in the range [1, 1000].
    -104 <= root.val <= 104
    -104 <= subRoot.val <= 104
 */
// Recursive
class Solution {
private:
    bool checkTrees(TreeNode* left, TreeNode* right) {
        if ((!left && right) || (!right && left)) return false;
        if (left == right && left == nullptr) return true;
        if (left->val != right->val) return false;

        // Note we need to check the left and right child trees and determine
        // whether they are symmetric!
        return checkTrees(left->left, right -> right) && checkTrees(left->right, right->left);

    }
public:
    bool isSymmetric(TreeNode* root) {
        if (!root) return true;
        return checkTrees(root->left, root -> right);
    }
};

// Solution 2
// Serialize the tree and apply the KMP to determine whether we can find the subtree
// In the pattern. From the official solution.
class Solution {
public:
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
         // Serialize given Nodes
        string r = "";
        serialize(root, r);
        string s = "";
        serialize(subRoot, s);
                 
         // Check if s is in r
         return kmp(s, r);
     }
    
    // Function to serialize Tree
    void serialize(TreeNode* node, string& treeStr) {
        if (node == nullptr){
            treeStr += "#";
            return;
        }

        treeStr += "^";
        treeStr += to_string(node->val);
        serialize(node->left, treeStr);
        serialize(node->right, treeStr);
    }

    // Knuth-Morris-Pratt algorithm to check if `needle` is in `haystack` or not
    bool kmp(string needle, string haystack) {
        int m = needle.length();
        int n = haystack.length();
        
        if (n < m)
            return false;
        
        // longest proper prefix which is also suffix
        vector<int> lps(m);
        // Length of Longest Border for prefix before it.
        int prev = 0;
        // Iterating from index-1. lps[0] will always be 0
        int i = 1;
        
        while (i < m) {
            if (needle[i] == needle[prev]) {
                // Length of Longest Border Increased
                prev += 1;
                lps[i] = prev;
                i += 1;
            } else {
                // Only empty border exist
                if (prev == 0) {
                    lps[i] = 0;
                    i += 1;
                } else {
                    // Try finding longest border for this i with reduced prev
                    prev = lps[prev-1];
                }
            }
        }
        
        // Pointer for haystack
        int haystackPointer = 0;
        // Pointer for needle.
        // Also indicates number of characters matched in current window.
        int needlePointer = 0;
        
        while (haystackPointer < n) {
            if (haystack[haystackPointer] == needle[needlePointer]) {
                // Matched Increment Both
                needlePointer += 1;
                haystackPointer += 1;
                // All characters matched
                if (needlePointer == m)
                    return true;                
            } else {                
                if (needlePointer == 0) {
                    // Zero Matched
                    haystackPointer += 1;                    
                } else {
                    // Optimally shift left needlePointer. Don't change haystackPointer
                    needlePointer = lps[needlePointer-1];
                }
            }
        }
        
        return false;
    }
};

// Solution 3:
// Iterate the tree and build the hash table to determine whether there is a sub-tree.
// From official solution.
class Solution {
   public:
    // CONSTANTS
    const int MOD_1 = 1000000007;
    const int MOD_2 = 2147483647;

    // Hashing a Node
    pair<unsigned long long, unsigned long long> hashSubtreeAtNode(TreeNode* node, bool needToAdd) {
        if (node == nullptr) return {3, 7};

        auto left = hashSubtreeAtNode(node->left, needToAdd);
        auto right = hashSubtreeAtNode(node->right, needToAdd);

        auto left1 = (left.first << 5) % MOD_1;
        auto right1 = (right.first << 1) % MOD_1;
        auto left2 = (left.second << 7) % MOD_2;
        auto right2 = (right.second << 1) % MOD_2;

        pair hashpair = {(left1 + right1 + node->val) % MOD_1,
                         (left2 + right2 + node->val) % MOD_2};

        if (needToAdd) memo.push_back(hashpair);

        return hashpair;
    }

    // Vector to store hashed value of each node.
    vector<pair<unsigned long long, unsigned long long>> memo;

    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        // Calling and adding hash to vector
        hashSubtreeAtNode(root, true);

        // Storing hashed value of subRoot for comparison
        pair<unsigned long long, unsigned long long> s = hashSubtreeAtNode(subRoot, false);

        // Check if hash of subRoot is present in memo
        return find(memo.begin(), memo.end(), s) != memo.end();
    }
};
