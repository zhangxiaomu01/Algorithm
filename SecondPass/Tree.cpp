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


 /*
    104. Maximum Depth of Binary Tree
    https://leetcode.com/problems/maximum-depth-of-binary-tree/
    Given the root of a binary tree, return its maximum depth.
    A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.


    Example 1:
    Input: root = [3,9,20,null,null,15,7]
    Output: 3

    Example 2:
    Input: root = [1,null,2]
    Output: 2
    

    Constraints:
    The number of nodes in the tree is in the range [0, 104].
    -100 <= Node.val <= 100
 */
// Recursive
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;

        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};

// Iterative: level order traversal
class Solution {
public:
    int maxDepth(TreeNode* root) {
        queue<TreeNode*> q;
        if (!root) return 0;
        q.push(root);
        int res = 0;
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                TreeNode* cur = q.front();
                q.pop();
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
            res++;
        }
        return res;
    }
};

 /*
    111. Minimum Depth of Binary Tree
    https://leetcode.com/problems/minimum-depth-of-binary-tree/
    Given a binary tree, find its minimum depth.
    The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

    Note: A leaf is a node with no children.

    

    Example 1:
    Input: root = [3,9,20,null,null,15,7]
    Output: 2

    Example 2:
    Input: root = [2,null,3,null,4,null,5,null,6]
    Output: 5
    

    Constraints:
    The number of nodes in the tree is in the range [0, 105].
    -1000 <= Node.val <= 1000
 */
// Recursive
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root) return 0;

        // We reached the leaf nodes! Note the minimum depth is the depth from root to the **leaf**!
        if (!root->left && !root->right) return 1;

        // If one side is null, then we can only get min height from the other side!
        if (!root->left) return minDepth(root->right) + 1;

        if (!root->right) return minDepth(root->left) + 1;
        
        return 1 + min(minDepth(root->left), minDepth(root->right));
    }
};

// Iterative
class Solution {
public:
    int minDepth(TreeNode* root) {
        queue<TreeNode*> q;
        if (!root) return 0;
        q.push(root);
        int res = 0;
         while (!q.empty()) {
            res++;
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                TreeNode* cur = q.front();
                if (!cur->left && !cur->right) return res;
                q.pop();
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
            
        }
        return res;
    }
};

 /*
    222. Count Complete Tree Nodes
    https://leetcode.com/problems/count-complete-tree-nodes/
    Given the root of a complete binary tree, return the number of the nodes in the tree.
    According to Wikipedia, every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.
    Design an algorithm that runs in less than O(n) time complexity.

    

    Example 1:
    Input: root = [1,2,3,4,5,6]
    Output: 6

    Example 2:
    Input: root = []
    Output: 0

    Example 3:
    Input: root = [1]
    Output: 1

    Constraints:
    The number of nodes in the tree is in the range [0, 5 * 104].
    0 <= Node.val <= 5 * 104
    The tree is guaranteed to be complete.
 */
 // For this impl, the complexity is O(lgn * lgn).
 // The key insight is to detect the complete and full sub trees. O(n) solution is trivial, omit here.
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (!root) return 0;

        int leftDepth = 0;
        int rightDepth = 0;
        TreeNode* left = root->left;
        TreeNode* right = root->right;
        while (left) {
            left = left->left;
            leftDepth ++;
        }

        while(right) {
            right = right->right;
            rightDepth ++;
        }

        // Note for a complete binary tree, if left most depth == right most depth, then 
        // it is a complete and full binary tree, we can easily calculate the node number
        // with 2^height - 1;
        if (leftDepth == rightDepth) {
            // Equivalent to 2 << leftDepth - 1;
            return pow(2, leftDepth + 1) - 1;
        }

        // We only do the search if we found out that the sub tree is not a complete full tree.
        // Which roughly will discard about half of the nodes after each search!
        return countNodes(root->left) + countNodes(root->right) + 1;
    }
};

 /*
    110. Balanced Binary Tree
    https://leetcode.com/problems/balanced-binary-tree/
    Given a binary tree, determine if it is height-balanced.

    
    Example 1:
    Input: root = [3,9,20,null,null,15,7]
    Output: true

    Example 2:
    Input: root = [1,2,2,3,3,null,null,4,4]
    Output: false

    Example 3:
    Input: root = []
    Output: true
    

    Constraints:
    The number of nodes in the tree is in the range [0, 5000].
    -104 <= Node.val <= 104
 */
class Solution {
private:
    int countHeight(TreeNode* node) {
        if (!node) return 0;

        int leftHeight = countHeight(node->left);
        if (leftHeight == -1) return -1;
        int rightHeight = countHeight(node->right);
        if (rightHeight == -1) return -1;

        if (abs(leftHeight - rightHeight) > 1) return -1;

        // We will always return the height of the current node.
        return max(leftHeight, rightHeight) + 1;

    }

public:
    bool isBalanced(TreeNode* root) {
        if (!root) return true;

        return countHeight(root) == -1 ? false : true;
    }
};

 /*
    257. Binary Tree Paths
    https://leetcode.com/problems/binary-tree-paths/
    Given the root of a binary tree, return all root-to-leaf paths in any order.
    A leaf is a node with no children.

    
    Example 1:
    Input: root = [1,2,3,null,5]
    Output: ["1->2->5","1->3"]

    Example 2:
    Input: root = [1]
    Output: ["1"]
    

    Constraints:
    The number of nodes in the tree is in the range [1, 100].
    -100 <= Node.val <= 100
 */
// Backtracking! Early impl:
class Solution {
private:

    void traversal(TreeNode* cur, vector<int>& path, vector<string>& result) {
        path.push_back(cur->val);
        if (cur->left == NULL && cur->right == NULL) {
            string sPath;
            for (int i = 0; i < path.size() - 1; i++) {
                sPath += to_string(path[i]);
                sPath += "->";
            }
            sPath += to_string(path[path.size() - 1]);
            result.push_back(sPath);
            return;
        }
        if (cur->left) { // Left
            traversal(cur->left, path, result);
            path.pop_back(); // backtrack left node
        }
        if (cur->right) { // right
            traversal(cur->right, path, result);
            path.pop_back(); // backtrack right node
        }
    }

public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> result;
        vector<int> path;
        if (root == NULL) return result;
        traversal(root, path, result);
        return result;
    }
};

// Slightly optimized code
class Solution {
private:
    // Note we copy the path each time. This will "revert" our change to the path
    // whenever we do the backtracking.
    void buildPath(vector<string>& res, string path, TreeNode* node) {
        if (!node) return;

        path += to_string(node->val);
        if (!node -> left && !node->right) {
            res.push_back(path);
            return;
        }
        buildPath(res, path + "->", node->left);
        buildPath(res, path + "->", node->right);
    }

public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> res;
        string path = "";
        buildPath(res, path, root);
        return res;
    }
};

// Iterative solution
// Warning: trying to implement the general formmat iterative algorithm to solve this question
// is super chanllenging!!!
class Solution {
public:
    vector<string> binaryTreePaths(TreeNode* root) {
        stack<TreeNode*> st;
        stack<string> path;
        vector<string> res;
        if (!root) return res;
        st.push(root);
        path.push(to_string(root->val));

        while (!st.empty()) {
            TreeNode* cur = st.top();
            string pre = path.top();
            st.pop();
            path.pop();

            if (!cur->left && !cur->right) res.push_back(pre);
            // Pre-order traversal!
            if (cur->right) {
                path.push(pre + "->" + to_string(cur->right->val));
                st.push(cur->right);
            }

            if (cur->left) {
                path.push(pre + "->" + to_string(cur->left->val));
                st.push(cur->left);
            }

        }
        return res;
    }
};

 /*
    404. Sum of Left Leaves
    https://leetcode.com/problems/sum-of-left-leaves/
    Given the root of a binary tree, return the sum of all left leaves.
    A leaf is a node with no children. A left leaf is a leaf that is the left child of another node.

    
    Example 1:
    Input: root = [3,9,20,null,null,15,7]
    Output: 24
    Explanation: There are two left leaves in the binary tree, with values 9 and 15 respectively.

    Example 2:
    Input: root = [1]
    Output: 0
    

    Constraints:
    The number of nodes in the tree is in the range [1, 1000].
    -1000 <= Node.val <= 1000
 */
// Recursive
class Solution {
private:

    void traversal(TreeNode* cur, vector<int>& path, vector<string>& result) {
        path.push_back(cur->val);
        if (cur->left == NULL && cur->right == NULL) {
            string sPath;
            for (int i = 0; i < path.size() - 1; i++) {
                sPath += to_string(path[i]);
                sPath += "->";
            }
            sPath += to_string(path[path.size() - 1]);
            result.push_back(sPath);
            return;
        }
        if (cur->left) { // Left
            traversal(cur->left, path, result);
            path.pop_back(); // backtrack left node
        }
        if (cur->right) { // right
            traversal(cur->right, path, result);
            path.pop_back(); // backtrack right node
        }
    }

public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> result;
        vector<int> path;
        if (root == NULL) return result;
        traversal(root, path, result);
        return result;
    }
};

// Iterative
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        stack<TreeNode*> st;
        stack<bool> isLeft;
        TreeNode* cur = root;
        if (!root) return 0;
        int sum = 0;
        bool isRight = true;

        // Inorder traversal.
        while(cur || !st.empty()) {
            while (cur) {
                st.push(cur);
                isLeft.push(isRight ? false : true);
                isRight = false;
                cur = cur->left;
            }

            cur = st.top();
            st.pop();
            if (isLeft.top() && !cur->left && !cur->right) sum += cur->val;
            isLeft.pop();

            cur = cur->right;
            isRight = true;
        }
        return sum;
    }
};

 /*
    513. Find Bottom Left Tree Value
    https://leetcode.com/problems/find-bottom-left-tree-value/
    Given the root of a binary tree, return the leftmost value in the last row of the tree.


    Example 1:
    Input: root = [2,1,3]
    Output: 1

    Example 2:
    Input: root = [1,2,3,4,null,5,6,null,null,7]
    Output: 7
    

    Constraints:
    The number of nodes in the tree is in the range [1, 104].
    -231 <= Node.val <= 231 - 1
 */
// Recursive
class Solution {
public:
    int maxDepth = INT_MIN;
    int result;
    void traversal(TreeNode* root, int depth) {
        if (root->left == NULL && root->right == NULL) {
            if (depth > maxDepth) {
                maxDepth = depth;
                result = root->val;
            }
            return;
        }
        if (root->left) {
            depth++;
            traversal(root->left, depth);
            depth--; // backtracking
        }
        if (root->right) {
            depth++;
            traversal(root->right, depth);
            depth--; // backtracking
        }
        return;
    }
    int findBottomLeftValue(TreeNode* root) {
        traversal(root, 0);
        return result;
    }
};

// Iterative
class Solution {
private:

    void traversal(TreeNode* cur, vector<int>& path, vector<string>& result) {
        path.push_back(cur->val);
        if (cur->left == NULL && cur->right == NULL) {
            string sPath;
            for (int i = 0; i < path.size() - 1; i++) {
                sPath += to_string(path[i]);
                sPath += "->";
            }
            sPath += to_string(path[path.size() - 1]);
            result.push_back(sPath);
            return;
        }
        if (cur->left) { // Left
            traversal(cur->left, path, result);
            path.pop_back(); // backtrack left node
        }
        if (cur->right) { // right
            traversal(cur->right, path, result);
            path.pop_back(); // backtrack right node
        }
    }

public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> result;
        vector<int> path;
        if (root == NULL) return result;
        traversal(root, path, result);
        return result;
    }
};

 /*
    112. Path Sum
    https://leetcode.com/problems/path-sum/
    Given the root of a binary tree and an integer targetSum, return true if the tree has
    a root-to-leaf path such that adding up all the values along the path equals targetSum.
    A leaf is a node with no children.

    

    Example 1:
    Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
    Output: true
    Explanation: The root-to-leaf path with the target sum is shown.

    Example 2:
    Input: root = [1,2,3], targetSum = 5
    Output: false
    Explanation: There two root-to-leaf paths in the tree:
    (1 --> 2): The sum is 3.
    (1 --> 3): The sum is 4.
    There is no root-to-leaf path with sum = 5.

    Example 3:
    Input: root = [], targetSum = 0
    Output: false
    Explanation: Since the tree is empty, there are no root-to-leaf paths.
    
    Constraints:
    The number of nodes in the tree is in the range [0, 5000].
    -1000 <= Node.val <= 1000
    -1000 <= targetSum <= 1000
 */
// Recursive
class Solution {
private:
    int target = -1;
    bool detectTarget(TreeNode* node, int currentSum, int target) {
        if (!node) return false;
        if (!node->left && !node->right) return currentSum + node->val == target;
        return detectTarget(node->left, currentSum + node->val, target) 
                || detectTarget(node->right, currentSum + node->val, target);
    }
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        return detectTarget(root, 0, targetSum);
    }
};

// Iterative
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        stack<TreeNode*> st;
        stack<int> sums;
        if (!root) return false;
        st.push(root);
        sums.push(root->val);
        while (!st.empty()) {
            TreeNode* cur = st.top();
            st.pop();
            int curSum = sums.top();
            sums.pop();
            if (!cur -> left && !cur -> right && curSum == targetSum) return true;
            if (cur -> left) {
                st.push(cur->left);
                sums.push(curSum + cur->left->val);
            } 
            if (cur -> right) {
                st.push(cur->right);
                sums.push(curSum + cur->right->val);
            }
        }
        return false;
    }
};

 /*
    113. Path Sum II
    https://leetcode.com/problems/path-sum-ii/
    Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of the node
    values in the path equals targetSum. Each path should be returned as a list of the node values, not node references.
    A root-to-leaf path is a path starting from the root and ending at any leaf node. A leaf is a node with no children.

    
    Example 1:
    Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
    Output: [[5,4,11,2],[5,8,4,5]]
    Explanation: There are two paths whose sum equals targetSum:
    5 + 4 + 11 + 2 = 22
    5 + 8 + 4 + 5 = 22

    Example 2:
    Input: root = [1,2,3], targetSum = 5
    Output: []

    Example 3:
    Input: root = [1,2], targetSum = 0
    Output: []
    

    Constraints:
    The number of nodes in the tree is in the range [0, 5000].
    -1000 <= Node.val <= 1000
    -1000 <= targetSum <= 1000
 */
// Recursive
class Solution {
    void generateSum(TreeNode* node, int currentSum, int targetSum, vector<vector<int>>& res, vector<int>& path) {
        if (!node) return;
        if (!node->left && !node->right && currentSum + node->val == targetSum) {
            path.push_back(node->val);
            res.push_back(path);
            // Backtracking.
            path.pop_back();
            return;
        }

        path.push_back(node->val);
        generateSum(node->left, currentSum + node->val, targetSum, res, path);
        generateSum(node->right, currentSum + node->val, targetSum, res, path);
        path.pop_back(); // Backtracking
    }
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> res;
        vector<int> path;
        generateSum(root, 0, targetSum, res, path);
        return res;
    }
};

// Iterative
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> res;
        stack<vector<int>> paths;
        stack<TreeNode*> st;
        stack<int> sums;
        if (!root) return res;

        st.push(root);
        sums.push(root->val);
        paths.push({ root->val });

        while(!st.empty()) {
            TreeNode* cur = st.top();
            st.pop();
            vector<int> path = paths.top();
            paths.pop();
            int curSum = sums.top();
            sums.pop();

            if (!cur->left && !cur->right && curSum == targetSum) res.push_back(path);

            if (cur->left) {
                st.push(cur->left);
                path.push_back(cur->left->val);
                paths.push(path);
                path.pop_back();
                sums.push(curSum + cur->left->val);
            }

            if (cur->right) {
                st.push(cur->right);
                path.push_back(cur->right->val);
                paths.push(path);
                path.pop_back();
                sums.push(curSum + cur->right->val);
            }
        }
        return res;
    }
};

// Iterative: optimized. TBD
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        stack<TreeNode*> st;
        vector<int> path;
        vector<vector<int>> res;
        int curSum = 0;
        if (!root) return res;
        TreeNode* cur = root;
        TreeNode* pre = nullptr;

        while (cur || !st.empty()) {
            while (cur) {
                st.push(cur);
                path.push_back(cur->val);
                curSum += cur->val;
                cur = cur->left;
            }

            cur = st.top();
            // Examine the right
            if (cur -> right && cur->right != pre) {
                pre = cur;
                cur = cur->right;
                continue;
            }
            
            if (!cur->left && !cur->right && curSum == targetSum) {
                res.push_back(path);
            }
            pre = cur;
            // Backtracking
            curSum -= cur->val;
            path.pop_back();
            st.pop();
            cur = nullptr;
        }
        return res;
    }
};


 /*
    106. Construct Binary Tree from Inorder and Postorder Traversal
    https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
    Given two integer arrays inorder and postorder where inorder is the inorder traversal of a binary tree and
    postorder is the postorder traversal of the same tree, construct and return the binary tree.

    Example 1:
    Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
    Output: [3,9,20,null,null,15,7]

    Example 2:
    Input: inorder = [-1], postorder = [-1]
    Output: [-1]
    

    Constraints:
    1 <= inorder.length <= 3000
    postorder.length == inorder.length
    -3000 <= inorder[i], postorder[i] <= 3000
    inorder and postorder consist of **unique** values.
    Each value of postorder also appears in inorder.
    inorder is guaranteed to be the inorder traversal of the tree.
    postorder is guaranteed to be the postorder traversal of the tree.
 */
// Recursive
// There is an interative solution which utilizes the unique value from the array, not general enough and hard
// to understand at first glance. Ignored.
class Solution {
private:
    TreeNode* construct(vector<int>& inorder, int inorderStart, int inorderEnd, vector<int>& postOrder, int postOrderStart, int postOrderEnd) {
        if (postOrderStart >= postOrderEnd || postOrderEnd < 1) return nullptr;
        
        TreeNode* cur = new TreeNode(postOrder[postOrderEnd - 1]);

        int newInorderEnd = 0;
        for (int i = inorderStart; i < inorderEnd; ++i) {
            if (inorder[i] == postOrder[postOrderEnd - 1]) {
                newInorderEnd = i;
                break;
            }
        }
        int newPostOrderEnd = postOrderStart + (newInorderEnd - inorderStart);
        cur->left = construct(inorder, inorderStart, newInorderEnd, postOrder, postOrderStart, newPostOrderEnd);
        // Note we need to exclude the last element from postOrder array!
        // We need to make sure the new post order start is exacly the same as `newPostOrderEnd`, because in postorder
        // array, we do not have a `root` node sit in between left child and right child!
        cur->right = construct(inorder, newInorderEnd + 1, inorderEnd, postOrder, newPostOrderEnd, postOrderEnd - 1);
        return cur;
    }

public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        return construct(inorder, 0, inorder.size(), postorder, 0, postorder.size());
    }
};

 /*
    105. Construct Binary Tree from Preorder and Inorder Traversal
    https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
    Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary
    tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.
 
    Example 1:
    Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
    Output: [3,9,20,null,null,15,7]

    Example 2:
    Input: preorder = [-1], inorder = [-1]
    Output: [-1]
    

    Constraints:
    1 <= preorder.length <= 3000
    inorder.length == preorder.length
    -3000 <= preorder[i], inorder[i] <= 3000
    preorder and inorder consist of unique values.
    Each value of inorder also appears in preorder.
    preorder is guaranteed to be the preorder traversal of the tree.
    inorder is guaranteed to be the inorder traversal of the tree.
 */
// Recursive
class Solution {
private:
    TreeNode* construct(vector<int>& inorder, int inorderStart, int inorderEnd, vector<int>& preorder, int preorderStart, int preorderEnd) {
        if (preorderStart >= preorderEnd) return nullptr;
        
        TreeNode* cur = new TreeNode(preorder[preorderStart]);

        int newInorderEnd = 0;
        for (int i = inorderStart; i < inorderEnd; ++i) {
            if (inorder[i] == preorder[preorderStart]) {
                newInorderEnd = i;
                break;
            }
        }
        int newPreorderEnd = preorderStart + 1 + (newInorderEnd - inorderStart);
        // Needs to exclude the left most node in preorder array.
        cur->left = construct(inorder, inorderStart, newInorderEnd, preorder, preorderStart + 1, newPreorderEnd);
        cur->right = construct(inorder, newInorderEnd + 1, inorderEnd, preorder, newPreorderEnd, preorderEnd);
        return cur;
    }
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return construct(inorder, 0, inorder.size(), preorder, 0, preorder.size());
    }
};

