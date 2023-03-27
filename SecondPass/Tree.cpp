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


 /*
    654. Maximum Binary Tree
    https://leetcode.com/problems/maximum-binary-tree/
    You are given an integer array nums with no duplicates. A maximum binary tree can be built recursively
    from nums using the following algorithm:

    Create a root node whose value is the maximum value in nums.
    Recursively build the left subtree on the subarray prefix to the left of the maximum value.
    Recursively build the right subtree on the subarray suffix to the right of the maximum value.
    Return the maximum binary tree built from nums.

    

    Example 1:
    Input: nums = [3,2,1,6,0,5]
    Output: [6,3,5,null,2,0,null,null,1]
    Explanation: The recursive calls are as follow:
    - The largest value in [3,2,1,6,0,5] is 6. Left prefix is [3,2,1] and right suffix is [0,5].
        - The largest value in [3,2,1] is 3. Left prefix is [] and right suffix is [2,1].
            - Empty array, so no child.
            - The largest value in [2,1] is 2. Left prefix is [] and right suffix is [1].
                - Empty array, so no child.
                - Only one element, so child is a node with value 1.
        - The largest value in [0,5] is 5. Left prefix is [0] and right suffix is [].
            - Only one element, so child is a node with value 0.
            - Empty array, so no child.

    Example 2:
    Input: nums = [3,2,1]
    Output: [3,null,2,null,1]
    

    Constraints:
    1 <= nums.length <= 1000
    0 <= nums[i] <= 1000
    All integers in nums are unique.
 */
// Recursive
class Solution {
private:
    int getMaxElementIndex (vector<int>& nums, int start, int end) {
        int res = -1;
        int curMax = INT_MIN;
        for (int i = start; i < end; ++i) {
            if (curMax < nums[i]) {
                curMax = nums[i];
                res = i;
            }
        }
        return res;
    }
    TreeNode* buildTree(vector<int>& nums, int start, int end) {
        if (start >= end) return nullptr;

        int maxIndex = getMaxElementIndex(nums, start, end);

        TreeNode* node = new TreeNode(nums[maxIndex]);
        node->left = buildTree(nums, start, maxIndex);
        node->right = buildTree(nums, maxIndex + 1, end);
        return node;
    }
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return buildTree(nums, 0, nums.size());
    }
};

 /*
    617. Merge Two Binary Trees
    https://leetcode.com/problems/merge-two-binary-trees/
    You are given two binary trees root1 and root2.
    Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped
    while the others are not. You need to merge the two trees into a new binary tree. The merge rule is
    that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the
    NOT null node will be used as the node of the new tree.

    Return the merged tree.

    Note: The merging process must start from the root nodes of both trees.

    

    Example 1:
    Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
    Output: [3,4,5,5,4,null,7]

    Example 2:
    Input: root1 = [1], root2 = [1,2]
    Output: [2,2]
    

    Constraints:
    The number of nodes in both trees is in the range [0, 2000].
    -104 <= Node.val <= 104
 */
// Recursive
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        if (!root1 && !root2) return nullptr;

        TreeNode* cur = nullptr;
        if (!root1) cur = root2;
        else if (!root2) cur = root1;
        else {
            cur = new TreeNode(root1->val + root2->val);
        }

        cur->left = mergeTrees(root1 ? root1->left : nullptr, root2 ? root2->left : nullptr);
        cur->right = mergeTrees(root1 ? root1->right : nullptr, root2 ? root2->right : nullptr);

        return cur;
    }
};

// Recursive: slightly optimized code
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        if (t1 == nullptr) return t2;
        if (t2 == nullptr) return t1;
        // 重新定义新的节点，不修改原有两个树的结构
        TreeNode* root = new TreeNode(0);
        root->val = t1->val + t2->val;
        root->left = mergeTrees(t1->left, t2->left);
        root->right = mergeTrees(t1->right, t2->right);
        return root;
    }
};

// Iterative: level order traversal!
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        if (t1 == NULL) return t2;
        if (t2 == NULL) return t1;
        queue<TreeNode*> que;
        que.push(t1);
        que.push(t2);
        while(!que.empty()) {
            TreeNode* node1 = que.front(); que.pop();
            TreeNode* node2 = que.front(); que.pop();
            node1->val += node2->val;


            if (node1->left != NULL && node2->left != NULL) {
                que.push(node1->left);
                que.push(node2->left);
            }

            if (node1->right != NULL && node2->right != NULL) {
                que.push(node1->right);
                que.push(node2->right);
            }

            // Assign the subtree here is the key!
            if (node1->left == NULL && node2->left != NULL) {
                node1->left = node2->left;
            }
            // Assign the subtree here is the key!
            if (node1->right == NULL && node2->right != NULL) {
                node1->right = node2->right;
            }
        }
        return t1;
    }
};

 /*
    98. Validate Binary Search Tree
    https://leetcode.com/problems/validate-binary-search-tree/
    Given the root of a binary tree, determine if it is a valid binary search tree (BST).

    A valid BST is defined as follows:
    The left subtreeof a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.
    

    Example 1:
    Input: root = [2,1,3]
    Output: true

    Example 2:
    Input: root = [5,1,4,null,null,3,6]
    Output: false
    Explanation: The root node's value is 5 but its right child's value is 4.
    

    Constraints:
    The number of nodes in the tree is in the range [1, 104].
    -2^31 <= Node.val <= 2^31 - 1
 */
// Recursive: inorder traversal and see whether the visited nodes is ordered from small to large!
class Solution {
private:
    TreeNode* pre = nullptr;
public:
    bool isValidBST(TreeNode* root) {
        if (!root) return true;

        bool res = isValidBST(root->left);
        // Saves the previous node! Check that current is greater than previous one!
        if (pre && pre->val >= root->val) return false;
        pre = root;
        return res && isValidBST(root->right);
    }
};

// Recursive
class Solution {
private:
    bool isBST(TreeNode* node, TreeNode* minNode, TreeNode* maxNode) {
        if (!node) return true;

        // Note we need to maintain the state of minNode & maxNode
        // It's not sufficient to just validate that left child < parent 
        // && right child > parent. Note all nodes from left subtree
        // needs to be smaller than node->val, and from right subtree
        // needs to be greater!
        if (minNode && node->val <= minNode->val) return false;
        if (maxNode && maxNode->val <= node->val) return false;

        // If we go left, we need to make sure all the left nodes still greater than the previous minNode
        // The same rule applies for the maxNode.
        return isBST(node->left, minNode, node) && isBST(node->right, node, maxNode);
    }
public:
    bool isValidBST(TreeNode* root) {
        return isBST(root, nullptr, nullptr);
    }
};

// Iterative
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        stack<TreeNode*> st;
        TreeNode* pre = nullptr;
        TreeNode* cur = root;

        while (cur || !st.empty()) {
            while (cur) {
                st.push(cur);
                cur = cur->left;
            }

            // Inorder traversal, check whether current val > previous val.
            cur = st.top();
            if (pre && pre->val >= cur->val) return false;
            st.pop();
            pre = cur;
            cur = cur->right;
        }
        return true;
    }
};

 /*
    530. Minimum Absolute Difference in BST
    https://leetcode.com/problems/minimum-absolute-difference-in-bst/
    Given the root of a Binary Search Tree (BST), return the minimum absolute difference between the
    values of any two different nodes in the tree.

    Example 1:
    Input: root = [4,2,6,1,3]
    Output: 1

    Example 2:
    Input: root = [1,0,48,null,null,12,49]
    Output: 1
    

    Constraints:
    The number of nodes in the tree is in the range [2, 104].
    0 <= Node.val <= 10^5
 */
// Recursive
class Solution {
private:
    int result = INT_MAX;
    TreeNode* pre = NULL;
    void traversal(TreeNode* cur) {
        if (cur == NULL) return;
        traversal(cur->left); 
        if (pre != NULL){      
            result = min(result, cur->val - pre->val);
        }
        pre = cur; 
        traversal(cur->right); 
    }
public:
    int getMinimumDifference(TreeNode* root) {
        traversal(root);
        return result;
    }
};

// Iterative
class Solution {
public:
    int getMinimumDifference(TreeNode* root) {
        stack<TreeNode*> st;
        TreeNode* cur = root;
        TreeNode* pre = nullptr;
        int res = INT_MAX;

        while(cur || !st.empty()) {
            while(cur) {
                st.push(cur);
                cur = cur->left;
            }

            // In order traversal!
            cur = st.top();
            st.pop();
            if (pre && abs(cur->val - pre->val) < res) res = abs(cur->val - pre->val);

            pre = cur;
            cur = cur->right;
        }
        return res;
    }
};

 /*
    501. Find Mode in Binary Search Tree
    https://leetcode.com/problems/find-mode-in-binary-search-tree/
    Given the root of a binary search tree (BST) with duplicates, return all the mode(s)
    (i.e., the most frequently occurred element) in it.
    If the tree has more than one mode, return them in any order.

    Assume a BST is defined as follows:
    The left subtree of a node contains only nodes with keys less than or equal to the node's key.
    The right subtree of a node contains only nodes with keys greater than or equal to the node's key.
    Both the left and right subtrees must also be binary search trees.
    

    Example 1:
    Input: root = [1,null,2,2]
    Output: [2]

    Example 2:
    Input: root = [0]
    Output: [0]
    

    Constraints:
    The number of nodes in the tree is in the range [1, 104].
    -10^5 <= Node.val <= 10^5
    
    Follow up: Could you do that without using any extra space? 
    (Assume that the implicit stack space incurred due to recursion does not count).
 */
// Recursive: Applying extra space is trivial, the solutions are for follow up!
class Solution {
private:
    int maxCount = -1;
    int count = 0;
    vector<int> res;
    TreeNode* pre = nullptr;

    void traverse(TreeNode* node) {
        if (!node) return;
        
        traverse(node->left);
        // Check whether the current val == previous val!
        // Reset count when the values are different!
        if (!pre || (pre->val != node->val)) count = 1;
        else {
            if (pre->val == node->val) count++;
        }
        if (count > maxCount) {
            maxCount = count;
            // The previous saved result is not based on the maximum count!
            // Needs to be cleared first!
            res.clear();
            res.push_back(node->val);
        } else if (count == maxCount) res.push_back(node->val);
        pre = node;
        traverse(node->right);
    }
public:
    vector<int> findMode(TreeNode* root) {
        traverse(root);
        return res;
    }
};

// Iterative
class Solution {
public:
    vector<int> findMode(TreeNode* root) {
        vector<int> res;
        int maxCount = -1;
        int count = 0;
        TreeNode* pre = nullptr;
        TreeNode* cur = root;
        stack<TreeNode*> st;
        // In order traversal!
        while (cur || !st.empty()) {
            while (cur) {
                st.push(cur);
                cur = cur -> left;
            }

            cur = st.top();
            st.pop();
            if (!pre || pre->val != cur->val) count = 1;
            if (pre && pre->val == cur->val) ++count;
            if (count == maxCount) res.push_back(cur->val);
            else if (count > maxCount) {
                res.clear();
                maxCount = count;
                res.push_back(cur->val);
            }
            pre = cur;

            // Check right!
            cur = cur->right;
        }
        return res;
    }
};

 /*
    236. Lowest Common Ancestor of a Binary Tree
    https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
    Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
    According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between
    two nodes p and q as the lowest node in T that has both p and q as descendants 
    (where we allow a node to be a descendant of itself).”

    

    Example 1:
    Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
    Output: 3
    Explanation: The LCA of nodes 5 and 1 is 3.

    Example 2:
    Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
    Output: 5
    Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.

    Example 3:
    Input: root = [1,2], p = 1, q = 2
    Output: 1
    

    Constraints:
    The number of nodes in the tree is in the range [2, 105].
    -109 <= Node.val <= 109
    All Node.val are unique.
    p != q
    p and q will exist in the tree.
 */
// Recursive
class Solution {
public:
    // Return the node eqauls to either p or q, we will utilize that as a way to 
    // determine whether we have detected p or q.
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || !q || !p) return nullptr;
        
        TreeNode* leftAncestor = lowestCommonAncestor(root->left, p, q);
        TreeNode* rightAncestor = lowestCommonAncestor(root->right, p, q);

        // We have detected either p or q or common ancestor.
        // Note we can safely return root when we detect either p or q.
        // There are two conditions:
        // 1. If q (p) is a child of p (q) and root == p (q), then root is already the lowest common
        // ancestor.
        // 2. If both left & right ancestors are not null, then current root must be the lowest common
        // ancestor. And once we backtrack, from another subtree, we must get a null return.
        // Essentially, we will have the right aggregated to the final return!
        if (root == p || root == q || (leftAncestor && rightAncestor)) return root;
        if (!leftAncestor && rightAncestor) return rightAncestor;
        if (leftAncestor && !rightAncestor) return leftAncestor;
        return nullptr; // leftAncestor == nullptr && rightAncestor == nullptr
    }
};

// Iterative: we need to build the path for that! Not very general.
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return root;
        //We allocate a map to store the ancestor of all child nodes
        unordered_map<TreeNode*, TreeNode*> ancestor;
        stack<TreeNode*> st;
        ancestor.insert({root, nullptr});
        st.push(root);
        while(!ancestor.count(p) || !ancestor.count(q)){
            TreeNode* node = st.top();
            st.pop();
            if(node->right){
                ancestor[node->right] = node;
                st.push(node->right);
            }
            if(node->left){
                ancestor[node->left] = node;
                st.push(node->left);
            }
        }
        unordered_set<TreeNode*> ancestorP;
        TreeNode* pCopy = p;
        //Build the path from root to node p
        while(pCopy){
            ancestorP.insert(pCopy);
            pCopy = ancestor[pCopy];
        }
        //Find the first common ancestor for both nodes p and q
        TreeNode* qCopy = q;
        while(!ancestorP.count(qCopy)){
            ancestorP.insert(qCopy);
            qCopy = ancestor[qCopy];
        }
        return qCopy;
    }
};

 /*
    235. Lowest Common Ancestor of a Binary Search Tree
    https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
    Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.
    According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p
    and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant
    of itself).”

    Example 1:
    Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
    Output: 6
    Explanation: The LCA of nodes 2 and 8 is 6.

    Example 2:
    Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
    Output: 2
    Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

    Example 3:
    Input: root = [2,1], p = 2, q = 1
    Output: 2
    

    Constraints:
    The number of nodes in the tree is in the range [2, 105].
    -10^9 <= Node.val <= 10^9
    All Node.val are unique.
    p != q
    p and q will exist in the BST.
 */
// Recursive: rely on the binary search tree feature
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) return nullptr;

        if (p->val < root-> val && q->val < root->val) return lowestCommonAncestor(root->left, p, q);
        if (p->val > root->val && q->val > root->val) return lowestCommonAncestor(root->right, p, q);
        return root;
    }
};

// Recursive: the more general approach
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) return nullptr;

        TreeNode* leftAncestor = lowestCommonAncestor(root->left, p, q);
        TreeNode* rightAncestor = lowestCommonAncestor(root->right, p, q);

        if (root == p || root == q || (leftAncestor && rightAncestor)) return root;
        if (!leftAncestor && rightAncestor) return rightAncestor;
        if (!rightAncestor && leftAncestor) return leftAncestor;

        return nullptr;
    }
};

// Iterative:
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        TreeNode* cur = root;
        while (cur) {
            if (p->val < cur->val && q->val < cur->val) 
                cur = cur->left;
            else if (p->val > cur->val && q->val > cur->val)
                cur = cur->right;
            else return cur;
        }
        return nullptr;
    }
};

 /*
    701. Insert into a Binary Search Tree
    https://leetcode.com/problems/insert-into-a-binary-search-tree/
    You are given the root node of a binary search tree (BST) and a value to insert into the tree. Return the root node of the BST after the insertion.
    It is guaranteed that the new value does not exist in the original BST.
    Notice that there may exist multiple valid ways for the insertion, as long as the tree remains a BST after
    insertion. You can return any of them.


    Example 1:
    Input: root = [4,2,7,1,3], val = 5
    Output: [4,2,7,1,3,5]
    Explanation: Another accepted tree is:

    Example 2:
    Input: root = [40,20,60,10,30,50,70], val = 25
    Output: [40,20,60,10,30,50,70,null,null,25]

    Example 3:
    Input: root = [4,2,7,1,3,null,null,null,null,null,null], val = 5
    Output: [4,2,7,1,3,5]
    

    Constraints:
    The number of nodes in the tree will be in the range [0, 104].
    -10^8 <= Node.val <= 10^8
    All the values Node.val are unique.
    -10^8 <= val <= 10^8
    It's guaranteed that val does not exist in the original BST.
 */
// Recursive!
 // Note: for this question, we do not need to worry about "rebalance" tree!
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (!root) {
            return new TreeNode(val);
        }
        if (root->val > val) root->left = insertIntoBST(root->left, val);
        if (root->val < val) root->right = insertIntoBST(root->right, val);
        return root;
    }
};

// Iterative!
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        TreeNode* cur = root;
        TreeNode* pre = nullptr;
        if (!root) {
            return new TreeNode(val);
        }
        while (cur) {
            pre = cur;
            if (cur -> val > val) cur = cur->left;
            else cur = cur -> right;
        }

        if (pre->val > val) pre->left = new TreeNode(val);
        else pre->right = new TreeNode(val);
        return root;
    }
};

 /*
    450. Delete Node in a BST
    https://leetcode.com/problems/delete-node-in-a-bst/
    Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference (possibly updated) of the BST.

    Basically, the deletion can be divided into two stages:

    Search for a node to remove.
    If the node is found, delete the node.
    

    Example 1:
    Input: root = [5,3,6,2,4,null,7], key = 3
    Output: [5,4,6,2,null,null,7]
    Explanation: Given key to delete is 3. So we find the node with value 3 and delete it.
    One valid answer is [5,4,6,2,null,null,7], shown in the above BST.
    Please notice that another valid answer is [5,2,6,null,4,null,7] and it's also accepted.

    Example 2:
    Input: root = [5,3,6,2,4,null,7], key = 0
    Output: [5,3,6,2,4,null,7]
    Explanation: The tree does not contain a node with value = 0.

    Example 3:
    Input: root = [], key = 0
    Output: []
    

    Constraints:
    The number of nodes in the tree is in the range [0, 104].
    -10^5 <= Node.val <= 10^5
    Each node has a unique value.
    root is a valid binary search tree.
    -10^5 <= key <= 10^5
    

    Follow up: Could you solve it with time complexity O(height of tree)?
 */
// Recursive: O(N)
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (!root) return root;

        // If root is the leaf
        if (root->val == key && !root->left && !root->right) {
            delete root;
            return nullptr;
        }

        // If right is null
        if (root->val == key && !root->right) {
            TreeNode* left = root->left;
            delete root;
            return left;
        }

        // If left is null
        if (root->val == key && !root->left) {
            TreeNode* right = root->right;
            delete root;
            return right;
        }

        // If root has two children!
        if (root->val == key) {
            // We need to put left sub tree under the leftmost node of right sub-tree.
            TreeNode* ptr = root->right;
            while (ptr->left) ptr = ptr->left;
            ptr->left = root->left;
            TreeNode* res = root->right;
            delete root;
            return res;
        }

        root->left = deleteNode(root->left, key);
        root->right = deleteNode(root->right, key);
        return root;
    }
};

// Recursive: O(h), much harder to implement.
class Solution {
private:
    TreeNode* findMin(TreeNode* root){
        while(root->left){
            root = root->left;
        }
        return root;
    }
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        //Corner case!
        if(!root) return root;
        //key is in right sub tree
        else if (root->val < key) root->right = deleteNode(root->right, key);
        else if (root->val > key) root->left = deleteNode(root->left, key);
        //We find right node
        else{
            //If our node is a leaf
            if(!root->left && !root->right){
                delete root;
                root = nullptr;
            }//If only have right node
            else if(!root->left){
                TreeNode* temp = root;
                root = root->right;
                delete temp;
            }
            else if(!root->right){
                TreeNode* temp = root;
                root = root->left;
                delete temp;
            }//We have two sub trees
            else{
                // Put the left subtree under the minimum node of the right subtree.
                TreeNode* temp = findMin(root->right);
                temp->left = root->left;
                temp = root;
                root = root->right;
                delete temp;
                
            }
        }
        return root;
    }
};

// Iterative:
class Solution {
private:
    // Similar approach as recursive version.
    TreeNode* deleteOneNode(TreeNode* target) {
        if (target == nullptr) return target;
        if (target->right == nullptr) return target->left;
        TreeNode* cur = target->right;
        while (cur->left) {
            cur = cur->left;
        }
        cur->left = target->left;
        return target->right;
    }
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (root == nullptr) return root;
        TreeNode* cur = root;
        TreeNode* pre = nullptr;
        while (cur) {
            if (cur->val == key) break;
            pre = cur;
            if (cur->val > key) cur = cur->left;
            else cur = cur->right;
        }
        if (pre == nullptr) {
            return deleteOneNode(cur);
        }
        if (pre->left && pre->left->val == key) {
            pre->left = deleteOneNode(cur);
        }
        if (pre->right && pre->right->val == key) {
            pre->right = deleteOneNode(cur);
        }
        return root;
    }
};

 /*
    669. Trim a Binary Search Tree
    https://leetcode.com/problems/trim-a-binary-search-tree/
    Given the root of a binary search tree and the lowest and highest boundaries as low and high, trim the tree so that all its elements lies in [low, high]. Trimming the tree should not change the relative structure of the elements that will remain in the tree (i.e., any node's descendant should remain a descendant). It can be proven that there is a unique answer.

    Return the root of the trimmed binary search tree. Note that the root may change depending on the given bounds.

    

    Example 1:


    Input: root = [1,0,2], low = 1, high = 2
    Output: [1,null,2]
    Example 2:


    Input: root = [3,0,4,null,2,null,null,1], low = 1, high = 3
    Output: [3,2,null,1]
    

    Constraints:

    The number of nodes in the tree is in the range [1, 104].
    0 <= Node.val <= 10^4
    The value of each node in the tree is unique.
    root is guaranteed to be a valid binary search tree.
    0 <= low <= high <= 10^4
 */
// Recursive: O(N)
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        if (!root) return nullptr;

        // Note if current node is smaller than the low, then we meed to double check its right
        // children to see whether they can be in [low, high].
        // Note we also potentially return the different root value for this check if root is out
        // the boundry of [low, high].
        if (root->val < low) return trimBST(root->right, low, high);
        if (root->val > high) return trimBST(root->left, low, high);

        root->left = trimBST(root->left, low, high);
        root->right = trimBST(root->right, low, high);
        return root;
    }
};

// Iterative
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        TreeNode* cur = root;
        if (!root) return nullptr;

        while (cur) {
            if (cur -> val < low) cur = cur->right;
            else if (cur -> val > high) cur = cur->left;
            else break;
        }
        if (!cur) return nullptr;

        TreeNode* res = cur;
        // Process left child
        while (cur && cur->left) {
            // Reassign the right subchild to the current left.
            // Here we need to rely on while loop instead of if to avoid the edge case that
            // a leaf needs to be deleted.
            // See an example: [3,1,4,null,2]
            while (cur->left && cur->left->val < low) cur->left = cur->left->right;
            cur = cur->left;
        }

        cur = res;
        // Process right child
        while (cur && cur->right) {
            while (cur->right && cur->right->val > high) cur->right = cur->right->left;
            cur = cur->right;
        }
        return res;
    }
};

 /*
    108. Convert Sorted Array to Binary Search Tree
    https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
    Given an integer array nums where the elements are sorted in ascending order, convert it to a 
    height-balanced
    binary search tree.

    Example 1:
    Input: nums = [-10,-3,0,5,9]
    Output: [0,-3,9,-10,null,5]
    Explanation: [0,-10,5,null,-3,null,9] is also accepted:

    Example 2:
    Input: nums = [1,3]
    Output: [3,1]
    Explanation: [1,null,3] and [3,1] are both height-balanced BSTs.
    

    Constraints:
    1 <= nums.length <= 104
    -104 <= nums[i] <= 104
    nums is sorted in a strictly increasing order.
 */
// Recursive: O(N)
class Solution {
private:
    TreeNode* buildTree(vector<int>& nums, int l, int r) {
        if (l >= r) return nullptr;
        int middle = l + (r - l) / 2;
        TreeNode* res = new TreeNode(nums[middle]);
        res->left = buildTree(nums, l, middle);
        res->right = buildTree(nums, middle + 1, r);
        return res;
    }
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return buildTree(nums, 0, nums.size());
    }
};

// Iterative
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        if (nums.size() == 0) return nullptr;

        TreeNode* root = new TreeNode(0);   
        queue<TreeNode*> nodeQue;           
        queue<int> leftQue;                 
        queue<int> rightQue;                
        nodeQue.push(root);                
        leftQue.push(0);                    
        rightQue.push(nums.size() - 1);     

        while (!nodeQue.empty()) {
            TreeNode* curNode = nodeQue.front();
            nodeQue.pop();
            int left = leftQue.front(); leftQue.pop();
            int right = rightQue.front(); rightQue.pop();
            int mid = left + ((right - left) / 2);

            curNode->val = nums[mid];      

            if (left <= mid - 1) {          
                curNode->left = new TreeNode(0);
                nodeQue.push(curNode->left);
                leftQue.push(left);
                rightQue.push(mid - 1);
            }

            if (right >= mid + 1) {         
                curNode->right = new TreeNode(0);
                nodeQue.push(curNode->right);
                leftQue.push(mid + 1);
                rightQue.push(right);
            }
        }
        return root;
    }
};

 /*
    538. Convert BST to Greater Tree
    https://leetcode.com/problems/convert-bst-to-greater-tree/
    Given the root of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus the sum of all keys greater than the original key in BST.

    As a reminder, a binary search tree is a tree that satisfies these constraints:

    The left subtree of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.
    

    Example 1:
    Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
    Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]

    Example 2:
    Input: root = [0,null,1]
    Output: [1,null,1]
    
    Constraints:
    The number of nodes in the tree is in the range [0, 104].
    -10^4 <= Node.val <= 10^4
    All the values in the tree are unique.
    root is guaranteed to be a valid binary search tree.
 */
// Recursive: O(N)
class Solution {
private:
    int preSum = 0;
    void calculateSum(TreeNode* node) {
        if (!node) return;
        calculateSum (node->right);
        preSum += node->val;
        // int res = preSum + node->val;
        node->val = preSum;
        calculateSum(node->left);
    }
public:
    TreeNode* convertBST(TreeNode* root) {
        calculateSum(root);
        return root;
    }
};

// Iterative
class Solution {
public:
    TreeNode* convertBST(TreeNode* root) {
        stack<TreeNode*> st;
        TreeNode* cur = root;
        int preSum = 0;
        
        while (cur || !st.empty()) {
            while (cur) {
                st.push(cur);
                cur = cur->right;
            }

            // Handle root
            cur = st.top();
            st.pop();
            cout << cur->val << " " << preSum << endl;
            cur->val = cur->val + preSum;
            preSum = cur->val;
            // Move the pointer to the left, note don't push it to the stack.
            cur = cur->left;
        }
        return root;
    }
};