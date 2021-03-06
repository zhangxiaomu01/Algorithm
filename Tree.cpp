 #include<windows.h>
#include<algorithm>
#include<vector>
#include<array>
#include<cmath>
#include<random>
#include<sstream>
#include<unordered_map>
#include<numeric>
#include<iterator>
#include<iostream>
using namespace std;

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

//Another version
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        if(!root) return res;
        stack<TreeNode*> st;
        TreeNode* pre = nullptr, *cur = root;
        while(cur || !st.empty()){
            while(cur){
                st.push(cur);
                res.push_back(cur->val);
                cur = cur->left;
            }
            cur = st.top();
            if(cur->right && cur->right != pre){
                cur = cur->right;
                //We need to continue for the next loop in order
                //to get the most left nodes to stack
                continue;
            }
            st.pop(); 
            pre = cur;
            cur = nullptr;
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
        //Maintain a variable for current value. For inorder traversal, 
        //we need to push left nodes to our stack, then add the value to res.
        TreeNode* cur = root;
        while(cur || !st.empty()){
            if(cur!= nullptr){
                st.push(cur);
                cur = cur->left;
            }
            else{
                //We need to update cur. because current it's nullptr
                cur = st.top();
                res.push_back(cur->val);
                st.pop();
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

//A generalized in order traversal
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        if(!root) return res;
        stack<TreeNode*> st;
        TreeNode* pre = nullptr, *cur = root;
        while(cur || !st.empty()){
            while(cur){
                st.push(cur);
                cur = cur->left;
            }
            cur = st.top();
            // pop immediately to avoid repetitively visit
            st.pop(); 
            res.push_back(cur->val);
            if(cur->right && cur->right != pre){
                cur = cur->right;
                //We need to continue for the next loop in order
                //to get the most left nodes to stack
                continue;
            }
            pre = cur;
            cur = nullptr;
        }
        return res;
    }
};

//145. Binary Tree Postorder Traversal
//https://leetcode.com/problems/binary-tree-postorder-traversal/
/*
Basically, we juest reverse the order of preorder traversal. Note in preorder, 
we have root->left->right. It's easy to calculate. However, we can first calculate 
root->right->left. Then reverse the order of the output.
 */
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        if(!root) return res;
        stack<TreeNode*> st;
        st.push(root);
        TreeNode* ptr;
        while(!st.empty()){
            ptr = st.top();
            st.pop();
            if(ptr){
                res.push_back(ptr->val);
                st.push(ptr->left);
                st.push(ptr->right);
            }
        }
        reverse(res.begin(), res.end());
        return res;
    }
};

class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> nodes;
        stack<TreeNode*> todo;
        TreeNode* last = NULL;
        while (root || !todo.empty()) {
            if (root) {
                todo.push(root);
                root = root -> left;
            } else {
                TreeNode* node = todo.top();
                if (node -> right && last != node -> right) {
                    root = node -> right;
                } else {
                    nodes.push_back(node -> val);
                    last = node;
                    todo.pop();
                }
            }
        }
        return nodes;
    }
};

class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> nodes;
        stack<TreeNode*> todo;
        TreeNode* last = NULL;
        while (root || !todo.empty()) {
            while (root) {
                todo.push(root);
                root = root -> left;
            } 
            
            TreeNode* node = todo.top();
            //can use if(...) {root = node->right; continue;}
            if (node -> right && last != node -> right) {
                root = node -> right;
            }
            else{
                nodes.push_back(node -> val);
                last = node;
                todo.pop();
            }
        }
        return nodes;
    }
};

//102. Binary Tree Level Order Traversal
//https://leetcode.com/problems/binary-tree-level-order-traversal/
/*
BFS + DFS: DFS solution is tricky
 */
//BFS
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        //We can use the length of the tree to indicate the elements in this level
        //A common trick, remember
        queue<TreeNode*> Q;
        Q.push(root);
        while(!Q.empty()){
            int len = Q.size();
            vector<int> tempStack;
            for(int i = 0; i < len; i++){
                TreeNode* cur = Q.front();
                Q.pop();
                tempStack.push_back(cur->val);
                if(cur->left) Q.push(cur->left);
                if(cur->right) Q.push(cur->right);
            }
            res.push_back(tempStack);
        }
        return res;
    }
};

//DFS
class Solution {
private:
    void dfs(vector<vector<int>>& res, TreeNode* node, int level){
        int len = res.size();
        //How to maintain the current layer list is critical here.
        if(res.empty() || len < level + 1)
            res.push_back(vector<int>());
        res[level].push_back(node->val);
        if(node->left) dfs(res, node->left, level+1);
        if(node->right) dfs(res, node->right, level + 1);
    }
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        dfs(res, root, 0);
        return res;
    }
};

//100. Same Tree
//https://leetcode.com/problems/same-tree/
/*
Simple recursion problem.
 */
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(!p)
            return !q;
        if(!q)
            return !p;
        return p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};

//Iterative Version
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        stack<TreeNode*> pStack, qStack;
        if(p == NULL || q ==NULL) return p == q;
        pStack.push(p);
        qStack.push(q);
        while(!pStack.empty() && !qStack.empty()){
            TreeNode* pv = pStack.top();
            TreeNode* qv = qStack.top();
            pStack.pop();
            qStack.pop();
            if(pv->val != qv->val) return false;
            if(pv->left != NULL) pStack.push(pv->left);
            if(qv->left != NULL) qStack.push(qv->left);
            if(pStack.size() != qStack.size()) return false;
            if(pv->right != NULL) pStack.push(pv->right);
            if(qv->right != NULL) qStack.push(qv->right);
            if(pStack.size() != qStack.size()) return false;
                      
        }
        return pStack.size() == qStack.size();
    }
};


//101. Symmetric Tree
//https://leetcode.com/problems/symmetric-tree/
/*
Recursion + Iteration: Interesting problem
 */
class Solution {
private:
    bool isValid(TreeNode* l, TreeNode* r){
        if(l == nullptr && r == nullptr) return true;
        else if( l == nullptr || r == nullptr) return false;
        else return l->val == r->val && isValid(l->left, r->right) && isValid(l->right, r->left);
    }
public:
    bool isSymmetric(TreeNode* root) {
        if(root == nullptr) return true;
        return isValid(root->left, root->right);
    }
};

//Iterative Version
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(!root) return true;
        queue<TreeNode*> Q;
        Q.push(root->left);
        Q.push(root->right);
        while(!Q.empty()){
            TreeNode* l = Q.front();
            Q.pop();
            TreeNode* r = Q.front();
            Q.pop();
            if((!l && r) || (!r && l)) return false;
            if(l != nullptr && r != nullptr){
                if(l->val != r->val) return false;
                Q.push(l->left);
                Q.push(r->right);
                Q.push(l->right);
                Q.push(r->left);
            }
        }
        return true;
    }
};

//226. Invert Binary Tree
//https://leetcode.com/problems/invert-binary-tree/
/*
Note we are actually swapping the left and right children level by level. 
This holds true for both recursive and iterative version.
 */
class Solution {
public:
    //Root 
    TreeNode* invertTree(TreeNode* root) {
        if(!root) return root;
        TreeNode* tempNode = root->left;
        root->left = invertTree(root->right);
        root->right = invertTree(tempNode);
        return root;
    }
};
//Iterative Version
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(!root) return root;
        
        stack<TreeNode*> st;
        st.push(root);
        while(!st.empty()){
            TreeNode* tempNode = st.top();
            st.pop();
            //Swap node
            TreeNode* tempLeft = tempNode->left;
            tempNode->left = tempNode->right;
            tempNode->right = tempLeft;
            //Push next level, we discard null node
            if(tempNode->left) st.push(tempNode->left);
            if(tempNode->right) st.push(tempNode->right);
        }
        return root;
    }
};


//257. Binary Tree Paths
//https://leetcode.com/problems/binary-tree-paths/
//DFS version
class Solution {
private:
    void dfs(vector<string>& res, TreeNode* node, string s){
        if(!node) return;
        s += to_string(node->val);
        if(!node->left && !node->right) {
            res.push_back(s);
            return;
        }
        if(node->left) {
            dfs(res, node->left, s + "->");
        }
        if(node->right){
            dfs(res, node->right, s + "->");
        }
    }
public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> res;
        if(!root) return res;
        dfs(res, root, "");
        return res;
    }
};

//Iterative Version
class Solution {
public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> res;
        if(!root) return res;
        stack<string> path;
        stack<TreeNode*> st;
        st.push(root);
        path.push(to_string(root->val));
        while(!st.empty()){
            TreeNode* node = st.top();
            st.pop();
            string s = path.top();
            path.pop();
            if(!node->left && !node->right) {
                res.push_back(s);
                continue;
            }
            if(node->right){
                st.push(node->right);
                path.push(s + "->" + to_string(node->right->val));
            }
            if(node->left){
                st.push(node->left);
                path.push(s + "->" + to_string(node->left->val));
            }
            
        }
        return res;
    }
};


//112. Path Sum
//https://leetcode.com/problems/path-sum/
//Iterative version
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        if(!root) return false;
        stack<TreeNode*> st;
        st.push(root);
        stack<int> sumSt;
        sumSt.push(root->val);
        while(!st.empty()){
            TreeNode* node = st.top();
            st.pop();
            int curSum = sumSt.top();
            sumSt.pop();
            if(!node->left && !node->right){
                if(sum == curSum)
                    return true;
            }
            if(node->left){
                st.push(node->left);
                sumSt.push(curSum + node->left->val);
            }
            if(node->right){
                st.push(node->right);
                sumSt.push(curSum + node->right->val);
            }
        }
        return false;
        
    }
};
//Recursive version
class Solution {
private:
    bool dfs(TreeNode* node, int sum, int cur){
        if(sum == cur + node->val && !node->left && !node->right)
            return true;
        bool res = false;
        if(node->left) res = res || dfs(node->left, sum, cur + node->val);
        if(res) return true;
        if(node->right) res = res || dfs(node->right, sum, cur + node->val);
        return res;
    }
public:
    bool hasPathSum(TreeNode* root, int sum) {
        if(!root) return false;
        return dfs(root, sum, 0);
    }
};

//113. Path Sum II
//https://leetcode.com/problems/path-sum-ii/
/*
Tree traversal problem: Iterative and recursive
 */
//Recursive version
class Solution {
private:
    void dfs(vector<vector<int>>& res, vector<int> V, const TreeNode* node, int curSum, int sum){
        curSum += node->val;
        V.push_back(node->val);
        if(!node->left && !node->right && sum == curSum){
            res.push_back(V);
            return;
        }
        if(!node->left && !node->right){
            V.pop_back();
            return;
        }
        if(node->left) dfs(res, V, node->left, curSum, sum);
        if(node->right) dfs(res, V, node->right, curSum, sum);
    }
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int>> res;
        if(!root) return res;
        vector<int> tempSequence;
        dfs(res, tempSequence, root, 0, sum);
        return res;
    }
};

//Iterative version: very effective
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int>> res;
        if(!root) return res;
        stack<TreeNode*> st;
        vector<int> path;
        //Using two flags to keep track of current node and previous node
        TreeNode* cur = root;
        TreeNode* pre = nullptr;
        int curNum = 0;
        while(cur!=nullptr || !st.empty()){
            while(cur){
                st.push(cur);
                path.push_back(cur->val);
                curNum += cur->val;
                cur = cur->left;
            }
            cur = st.top(); //We step back to the previous node
            if(cur->right != nullptr && cur->right != pre){
                //if we haven't visited the right node, we need to visit it
                cur = cur->right;
                continue;
            }
            if(cur->right == nullptr && cur->left == nullptr && sum == curNum){
                res.push_back(path);
            }
            
            st.pop();
            pre = cur;
            curNum -= cur->val;
            path.pop_back();
            cur = nullptr;
        }
       return res; 
    }
};

//129. Sum Root to Leaf Numbers
//https://leetcode.com/problems/sum-root-to-leaf-numbers/
/*
DFS or BFS
 */
//DFS
class Solution {
private:
    void dfs(TreeNode* node, string s, int& sum){
        s.push_back(node->val + '0');
        if(!node->left && !node->right){
            sum += stoi(s);
        }
        if(node->left) dfs(node->left, s, sum);
        if(node->right) dfs(node->right, s, sum);
    }
public:
    int sumNumbers(TreeNode* root) {
        if(!root) return 0;
        int totalSum = 0;
        dfs(root, "", totalSum);
        return totalSum;
    }
};
//
class Solution {
public:
    int sumNumbers(TreeNode* root) {
        stack<TreeNode*> st;
        if(!root) return 0;
        string s = "";
        int totalSum = 0;
        TreeNode* cur = root;
        TreeNode* pre = nullptr;
        while(cur || !st.empty()){
            while(cur){
                st.push(cur);
                s.push_back(cur->val + '0');
                cur = cur->left;
            }
            cur = st.top();
            if(cur->right && cur->right != pre){
                cur = cur->right;
                continue;
            }
            //We need to add !cur->left, because when we traceback, we still need to guarantee that cur is a leaf
            if(!cur->right && !cur->left){
                totalSum += stoi(s);
            }
            pre = cur;
            //We must set cur to nullptr, or we will in a loop
            cur = nullptr;
            st.pop();
            s.pop_back();
        }
        return totalSum;
    }
};

//Memory Efficient DFS: We do not need to save the whole string and covert it to integer
class Solution {
public:
    int sumNumbers(TreeNode* root) {
        if(root == NULL) return 0;
        return rec(root, 0);
        
    }
    int rec(TreeNode* root, int currentSum){
        if(root == NULL){
            return 0;
        }
        //Save the result on the fly
        int temp = currentSum*10+root->val;
        if(root->left == NULL && root->right == NULL) return temp;
        return rec(root->left, temp) + rec(root->right, temp);
    }
};

//An interesting idea: save the result on each node...
class Solution {
public:
    int sumNumbers(TreeNode* root) {
        if(root == NULL) return 0;
        stack<TreeNode*> st;
        st.push(root);
        int sum = 0;
        while(!st.empty()){
            TreeNode* node = st.top();
            st.pop();
            
            if(node->right != NULL){
                node->right->val = node->val*10 + node->right->val;                
                st.push(node->right);
            }
            if(node->left != NULL){
                node->left->val = node->val*10 + node->left->val;
                st.push(node->left);
            }
            
            if(node->left == NULL && node->right == NULL)
                sum += node->val;
        }        
        return sum;        
    }
};

//111. Minimum Depth of Binary Tree
//https://leetcode.com/problems/minimum-depth-of-binary-tree/
//DFS
class Solution {
public:
    int minDepth(TreeNode* root) {
        if(root == nullptr) return 0;
        if(!root->left) return minDepth(root->right)+1;
        if(!root->right) return minDepth(root->left) + 1;
        return min(minDepth(root->left), minDepth(root->right))+1;
    }
};
//BFS
class Solution {
public:
    int minDepth(TreeNode* root) {
        if(!root) return 0;
        queue<TreeNode*> Q;
        int level = 0;
        Q.push(root);
        while(!Q.empty()){
            int len = Q.size();
            level ++;
            for(int i = 0; i < len; i++){
                TreeNode* node = Q.front();
                Q.pop();
                if(node->left)
                    Q.push(node->left);
                if(node->right)
                    Q.push(node->right);
                if(!node->left && !node->right)
                    return level;
            }
        }
        return level;
    }
};


//104. Maximum Depth of Binary Tree
//https://leetcode.com/problems/maximum-depth-of-binary-tree/
//DFS
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        int maxLeft = maxDepth(root->left);
        int maxRight = maxDepth(root->right);
        return 1 + max(maxLeft, maxRight);
    }
};

//BFS
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        queue<TreeNode*> Q;
        Q.push(root);
        int level = 0;
        while(!Q.empty()){
            int len = Q.size();
            level++;
            for(int i = 0; i < len; i++){
                TreeNode* node = Q.front();
                Q.pop();
                if(node->left) Q.push(node->left);
                if(node->right)Q.push(node->right);
            }
        }
        return level;
    }
};

//110. Balanced Binary Tree
//https://leetcode.com/problems/balanced-binary-tree/
/* DFS */
class Solution {
    int rec(TreeNode* node){
        if(!node) return 1;
        int l_height = rec(node->left);
        if(l_height == -1) return -1;
        int r_height = rec(node->right);
        if(r_height == -1) return -1;
        
        if(abs(l_height - r_height) > 1) return -1;
        else return max(l_height, r_height) + 1;
    }
public:
    bool isBalanced(TreeNode* root) {
        if(!root) return true;
        return rec(root) != -1;        
    }
};




//124. Binary Tree Maximum Path Sum
//https://leetcode.com/problems/binary-tree-maximum-path-sum/
//A tricky problem. The insight how to form the recursion formula is critical.
class Solution {
private:
    int maxVal = numeric_limits<int>::min();
    int rec(TreeNode* node){
        if(!node) return 0;
        int left = max(0, rec(node->left));
        int right = max(0, rec(node->right));
        maxVal = max(maxVal, left + right + node->val);
        //Note we only need to return the maximum path of the two, 
        //then we can from the valid path
        return max(left, right) + node->val;
    }
public:
    int maxPathSum(TreeNode* root) {
        rec(root);
        return maxVal;
    }
};

//Iterative: Post order traversal. Not memory efficient, not easy to get the insight
class Solution {
private:
    deque<TreeNode*>* buildTopology(TreeNode* node){
        stack<TreeNode*> st;
        st.push(node);
        deque<TreeNode*>* Qptr = new deque<TreeNode*>();
        while(!st.empty()){
            TreeNode* tempNode = st.top();
            st.pop();
            Qptr->push_front(tempNode);
            if(tempNode->left) st.push(tempNode->left);
            if(tempNode->right) st.push(tempNode->right);
        }
        return Qptr;
    }
public:
    int maxPathSum(TreeNode* root) {
        if(!root) return 0;
        //Save intermedia result, the leaft node will have 0 value
        unordered_map<TreeNode*, int> dict;
        dict.insert({nullptr, 0});
        int maxValue = numeric_limits<int>::min();
        deque<TreeNode*>* Qptr = buildTopology(root);
        for(auto it = Qptr->begin(); it != Qptr->end(); it++){
            //Since it's post order traversal, we always start with null node
            int leftMax = max(dict[(*it)->left], 0);
            int rightMax = max(dict[(*it)->right], 0);
            //Note minimum possible value of leftMax and rightMax is 0
            maxValue = max(maxValue, leftMax + rightMax + (*it)->val);
            //Save the current maximum value when this node is included in the path
            dict[(*it)] = max(leftMax, rightMax) + (*it)->val;
            
        }
        return maxValue;
    }
};


//337. House Robber III
//https://leetcode.com/problems/house-robber-iii/
/*In general, this problem is DP + Tree Traversal. For each node, we need to consider two conditions, whether we rob this node, or we skip it. 
We utilize a pair data structure to store the maximum profit we can get if we do not rob this house and the maximum profit if we rob this house. 
We will consider the whole tree as our DP table and build this table from bottom (leaf -> root). 
Then the general idea will be we do postorder traversal, and build the tree, and in the end, compare the maximum profit of robbing root and not
robbing root.
 */
class Solution {
private:
    pair<int, int> doRob(TreeNode* node){
        if(!node) return make_pair(0, 0);
        auto l = doRob(node->left);
        auto r = doRob(node->right);
        int robNodeProfit = l.first + r.first + node->val;
        int nRobNodeProfit = max(l.first, l.second) + max(r.first, r.second);
        return make_pair(nRobNodeProfit, robNodeProfit);
    }
public:
    int rob(TreeNode* root) {
        if(!root) return 0;
        auto robHouse = doRob(root);
        return max(robHouse.first, robHouse.second);
    }
};

//Itrative version, first build the post-order traversal queue. Then we can replicate the recursive process.
class Solution {
private:
    deque<TreeNode*>* buildQueue(TreeNode* node){
        deque<TreeNode*>* Qptr = new deque<TreeNode*>();
        stack<TreeNode*> st;
        st.push(node);
        while(!st.empty()){
            TreeNode* tempNode = st.top();
            st.pop();
            Qptr->push_front(tempNode);
            if(tempNode->right) st.push(tempNode->right);
            if(tempNode->left) st.push(tempNode->left);
        }
        return Qptr;
    }
    
public:
    int rob(TreeNode* root) {
        if(!root) return 0;
        deque<TreeNode*>* Qptr = buildQueue(root);
        unordered_map<TreeNode*, pair<int, int>> dict;
        //The first element of pair means without robbing the house, the potential maximum profit; second means we robber the house
        dict.insert({nullptr, make_pair(0, 0)});
        for(auto it = Qptr->begin(); it != Qptr->end(); it++){
            //The max profit we can get if we rob the node
            int robNode = dict[(*it)->left].first + dict[(*it)->right].first + (*it)->val;
            //The max profit we can get if we do not rob the node
            int robnNode = max(dict[(*it)->left].first, dict[(*it)->left].second) + max(dict[(*it)->right].first, dict[(*it)->right].second);
            //Save the result to hash table
            dict[(*it)] = make_pair(robnNode, robNode);
        }
        return max(dict[root].first, dict[root].second);
    }
};

//107. Binary Tree Level Order Traversal II
//https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
//Iterative BFS
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        queue<TreeNode*> Q;
        Q.push(root);
        while(!Q.empty()){
            int len = Q.size();
            vector<int> tempVec;
            for(int i = 0; i < len; i++){
                TreeNode* tempNode = Q.front();
                Q.pop();
                tempVec.push_back(tempNode->val);
                if(tempNode->left) Q.push(tempNode->left);
                if(tempNode->right) Q.push(tempNode->right);
            }
            res.push_back(tempVec);
        }
        reverse(res.begin(), res.end());
        return res;
    }
};

//DFS
class Solution {
private:
    void dfs(vector<vector<int>>& res, TreeNode* node, int level){
        if(!node) return;
        if(res.size() <= level)
            res.push_back(vector<int>());
        res[level].push_back(node->val);
        if(node->left) dfs(res, node->left, level+1);
        if(node->right) dfs(res, node->right, level+1);
    }
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        dfs(res, root, 0);
        reverse(res.begin(), res.end());
        return res;
    }
};

//103. Binary Tree Zigzag Level Order Traversal
//https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
//Iterative BFS
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if(root == NULL) return res;
        queue<TreeNode*> tree;
        tree.push(root);
        bool flag = true;
        
        while(!tree.empty()){
            int count = tree.size();
            vector<int> level(count);
            for(int i = 0; i < count; i++){
                TreeNode* temp = tree.front();
                tree.pop();
                int index = flag? i: count - i - 1;
                level[index] = temp->val;
                
                if(temp->left!=NULL){
                    tree.push(temp->left);
                }
                if(temp->right!= NULL)
                    tree.push(temp->right);
            }
            res.push_back(level);
            flag = !flag;
        }
        return res;
    }
};

//Recursive BFS
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<list<int>> res;
        vector<vector<int>> f_res;
        rec(root, res, 0);
        for(int i = 0; i < res.size(); i++)
        {
            f_res.push_back(vector<int>());
            int list_s = res[i].size();
            for(int j = 0; j < list_s; j++){
                f_res[i].push_back(res[i].front());
                res[i].pop_front();
            }
        }
        return f_res;
    }   
    void rec(TreeNode* root, vector<list<int>> &res, int level){
        if(root == NULL) return;
        
        if(res.empty() || level > res.size() - 1)
            res.push_back(list<int>());
        if(level%2 == 0){
            res[level].push_back(root->val);  
        }
        else
            res[level].push_front(root->val);
        
        rec(root->left, res, level+1);
        rec(root->right, res, level+1);
    }
};

//199. Binary Tree Right Side View
//https://leetcode.com/problems/binary-tree-right-side-view/
/* Iterative BFS */
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res;
        if(!root) return res;
        queue<TreeNode*> Q;
        Q.push(root);
        while(!Q.empty()){
            int len = Q.size();
            for(int i = 0; i < len; i++){
                TreeNode* node = Q.front();
                Q.pop();
                if(i == len-1){
                    res.push_back(node->val);
                }
                if(node->left) Q.push(node->left);
                if(node->right) Q.push(node->right);
            }
        }
        return res;
    }
};
/* Recursive version */
class Solution {
private:
    void BFS(vector<int>& res, TreeNode* node, int level){
        if(!node) return;
        if(res.size() <= level){
            res.push_back(node->val);
        }else{
            //We save the final element of this level to our result vector
            res[level] = node->val;
        }
        BFS(res, node->left, level+1);
        BFS(res, node->right, level+1);
        
    }
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res;
        if(!root) return res;
        BFS(res, root, 0);
        return res;
    }
};

//98. Validate Binary Search Tree
//https://leetcode.com/problems/validate-binary-search-tree/
/* In order traversal */
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        if(!root) return true;
        TreeNode* cur = root;
        TreeNode* pre = nullptr;
        stack<TreeNode*> st;
        while(cur || !st.empty()){
            while(cur){
                st.push(cur);
                cur = cur->left;
            }
            cur = st.top();
            st.pop();
            //Note pre always points to the left child, we will check each pair from leaf to root
            if(pre != nullptr && pre->val >= cur->val) return false;
            pre = cur;
            cur = cur->right;
        }
        return true;
        
    }
};
/* In order traversal */
class Solution {
private:
    bool inOrder(TreeNode* node, TreeNode* &pre){
        if(!node) return true;
        bool lVal = inOrder(node->left, pre);
        if(pre != nullptr && pre->val >= node->val)
            return false;
        pre = node;
        bool rVal = inOrder(node->right, pre);
        return lVal && rVal;
        
    }
public:
    bool isValidBST(TreeNode* root) {
        TreeNode* pre = nullptr;
        return inOrder(root, pre);
    }
};
/* Beautiful recursive implementation */
class Solution {
private:
    bool CheckValid(TreeNode* root, TreeNode* minVal, TreeNode* maxVal){
        if(!root) return true;
        if(minVal && minVal->val >= root->val || maxVal && maxVal->val <= root->val){
            return false;
        }
        //Using minVal and maxVal to set the range...
        return CheckValid(root->left, minVal, root) && CheckValid(root->right, root, maxVal);
    }
public:
    bool isValidBST(TreeNode* root) {
        if(!root) return true;
        return CheckValid(root, nullptr, nullptr);
    }
};


//235. Lowest Common Ancestor of a Binary Search Tree
//https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
/* Recursive version */
//General idea is to find the split point of the sub BST which includes both p 
//and q
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return root;
        int rootVal = root->val;
        int pVal = p->val;
        int qVal = q->val;
        if(pVal > rootVal && qVal > rootVal)
            return lowestCommonAncestor(root->right, p, q);
        else if(pVal < rootVal && qVal < rootVal)
            return lowestCommonAncestor(root->left, p, q);
        else
            return root;
    }
};

/* Iterative version, exactly the same */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return root;
        TreeNode* node = root;
        while(node){
            if(p->val > node->val && q->val > node->val)
                node = node->right;
            else if(p->val < node->val && q->val < node->val)
                node = node->left;
            else return node;
        }
        return node;
    }
};

//236. Lowest Common Ancestor of a Binary Tree
//https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
/* A more generalized approach, can also be used to solve problem 235 */
//It's postorder traversal. During the traversal, we add flag to notify whether p and q are in the sub tree
//It's elegant!!!
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return root;
        
        if(root == p || root == q)
            return root;
        
        root->left = lowestCommonAncestor(root->left, p, q);
        root->right = lowestCommonAncestor(root->right, p, q);
        
        if(root->left && root->right)
            return root;
        if(root->left)
            return root->left;
        if(root->right) 
            return root->right;
        return nullptr;
    }
};

/*
Iterative version: still post order traversal. We first need to find the ancestors of all chilren.
Then we find the ancestor of p, after that, we find the common ancestor of p and q.
It's not memory efficiency...
*/
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

//108. Convert Sorted Array to Binary Search Tree
//https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
//Recursive version: we need to build the left child tree and right child tree respectively.
//The recursive solution is not complex. A little bit hard to get to the right direction
class Solution {
private:
    TreeNode* buildTree(vector<int>& nums, int start, int end){
        if(start >= end) return nullptr;
        int mid = start + (end - start) /2;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = buildTree(nums, start, mid);
        root->right = buildTree(nums, mid+1, end);
        return root;
    }
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        if(nums.empty()) return nullptr;
        int len = nums.size();
        return buildTree(nums, 0, len);
    }
};

//Iterative version
//Iterative version is not easy... Be careful about the binary search
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        if(nums.empty()) return nullptr;
        queue<TreeNode*> Q;
        queue<pair<int, int>> nodeBoundryQ;
        //Note we need to make right to be nums.size() in order to make the code work
        int right = nums.size(), left = 0;
        int mid = left + (right - left) / 2;
        
        TreeNode* root = new TreeNode(nums[mid]);
        Q.push(root);
        nodeBoundryQ.push(make_pair(left, mid));
        nodeBoundryQ.push(make_pair(mid+1, right));
        
        while(!Q.empty()){
            TreeNode* node = Q.front();
            Q.pop();
            
            //Create left sub tree
            pair<int, int> lrNode = nodeBoundryQ.front();
            nodeBoundryQ.pop();
            if(lrNode.first >= lrNode.second)
                node->left = nullptr;
            else{
                mid = lrNode.first + (lrNode.second - lrNode.first) / 2;
                TreeNode* lNode = new TreeNode(nums[mid]);
                node->left = lNode;
                Q.push(lNode);
                nodeBoundryQ.push(make_pair(lrNode.first, mid));
                nodeBoundryQ.push(make_pair(mid+1, lrNode.second));
            }
            
            //Create right sub tree
            lrNode = nodeBoundryQ.front();
            nodeBoundryQ.pop();
            if(lrNode.first >= lrNode.second)
                node->right = nullptr;
            else{
                mid = lrNode.first + (lrNode.second - lrNode.first) / 2;
                TreeNode* rNode = new TreeNode(nums[mid]);
                node->right = rNode;
                Q.push(rNode);
                nodeBoundryQ.push(make_pair(lrNode.first, mid));
                nodeBoundryQ.push(make_pair(mid+1, lrNode.second));
            }
            
        }
        return root;
    }
};

//109. Convert Sorted List to Binary Search Tree
//https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/
/* In order traversal. How to move forward the list node is tricky!! */
//Convert list to array, then we can use the same technique to implement iterative version just like problem 108
class Solution {
private:
    int FindLength(ListNode* node){
        int len = 0;
        while(node){
            len++;
            node = node->next;
        }
        return len;
    }
    //Note we need to pass the node as reference
    TreeNode* buildTree(ListNode* &node, int l, int r){
        if(l >= r) return nullptr;
        int mid = l + (r - l) / 2;
        TreeNode* left = buildTree(node, l, mid);
        
        //Create new tree node
        TreeNode* root = new TreeNode(node->val);
        root->left = left;
        node = node->next;
        
        root->right = buildTree(node, mid+1, r);
        return root;
    }
    
public:
    TreeNode* sortedListToBST(ListNode* head) {
        int len = FindLength(head);
        return buildTree(head, 0, len);
    }
};


//173. Binary Search Tree Iterator
//https://leetcode.com/problems/binary-search-tree-iterator/
/* In order traversal, not that hard! */
class BSTIterator {
private:
    TreeNode* m_nextNodePtr;
    stack<TreeNode*> m_st;
public:
    BSTIterator(TreeNode* root) {
        m_nextNodePtr = root;
        while(m_nextNodePtr){
            m_st.push(m_nextNodePtr);
            m_nextNodePtr = m_nextNodePtr->left;
        }
        
    }
    
    /** @return the next smallest number */
    int next() {
        m_nextNodePtr = m_st.top();
        m_st.pop();
        int val = m_nextNodePtr->val;
        m_nextNodePtr = m_nextNodePtr->right;
        while(m_nextNodePtr){
            m_st.push(m_nextNodePtr);
            m_nextNodePtr = m_nextNodePtr->left;
        }
        //m_nextNodePtr = m_st.top();
        //m_st.pop();
        return val;
    }
    
    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !m_st.empty();
    }
};

//230. Kth Smallest Element in a BST
//https://leetcode.com/problems/kth-smallest-element-in-a-bst/
/* Iterative in order traversal */
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        TreeNode* ptr = root;
        stack<TreeNode*> st;
        int count = 0;
        while(ptr || !st.empty()){
            if(ptr){
                st.push(ptr);
                ptr = ptr->left;
            }else{
                ptr = st.top();
                st.pop();
                count ++;
                if(count == k) return ptr->val;
                if(ptr->right)
                    ptr = ptr->right;
                else ptr = nullptr;
            }
        }
        return -1;
    }
};

/* Recursive implementation */
class Solution {
private:
    void inOrder(TreeNode* node, int &k, int &res){
        if(!node) return;
        inOrder(node->left, k, res);
        //We find the right answer.
        if(k == 0) return;
        res = node->val;
        inOrder(node->right, --k, res);
    }
public:
    int kthSmallest(TreeNode* root, int k) {
        int res = 0;
        inOrder(root, k, res);
        return res;
    }
};


//297. Serialize and Deserialize Binary Tree
//https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
class Codec {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string res;
        if(!root) return res;
        queue<TreeNode*> Q;
        Q.push(root);
        while(!Q.empty()){
            int len = Q.size();
            for(int i = 0; i < len; i++){
                TreeNode* node = Q.front();
                Q.pop();
                if(!node){
                    res.push_back('n');
                    res.push_back(',');
                }
                else{
                    res.append(to_string(node->val));
                    res.push_back(',');
                    Q.push(node->left);
                    Q.push(node->right);
                }
            }
        }
        return res;
    }
    
    void buildStrVector(const string& s, vector<string>& res){
        //vector<string> res;
        string tempS;
        for(int i = 0; i < s.size(); ++i){
            if(s[i] != ','){
                tempS.push_back(s[i]);
            }else{
                res.push_back(tempS);
                tempS.clear();
            }
        }
    }
    
    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if(data.empty()) return nullptr;
        
        vector<string> res;
        buildStrVector(data, res);
        
        queue<string> nodeQ;
        queue<TreeNode*> Q;
        TreeNode* root = new TreeNode(stoi(res[0]));
        Q.push(root);
        nodeQ.push(res[0]);
        int k = 0;
        int len = res.size();
        
        while(!Q.empty()){
            int level = Q.size();
            for(int i = 0; i < level; i++){
                TreeNode* node = Q.front();
                Q.pop();
                //left node
                k++;
                if(res[k] != "n"){
                    TreeNode* tempNode = new TreeNode(stoi(res[k]));
                    node->left = tempNode;
                    Q.push(tempNode);
                }
                //right node
                k++;
                if(res[k] != "n"){
                    TreeNode* tempNode = new TreeNode(stoi(res[k]));
                    node->right = tempNode;
                    Q.push(tempNode);
                }
            }
        }
        
        return root;
    }
};

//99. Recover Binary Search Tree
//https://leetcode.com/problems/recover-binary-search-tree/
/* In order traversal, we have a pre pointer to indicate the previous node, 
and we compare this node with our current node to find out which node is swapped.
This version is O(log n) space
*/
class Solution {
private:
    TreeNode* f = nullptr, *s = nullptr;
    TreeNode* pre = new TreeNode(numeric_limits<int>::min());
    void dfs(TreeNode* node){
        if(!node) return;
        dfs(node->left);
        
        if(f == nullptr && pre->val > node->val)
            f = pre;
        //Here we must finish the traversal, to find the smallest element from the rest of the node, 
        //this node should be swapped with f node 
        if(f != nullptr && pre->val > node->val)
            s = node;
        pre = node; //update pre to be the node we have examed!
        dfs(node->right);
    }
public:
    void recoverTree(TreeNode* root) {
        TreeNode* tempPre = pre;
        dfs(root);
        if(f && s)
            swap(s->val, f->val);
        delete tempPre;
    }
};

//Iterative version
class Solution {
public:
    void recoverTree(TreeNode* root) {
        if(!root) return;
        TreeNode* pre = new TreeNode(numeric_limits<int>::min());
        TreeNode* tempPre = pre;
        TreeNode *f = nullptr, *s = nullptr;
        stack<TreeNode*> st;
        TreeNode* cur = root;
        while(cur || !st.empty()){
            if(cur){
                st.push(cur);
                cur = cur->left;
            }else{
                cur = st.top();
                st.pop();
                if(f == nullptr && pre->val > cur->val)
                    f = pre;
                if(f != nullptr && pre->val > cur->val)
                    s = cur;
                pre = cur;
                if(cur->right) cur = cur->right;
                else cur = nullptr;
            }
        }
        if(s && f)
            swap(f->val, s->val);
        delete tempPre;
    }
};


//96. Unique Binary Search Trees
//https://leetcode.com/problems/unique-binary-search-trees/
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;

    Node() {}

    Node(int _val, Node* _left, Node* _right, Node* _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};
*/
/*
It's a DP problem. We need to check every possible combination of left sub tree 
and right tree, and sum them together to get the total number. Recursive and 
iterative versions are available!
 */
//A dp problem. Note we need to recursively calculate how many possible BSTs for both left and right sub tree.
class Solution {
private:
    int rec(int n, vector<int>& memo){
        if(memo[n] != -1) return memo[n];
        //Base case
        if(n == 0) return 1;
        if(n == 1) return 1;
        
        int totalSum = 0;
        //We need to start from i = 1, or we will go to infinity loop
        //i defines the number of nodes in sub tree, it also represents the root of current tree
        for(int i = 1; i <= n; i++){
            //Left part
            int leftNum = rec(i-1, memo);
            //Right part
            int rightNum = rec(n-i, memo);
            
            if(leftNum == 0)
                totalSum += rightNum;
            else if(rightNum == 0)
                totalSum += leftNum;
            else totalSum += leftNum* rightNum;
        }
        memo[n] = totalSum;
        return memo[n];
    }
public:
    int numTrees(int n) {
        vector<int> memo(n+1, -1);
        return rec(n, memo);
    }
};

//Iterative dp. Very tricky! Note by default each entry is 0, and dp[0] is 1
//Note that j in this case, represents the root node, so j-1 means the total node 
//in the left, and i - j means the total node in the right
class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(n+1, 0);
        dp[0] = 1;
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= i; j++)
                dp[i] += dp[j-1] * dp[i - j];
        }
        return dp[n];
    }
};

//My recursive one
class Solution {
private:
    int helper(int n, vector<int>& memo){
        if(n == 0) return 1;
        if(n == 1) return 1;
        if(n == 2) return 2;
        if(memo[n] != -1) return memo[n];
        int res = 0;
        for(int i = 0; i < n/2; ++i){
            res += helper(n-1-i, memo) * helper(i, memo) * 2;
        }
        if(n % 2 == 1) {
            int temp = helper((n-1)/2, memo) ;
            res += temp * temp;
        }
        memo[n] = res;    
        return res;
    }
public:
    int numTrees(int n) {
        vector<int> memo(n+1, -1);
        return helper(n, memo);
    }
};


//95. Unique Binary Search Trees II
//https://leetcode.com/problems/unique-binary-search-trees-ii/
//Note we do not implement memorization in this approach. 
class Solution {
private:
    vector<TreeNode*> buildTree(int s, int e){
        vector<TreeNode*> res;
        //should be s > e. then we push null
        if(s > e) {
            res.push_back(nullptr);
            return res;
        }
        
        for(int i = s; i <= e; ++i){
            //Build the left tree and right tree
            vector<TreeNode*> leftSub = buildTree(s, i-1);
            vector<TreeNode*> rightSub = buildTree(i+1, e);
            
            for(TreeNode* l : leftSub){
                for(TreeNode* r : rightSub){
                    TreeNode* root = new TreeNode(i);
                    root->left = l;
                    root->right = r;
                    res.push_back(root);
                }
            }
        }
        return res;
        
    }
public:
    vector<TreeNode*> generateTrees(int n) {
        if(n == 0) return vector<TreeNode*>();
        return buildTree(1, n);
    }
};

//Optimized version, with memo. Not implemented by me. Code from others.
class Solution {
public:
    vector<TreeNode*> generateTrees(int n) {
        vector<TreeNode*> r;
        if(n==0) return r;
        unordered_map<int, vector<TreeNode*>> m;
        r = generateTrees(1, n, m);
        return r;
    }
    
    vector<TreeNode*> generateTrees(int s, int e, unordered_map<int, vector<TreeNode*>>& m)
    {
        vector<TreeNode*> r;
        if(s>e)
        {
            r.push_back(NULL);
            return r;
        }
        
        for(int i=s;i<=e;i++)
        {
            vector<TreeNode*> left;
            int pos = s*1000 + i-1;
            auto iter = m.find(pos);
            if(iter!=m.end())
            {
                left = iter->second;
            }
            else
            {
                left =  generateTrees(s, i-1, m);
                m.emplace(pos, left);
            }
            
            pos = (i+1)*1000 + e;
            iter = m.find(pos);
            vector<TreeNode*> right;
            if(iter!=m.end())
            {
                right = iter->second;
            }
            else
            {
                right = generateTrees(i+1, e, m);
                m.emplace(pos, right);
            }
            
            for(int j=0;j<left.size();j++)
            {
                for(int k=0;k<right.size();k++)
                {
                    TreeNode* n = new TreeNode(i);
                    n->left = left[j];
                    n->right= right[k];
                    
                    r.push_back(n);
                }
            }
        }
        
        return r;
    }
};


//116. Populating Next Right Pointers in Each Node
//https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
//We can do level order traversal, but this problem requires us to use constant 
//space. Iterative version is a little bit tricky. Be careful!
/* Recursive version is trivial. Not efficient!!*/
class Solution {
    void dfs(Node* lNode, Node* rNode){
        if(!lNode && !rNode) return;
        if(lNode && !rNode){
            dfs(lNode->left, lNode->right);
        }
        if(!lNode && rNode){
            dfs(rNode->left, rNode->right);
        }
        if(lNode && rNode){
            lNode->next = rNode;
            dfs(lNode->left, lNode->right);
            dfs(lNode->right, rNode->left);
            dfs(rNode->left, rNode->right);
        }
        
    }
public:
    Node* connect(Node* root) {
        if(!root) return root;
        dfs(root->left, root->right);
        return root;
    }
};

//Level order traversal!
class Solution {
public:
    Node* connect(Node* root) {
        if(!root) return root;
        Node* cur = root;
        while(cur != nullptr){
            Node* tempNode = cur;
            while(cur!=nullptr){
                if(cur->left) cur->left->next = cur->right;
                if(cur->next && cur->right) cur->right->next = cur->next->left;
                cur = cur->next;
            }
            cur = tempNode;
            cur = cur->left;
        }
        return root;
    }
};

class Solution {
private:
    void buildConnect(Node* lNode, Node* rNode){
        if(!lNode && !rNode) return;
        lNode->next = rNode;
        buildConnect(lNode->left, lNode->right);
        buildConnect(lNode->right, rNode == nullptr ? nullptr : rNode->left);

    }
public:
    Node* connect(Node* root) {
        if(!root) return root;
        if(!root->left && !root->right)
            return root;
        
        buildConnect(root->left, root->right);
        buildConnect(root->right, nullptr);
        return root;
    }
};

//117. Populating Next Right Pointers in Each Node II
//https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/
/* 
A very tricky and elegant approach by using an extra node as a ladder. 
By attaching the ladder to each level, we traverse from left to right,
and connect each node one by one. Then we move the root node from left
to right, until reaches the most right node. Then we set the root to be 
the first node in next level!
*/
class Solution {
public:
    Node* connect(Node* root) {
        if(!root) return root;
        Node* aNode = new Node(0, nullptr, nullptr, nullptr);
        Node* ladderNode = aNode;
        Node* node = root;
        while(node!=nullptr){
            ladderNode = aNode;
            while(node){
                if(node->left){
                    ladderNode->next = node->left;
                    ladderNode = ladderNode->next;
                }
                if(node->right){
                    ladderNode->next = node->right;
                    ladderNode = ladderNode->next;
                }
                node = node->next;
            }
            node = aNode->next; // point to the first node in the next level!
            aNode->next = nullptr;
        }
        delete aNode;
        return root;
    }
};

//A more general approach using BFS
class Solution {
public:
    Node* connect(Node* root) {
        if(!root) return root;
        Node* node = root;
        Node* pre = nullptr, *head = nullptr;
        while(node){
            while(node){
                if(node->left) {
                    if(pre != nullptr) pre->next = node->left;
                    else head = node->left; //Update the first node in next level
                    pre = node->left;
                }
                if(node->right){
                    if(pre!=nullptr) pre->next = node->right;
                    else head = node->right; //Update the first node in next level 
                    pre = node->right;
                }
                node = node->next; //node will be one level upper compared with pre and head
            }
            node = head;
            pre = head = nullptr; //set the auxiliary pointers to nullptr
        }
        return root;
    }
};


//Interview: Find the maximum sum from root to leaf in a binary tree
//You also need to print the path of the maximum sum
//***********************************************************************//
struct Node {
	int val;
	Node* left;
	Node* right;
	Node(int x) : val(x), left(nullptr), right(nullptr) {}
};

//Now the time complexity should be O(n) - post order traversal
//Space complexity should be O(h) - h is the height of the tree
int TreeSum(Node* node, int& maxSum, vector<int>& res) {
	if (!node) return 0;

	vector<int> pathL, pathR;
	int leftSum = TreeSum(node->left, maxSum, pathL);
	int rightSum = TreeSum(node->right, maxSum, pathR);

	if (leftSum > rightSum) {
		maxSum = leftSum + node->val;
		pathL.push_back(node->val);
		res.swap(pathL);
	}
	else if (leftSum <= rightSum) {
		maxSum = rightSum + node->val;
		pathR.push_back(node->val);
		res.swap(pathR);
	}
	return maxSum;
}

// Function to insert nodes in level order 
Node* insertLevelOrder(vector<int>& arr, Node* root,
	int i)
{ 
	if (i < arr.size())
	{
		Node* temp = new Node(arr[i]);
		root = temp;

		// insert left child 
		root->left = insertLevelOrder(arr,
			root->left, 2 * i + 1);

		// insert right child 
		root->right = insertLevelOrder(arr,
			root->right, 2 * i + 2);
	}
	return root;
}

int main() {
	//Define the tree value
	vector<int> TreeVal = {1, 8, -98, -34, 23};
	Node* root = nullptr;
	//Build tree in level order 
	root = insertLevelOrder(TreeVal, root, 0);

	//Calculate the maximum path sum
	int maxSum = 0;
	vector<int> res;
	TreeSum(root, maxSum, res);

	//print the path from leaf to root
	for (int n : res)
		cout << n << " " ;
	cout << maxSum << endl;

	//We need to destroy tree here, omit now
	//destroyTree(root);

	system("pause");
	return 0;
}
//***********************************************************************//


//1022. Sum of Root To Leaf Binary Numbers
//https://leetcode.com/problems/sum-of-root-to-leaf-binary-numbers/
/* Recursive version, not very intuitive. Try another implementation */
class Solution {
private:
    int postOrder(TreeNode* node, int preVal){
        if(!node) return 0;
        preVal = 2 * preVal + node->val;
        if(!node->left && !node->right) return preVal;
        
        int leftSum = postOrder(node->left, preVal);
        int rightSum = postOrder(node->right, preVal);
        
        return leftSum + rightSum;
    }
public:
    int sumRootToLeaf(TreeNode* root) {
        return postOrder(root, 0);
    }
};


//404. Sum of Left Leaves
//https://leetcode.com/problems/sum-of-left-leaves/
class Solution {
    int sumLeft(TreeNode* node, bool isLeft){
        if(!node) return 0;
        if(!node->left && !node->right && isLeft) return node->val;
        int res = 0;
        res = sumLeft(node->left, true) + sumLeft(node->right, false);
        return res;
    }
public:
    int sumOfLeftLeaves(TreeNode* root) {
        if(!root) return 0;
        return sumLeft(root->left, true) + sumLeft(root->right, false);
    }
};

/* I  come up with this strange implementation. Post-order traversal
will be better. Iterative version.*/
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        if(!root || (!root->left && !root->right)) return 0;
        stack<TreeNode*> St;
        stack<int> bSt;
        TreeNode* cur = root;
        int sum = 0;
        int bFlag = 1;
        while(cur || !St.empty()){
            if(cur){
                St.push(cur);
                bSt.push(bFlag);
                cur = cur->left;
                bFlag = 1;
            }else{
                cur = St.top();
                St.pop();
                bFlag = bSt.top();
                bSt.pop();
                if(!cur->right && ! cur->left && bFlag)
                    sum += cur->val;
                
                if(cur->right){
                    cur = cur->right;
                    bFlag = 0;
                }
                else if(!cur->right) 
                    cur = nullptr;    
            }
        }
        return sum;
    }
};

/* Post order, not that efficient! */
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        if(!root || (!root->left && !root->right)) return 0;
        stack<TreeNode*> St;
        stack<int> bSt;
        TreeNode* cur = root;
        TreeNode* pre = nullptr;
        int sum = 0;
        int bFlag = 1;
        while(cur || !St.empty()){
            if(cur){
                St.push(cur);
                bSt.push(bFlag);
                cur = cur->left;
                bFlag = 1;
            }else{
                cur = St.top();
                bFlag = bSt.top();
                if(cur->right && pre != cur->right){
                    cur = cur->right;
                    bFlag = 0;
                }
                else{
                    if(!cur->right && ! cur->left && bFlag)
                        sum += cur->val;
                    St.pop();
                    bSt.pop();
                    pre = cur;
                    cur = nullptr;
                } 
                       
            }
        }
        return sum;
    }
};


//450. Delete Node in a BST
//https://leetcode.com/problems/delete-node-in-a-bst/
//Recursive version is popular!
//Good explanation:
//https://www.youtube.com/watch?v=gcULXE7ViZw&vl=en
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
                //We need to find the minimum value in right sub tree
                //or we find the maximum value in left sub tree
                TreeNode* temp = findMin(root->right);
                //We set the our current root to be the minimum value
                root->val = temp->val;
                //Reduce to case 1 or 2
                root->right = deleteNode(root->right, temp->val);
                
            }
        }
        return root;
    }
};


//701. Insert into a Binary Search Tree
//https://leetcode.com/problems/insert-into-a-binary-search-tree/
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        //Base case, when tree is empty
        if(!root){
            root = new TreeNode(val);
        }//insert to left sub tree
        else if(root->val >= val)
            root->left = insertIntoBST(root->left, val);
        else
            root->right = insertIntoBST(root->right, val);
        return root;
    }
};


//Interview: Maximum path sum of a tree (from root -> leaf)
//Calculate the path sum from root to all leaves, and return the maximum path sum
//and also return the path contains the sum
//Now the time complexity should be O(n) - post order traversal
//Space complexity should be O(h) - h is the height of the tree
int TreeSum(Node* node, int& maxSum, vector<int>& res) {
	if (!node) return 0;

	vector<int> pathL, pathR;
	int leftSum = TreeSum(node->left, maxSum, pathL);
	int rightSum = TreeSum(node->right, maxSum, pathR);

	if (leftSum > rightSum) {
		maxSum = leftSum + node->val;
		pathL.push_back(node->val);
		res.swap(pathL);
	}
	else if (leftSum <= rightSum) {
		maxSum = rightSum + node->val;
		pathR.push_back(node->val);
		res.swap(pathR);
	}
	return maxSum;
}


//Build binary tree from an array in level order
//The array defines the tree in the following order
//arr[0] is the root, the left child and right child of node i are
//(2*i+1) and (2*i+2)
// Function to insert nodes in level order 
Node* insertLevelOrder(vector<int>& arr, Node* root,
	int i)
{ 
	if (i < arr.size())
	{
		Node* temp = new Node(arr[i]);
		root = temp;

		// insert left child 
		root->left = insertLevelOrder(arr,
			root->left, 2 * i + 1);

		// insert right child 
		root->right = insertLevelOrder(arr,
			root->right, 2 * i + 2);
	}
	return root;
}


//222. Count Complete Tree Nodes
//https://leetcode.com/problems/count-complete-tree-nodes/
//Level order Traversal
class Solution {
public:
    int countNodes(TreeNode* root) {
        int count = 0;
        if(!root) return count;
        queue<TreeNode*> Q;
        Q.push(root);
        while(!Q.empty()){
            TreeNode* node = Q.front();
            Q.pop();
            count++;
            if(node->left) Q.push(node->left);
            if(node->right) Q.push(node->right);
        }
        return count;
    }
};

//We can do binary search
//O(h logh)
class Solution {
private:
    int countDepth(TreeNode* node, bool isLeft){
        int depth = 0;
        while(node){
            depth++;
            node = isLeft ? node->left : node->right;
        }
        return depth;
    }
public:
    int countNodes(TreeNode* root) {
        if(!root) return 0;
        int leftDepth = countDepth(root, true);
        int rightDepth = countDepth(root, false);
        if(leftDepth == rightDepth)
            return (1 << leftDepth) - 1;
        else
            return 1 + countNodes(root->left) + countNodes(root->right);
    }
};


//1161. Maximum Level Sum of a Binary Tree
//https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/
//General BFS
class Solution {
public:
    int maxLevelSum(TreeNode* root) {
        if(!root) return 0;
        queue<TreeNode*> Q;
        Q.push(root);
        int maxLevelSum = INT_MIN;
        int minLevel = 1;
        int curLevel = 0;
        while(!Q.empty()){
            int lenQ = Q.size();
            int tempSum = 0;
            curLevel++;
            for(int i = 0; i < lenQ; ++i){
                TreeNode* node = Q.front();
                Q.pop();
                tempSum += node->val;
                if(node->left) Q.push(node->left);
                if(node->right) Q.push(node->right);
            }
            if(maxLevelSum < tempSum){
                maxLevelSum = tempSum;
                minLevel = curLevel;
            }
        }
        return minLevel;
    }
};


//Google interview:
//Create a class LightString to manage a string of lights, each of which has 
//a value 0 or 1 indicating whether it is on or off.
/*
class LightString {
    public LightString(int numOfLights) {
	}

    // Return if the i-th light is on or off.
    public boolean isOn(int i) {
	}

	//Switch state (turn on if it's off, turn off if it's on) of every 
    //light in the range [start, end].
    public void toggle(int start, int end) {
	}
}
*/
/*
Example:

LightString str = new LightString(5); // all lights are initially off
str.isOn(0); // false
str.isOn(1); // false
str.isOn(2); // false
str.toggle(0, 2);
str.isOn(0); // true
str.isOn(1); // true
str.isOn(2); // true
str.isOn(3); // false
str.toggle(1, 3);
str.isOn(1); // false
str.isOn(2); // false
str.isOn(3); // true

Can you do better than O(n) for update?
*/
class LightString {
	public:
	LightString(int numOfLights) : n(numOfLights) {
		tree.resize(n << 1);
	}
	
	bool isOn(int i) {
		i += n;
		int cnt = 0;
		while (i > 0) {
			cnt += tree[i];
			i >>= 1;
		}
		return cnt & 1;
	}
	
	void toggle(int start, int end) {
		for (start += n, end += n; start <= end; start >>= 1, end >>= 1) {
			if (start & 1) {
				++tree[start++];
			}
			if (!(end & 1)) {
				++tree[end--];
			}
		}
	}
	
	private:
	int n;
	vector<int> tree;
};


//Google Interview! 
//Good question! 
//https://leetcode.com/discuss/interview-question/294183/
//Java code:
//The key insight is to find the biggest opponentMove's 3 adjacent components!
//Since that's the potential best outcome I can get!
public static boolean canWin(TreeNode root, TreeNode opponentMove) {
    int parentSize = countNodes(opponentMove.parent, opponentMove); // size of parent component
    int leftSize = countNodes(opponentMove.left, opponentMove);   // size of left subtree component
    int rightSize = countNodes(opponentMove.right, opponentMove); // size of right subtree component

    int myScore = Math.max(Math.max(parentSize, leftSize), rightSize); // I take the biggest component

    int treeSize = 1 + parentSize + leftSize + rightSize;

    int opponentScore = treeSize - myScore; // opponent takes remaining nodes
    System.out.print("my best score is " + myScore + "/" + treeSize + ". ");

    if (myScore > opponentScore) {
        TreeNode bestMove = myScore == parentSize ? opponentMove.parent : myScore == leftSize ? opponentMove.left : opponentMove.right;
        System.out.println("my first move on " + bestMove.val);
    }

    return myScore > opponentScore;
}

private static int countNodes(TreeNode node, TreeNode ignore) {
    if (node == null) return 0;
    int count = 1;
    if (node.parent != ignore) {
        count += countNodes(node.parent, node);
    }
    if (node.left != ignore) {
        count += countNodes(node.left, node);
    }
    if (node.right != ignore) {
        count += countNodes(node.right, node);
    }
    return count;
}

//Follow up: Jave code!
//This is the same concept like the original one. If I do the first move, I 
//will choose a node which split the tree to 3 components, with the largest 
//possible component to be less than num of tree nodes / 2. Then no matter how
//opponent moves, we can guarantee a win.
public static TreeNode bestMove(TreeNode root) {
    if (root == null) return null;
    if (root.left == null && root.right == null) return root;

    // map stores size of every component
    // each node at most has 3 components - to its left, to its right, to its top (parent)
    // Map<node, Map<which component, size>>
    Map<TreeNode, Map<TreeNode, Integer>> components = new HashMap<>();
    TreeNode dummy = new TreeNode(-1);
    dummy.left = root;

    // calculate size of child components for all nodes
    getComponentSize(dummy, root, components);

    int treeSize = components.get(dummy).get(root);
    for (TreeNode node : components.keySet()) {
        int maxSize = 0; // maximum score possible for opponent
        for (int size : components.get(node).values()) {
            maxSize = Math.max(maxSize, size);
        }
        if (treeSize / 2.0 > maxSize) { // opponent cannot get half of the tree. You win.
            return node; // best first move
        }
    }
    return null; // no winning play
}

private static int getComponentSize(TreeNode n, TreeNode branch, Map<TreeNode, Map<TreeNode, Integer>> components) {
    if (n == null || branch == null) return 0;

    if (components.containsKey(n) && components.get(n).containsKey(branch)) {
        return components.get(n).get(branch); // component size of a branch from node n (n excluded)
    }
    // a node n has 3 branches at most - parent, left, right
    if (!components.containsKey(n)) {
        components.put(n, new HashMap<>());
    }

    int size = 1; // 1 is the size of TreeNode branch
    if (branch == n.left || branch == n.right) {
        // size of the subtree 'branch' is size(branch.left) + size(branch.right) + 1
        size += getComponentSize(branch, branch.left, components);
        size += getComponentSize(branch, branch.right, components);
    } else { //else when (branch == n.parent)
        // see the tree from left-side or right-side view (see parent as a child; see one of the children as parent)
        // size of the component is size(branch.parent) + size(branch.left/right child)
        size += getComponentSize(branch, branch.parent, components);
        size += branch.left == n ? getComponentSize(branch, branch.right, components) : getComponentSize(branch, branch.left, components);
    }
    components.get(n).put(branch, size); // cache the result of recursion
    getComponentSize(n, n.parent, components); // calculate size of parent component for current node
    return size;
}


//https://leetcode.com/discuss/interview-question/376813/
//Google interview!
//The tricky part is how to randomly pick up a word. 
/*
For each trie node, keep track of the number of words below/ending at it. 
Start at the root level and for each level generate a random number r between 
(1, total number of words of all the nodes for that level) and route 
accordingly. Once you select the trie node by above method, if no word ends 
at it, proceed to the below level and repeat the same. Otherwise stop there 
by generating a random number r between (1, total number of words for that 
node)

eg) using the one from @btaylor4
Art, Apple, Basic ,Cast, Care, Case, Dying, Daring, Dashing, Doll.
Root has A-2, B-1, C-3, D-4. Generate a random number between (1, 10). 
Assume 8 so D is selected. Prob - 4/10
For next level A-2, O-1, Y-1. Generate (1, 4). Assume 1 so A is selected. 
Prob - 2/4
Next R-1, S-1. Generate (1,2). Assume 2 so S. Prob - 1/2
Net prob = 4/10 * 2/4 * 1/2 = 1/10
*/
//Try tomorrow


//429. N-ary Tree Level Order Traversal
//https://leetcode.com/problems/n-ary-tree-level-order-traversal/
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
*/
class Solution {
public:
    vector<vector<int>> levelOrder(Node* root) {
        vector<vector<int>> res;
        if(!root) return res;
        queue<Node*> Q;
        Q.push(root);
        while(!Q.empty()){
            int lenQ = Q.size();
            vector<int> levelNodes;
            for(int i = 0; i < lenQ; ++i){
                Node* tempNode = Q.front();
                Q.pop();
                levelNodes.push_back(tempNode->val);
                for(Node*& n : tempNode->children){
                    if(n) Q.push(n);
                }
            }
            res.push_back(levelNodes);
        }
        return res;
    }
};


//437. Path Sum III
//https://leetcode.com/problems/path-sum-iii/
//Double check later! you did not get it! 
//Try something intuitive to you. Not very efficient!
class Solution {
private:
    int rootSum(TreeNode* node, int sum){
        if(!node) return 0;
        int nextSum = sum - node->val;
        int totalWithNode = (nextSum == 0 ? 1 : 0) + rootSum(node->left, nextSum) + rootSum(node->right, nextSum);
        return totalWithNode;
    }
public:
    int pathSum(TreeNode* root, int sum) {
        if(!root) return 0;
        return rootSum(root, sum) + pathSum(root->left, sum) + pathSum(root->right, sum);
    }
};

//Optimized version with memorization + prefixsum
/*
prefix sum + tree traversal. 
Tricky, almost impossible during the interview!
*/
class Solution {
private:
    int helper(TreeNode* node, int sum, unordered_map<int, int>& uMap, int preVal){
        if(!node) return 0;
        //Calculate the presum for node i (sum of this node with parent node)
        node->val += preVal;
        //uMap: current - sum equals some previous sum, which means we have
        //a valid path.
        int res = (node->val == sum) + (uMap.count(node->val - sum) ? uMap[node->val - sum] : 0);
        //add current sum to map
        uMap[node->val] ++;
        res += helper(node->left, sum, uMap, node->val) + helper(node->right, sum, uMap, node->val);
        //Note we are backtracking here
        uMap[node->val] --;
        return res;
        
    }
public:
    int pathSum(TreeNode* root, int sum) {
        //We will use this map to record the presum we have already found!
        unordered_map<int, int> uMap;
        return helper(root, sum, uMap, 0);
    }
};


//427. Construct Quad Tree
//https://leetcode.com/problems/construct-quad-tree/
//The following code does not work! while I cannot tell the difference between
//it with the valid java code:
//
/*
class Solution {
private:
    Node* helper(vector<vector<int>>& G, int left, int right, int up, int bottom){
        //if(left > right || up > bottom) return nullptr;
        if(left == right || up == bottom) 
            return new Node(G[left][up] == 1, true, nullptr, nullptr, nullptr, nullptr);
        int midX = left + (right - left) / 2;
        int midY = up + (bottom - up) / 2;
        Node* topL = helper(G, left, midX, up, midY);
        Node* topR = helper(G, midX+1, right, up, midY);
        Node* bottomL = helper(G, left, midX, midY+1, bottom);
        Node* bottomR = helper(G, midX+1, right, midY+1, bottom);
        
        Node* root = new Node(G[left][up] == 1, true, topL, topR, bottomL, bottomR);
        
        if(!topL || !topR || !bottomL || !bottomR){
            return root;
        }
        
        if(topL->isLeaf && topR->isLeaf && bottomL->isLeaf && bottomR->isLeaf && topL->val == topR->val && topL->val == bottomL->val && topL->val == bottomR->val){
            root->val = topL->val;
            root->topLeft = nullptr;
            delete topL;
            root->topRight = nullptr;
            delete topR;
            root->bottomLeft = nullptr;
            delete bottomL;
            root->bottomRight = nullptr;
            delete bottomR;
            
        }else{
            root->isLeaf = false;
        }
        
        return root;
    }
public:
    Node* construct(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = m ? grid[0].size() : 0;
        if(!m || !n) return nullptr;
        return helper(grid, 0, n-1, 0, m-1);
    }
};
*/


/*
// Definition for a QuadTree node.
class Node {
public:
    bool val;
    bool isLeaf;
    Node* topLeft;
    Node* topRight;
    Node* bottomLeft;
    Node* bottomRight;

    Node() {}

    Node(bool _val, bool _isLeaf, Node* _topLeft, Node* _topRight, Node* _bottomLeft, Node* _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
};
*/
//The same idea with length works. Weird!
class Solution {
private:
    Node* buildNode(vector<vector<int>>& grid, int x, int y, int length) {
        if (length == 1) {
            return new Node(grid[x][y] == 1, true, nullptr, nullptr, nullptr, nullptr);
        }
        
        int newLength = length / 2;
        Node* topLeft = buildNode(grid, x, y, newLength);
        Node* topRight = buildNode(grid, x, y + newLength, newLength);
        Node* botLeft = buildNode(grid, x + newLength, y, newLength);
        Node* botRight = buildNode(grid, x + newLength, y + newLength, newLength);
        
        if (topLeft -> isLeaf && topRight -> isLeaf && botRight -> isLeaf && botLeft -> isLeaf &&
            ((topLeft -> val && topRight -> val && botLeft -> val && botRight -> val) ||
            !(topLeft -> val || topRight -> val || botLeft -> val || botRight -> val))) {
            bool val = topLeft -> val;
            delete topLeft;
            topLeft = nullptr;
            delete topRight;
            topRight = nullptr;
            delete botLeft;
            botLeft = nullptr;
            delete botRight;
            botRight = nullptr;
            return new Node(val, true, nullptr, nullptr, nullptr, nullptr);
        }
        return new Node(true, false, topLeft, topRight, botLeft, botRight);
    }
public:
    Node* construct(vector<vector<int>>& grid) {
        int N = grid.size();
        if (N == 0) {
            return nullptr;
        }
        return buildNode(grid, 0, 0, N);
    }
};

//Optimized version: We check whether the valid grid can forms a leaf, so we 
//no longer need to recursively explore all the results
class Solution {
public:
    Node* f(const vector<vector<int>>& grid, int i1, int j1,int i2,int j2){
        if(i1==i2 && j1==j2){
            Node* ret = new Node();
            ret -> val = grid[i1][j1]?true:false;
            ret->isLeaf = true;
            ret->topLeft = ret->topRight = ret->bottomLeft = ret->bottomRight = nullptr;
            return ret;
        }
        //Optimization here!
        int val = grid[i1][j1];
        int allSame = true;
        for(int i = i1;i<=i2;i++) {
            for(int j = j1;j<=j2;j++) {
                if(grid[i][j]!=val) {
                    allSame = false;
                    break;
                }
            }
        }
        if(allSame) {
            Node* ret = new Node();
            ret->val = val?true:false;
            ret->isLeaf = true;
            ret->topLeft = ret->topRight = ret->bottomLeft = ret->bottomRight = nullptr;
            return ret;
        }
        Node* ret = new Node();
        ret->isLeaf = false;
        ret->val = true;
        int i_mid = (i1+i2)/2;
        int j_mid = (j1+j2)/2;
        ret->topLeft = f(grid,i1,j1,i_mid,j_mid);
        ret->topRight = f(grid,i1,j_mid+1,i_mid,j2);
        ret->bottomLeft = f(grid,i_mid+1,j1,i2,j_mid);
        ret->bottomRight = f(grid,i_mid+1,j_mid+1,i2,j2);
        return ret;
    }
    Node* construct(vector<vector<int>>& grid) {
        if(grid.empty()){
            return nullptr;
        }
        return f(grid,0,0,grid.size()-1,grid[0].size()-1);
    }
};


//979. Distribute Coins in Binary Tree
//https://leetcode.com/problems/distribute-coins-in-binary-tree/
//You should consider leaves first. If leaf has more coins, it should 
//push the coins to its parent, or it needs more coins from its parent!
class Solution {
    int DFS(TreeNode* root, int& count){
        if(!root) return 0;
        int leftTotal = DFS(root->left, count);
        int rightTotal = DFS(root->right, count);
        count += abs(leftTotal) + abs(rightTotal);
        return root->val - 1 + leftTotal + rightTotal;
    }
public:
    int distributeCoins(TreeNode* root) {
        if(!root) return 0;
        int count = 0;
        DFS(root, count);
        return count;
    }
};


//617. Merge Two Binary Trees
//https://leetcode.com/problems/merge-two-binary-trees/
class Solution {
private:
    TreeNode* buildTree(TreeNode* node){
        if(!node) return nullptr;
        TreeNode* root = new TreeNode(node->val);
        if(node->left)
            root->left = buildTree(node->left);
        if(node->right)
            root->right = buildTree(node->right);
        return root;
    }
    TreeNode* helper(TreeNode* t1, TreeNode* t2){
        if(!t1 && !t2) return nullptr;
        if(!t1) return buildTree(t2);
        if(!t2) return buildTree(t1);
        
        TreeNode* root = new TreeNode(t1->val + t2->val);
        root->left = helper(t1->left, t2->left);
        root->right = helper(t1->right, t2->right);
        return root;
    }
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        return helper(t1, t2);
    }
};



//1145. Binary Tree Coloring Game
//https://leetcode.com/problems/binary-tree-coloring-game/
//The problem is not hard, we just traverse the tree and find x, and check the
//3 partitions that x divides. If the sum of any of the two is less than third
//one, which means our y can start from partiton 3 and guarantee a win!
class Solution {
    int countNodes(TreeNode* root){
        if(!root) return 0;
        int leftCount = countNodes(root->left);
        int rightCount = countNodes(root->right);
        return leftCount+ rightCount + 1;
    }
    TreeNode* findNode(TreeNode* root, int x){
        if(!root) return nullptr;
        if(root->val == x) return root;
        TreeNode* leftNode = findNode(root->left, x);
        TreeNode* rightNode = findNode(root->right, x);
        return leftNode ? leftNode : rightNode;
    }
public:
    bool btreeGameWinningMove(TreeNode* root, int n, int x) {
        if(!root) return false;
        TreeNode* nodeP1 = findNode(root, x);
        int partition1 = countNodes(nodeP1->left);
        int partition2 = countNodes(nodeP1->right);
        int partition3 = n - 1 - partition1 - partition2;
        return (partition3 > (partition1 + partition2)) || 
        (partition1 > (partition3 + partition2)) || 
        partition2 > (partition1 + partition3);
    }
};


//449. Serialize and Deserialize BST
//https://leetcode.com/problems/serialize-and-deserialize-bst/
//Following code: in order traversal! do not work for this kind of problem!
//Hard to maintain tree structure!
/*
class Codec {
private:
    string m_series;
public:
    
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        if(!root) return "n ";
        string s = serialize(root->left);
        
        int val = root->val;
        s.append(to_string(val) + ' ');
        
        s.append(serialize(root->right));
        
        return s;
    }
    
    TreeNode* buildTree(vector<string>& v, int l, int r){
        if(l > r) return nullptr;
        // We always right side node first
        int mid = l + (r - l) / 2;
        if(v[mid] == "n") return nullptr;
        TreeNode* root = new TreeNode(stoi(v[mid]));
        
        root->left = buildTree(v, l, mid-1);
        root->right = buildTree(v, mid+1, r);
        return root;
    }
    
    void stv(string& s, vector<string>& v){
        string tempS;
        for(int i = 0; i < s.size(); ++i){
            if(s[i] != ' '){
                if(s[i] == 'n') v.push_back(string(1, s[i]));
                else {
                    tempS.push_back(s[i]);
                }
            }else{
                if(!tempS.empty()){
                    v.push_back(tempS);
                    tempS.clear();
                }
            }
        }
    }
    
    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        cout << data << endl;
        vector<string> v;
        stv(data, v);
        //for(auto s : v) cout << s << endl;
        return buildTree(v, 0, v.size()-1);
    }
};
*/

//Level-order traversal!
class Codec {
public:
    
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        if(!root) return "";
        queue<TreeNode*> Q;
        string res;
        Q.push(root);
        res.append(to_string(root->val) + ',');
        while(!Q.empty()){
            TreeNode* node = Q.front();
            Q.pop();
            if(!node->left) res.append("n,");
            else {
                res.append(to_string(node->left->val) + ',');
                Q.push(node->left);
            }
            
            if(!node->right) res.append("n,");
            else {
                res.append(to_string(node->right->val) + ',');
                Q.push(node->right);
            }
        }
        return res;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if(data.empty()) return nullptr;
        vector<string> strVector;
        string tempStr;
        for(int i = 0; i < data.size(); ++i){
            if(data[i] == 'n') strVector.push_back("n");
            else if(data[i]!=','){
                tempStr.push_back(data[i]);
            }else if(data[i] == ','){
                if(!tempStr.empty()){
                    strVector.push_back(tempStr);
                    tempStr.clear();
                }
            }
        }
        
        queue<TreeNode*> Q;
        int k = 0;
        TreeNode* root = new TreeNode(stoi(strVector[k]));
        Q.push(root);
        while(!Q.empty()){
            TreeNode* node = Q.front();
            Q.pop();
            k++;
            if(k < strVector.size() && strVector[k] != "n"){
                int val = stoi(strVector[k]);
                TreeNode* leftNode = new TreeNode(val);
                node->left = leftNode;
                Q.push(leftNode);
            }
            k++;
            if(k < strVector.size() && strVector[k] != "n"){
                int val = stoi(strVector[k]);
                TreeNode* rightNode = new TreeNode(val);
                node->right = rightNode;
                Q.push(rightNode);
            }
        }
        return root;
    }
};



//951. Flip Equivalent Binary Trees
//https://leetcode.com/problems/flip-equivalent-binary-trees/
//My implementation!
class Solution {
public:
    bool flipEquiv(TreeNode* root1, TreeNode* root2) {
        if(!root1 && !root2) return true;
        if(!root1 || !root2) return false;
        
        bool noFlip = flipEquiv(root1->left, root2->left) && 
            flipEquiv(root1->right, root2->right);
        
        bool flip = flipEquiv(root1->left, root2->right) && 
            flipEquiv(root1->right, root2->left);
        
        if(root1->val == root2->val && (noFlip || flip)) return true;
        
        return false;
    }
};


//298. Binary Tree Longest Consecutive Sequence
//https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/
class Solution {
private:
    int maxPossible = INT_MIN;
    void helper(TreeNode* node, int preVal, int cnt){
        if(!node) return;
        int res = 0;
        if(preVal == node->val){
            res = 1 + cnt;
        }else{
            res = 1;
        }
        maxPossible = max(maxPossible, res);
        
        helper(node->left, node->val+1, res);
        helper(node->right, node->val+1, res);
    }
public:
    int longestConsecutive(TreeNode* root) {
        if(!root) return 0;
        int cnt = 0;
        helper(root, root->val, cnt);
        return maxPossible;
    }
};



//250. Count Univalue Subtrees
//https://leetcode.com/problems/count-univalue-subtrees/
//The implementation is tricky! Be caureful.
class Solution {
private:
    int cnt = 0;
    bool helper(TreeNode* node, int preVal){
        if(!node) return true;
        
        //If the left or the right subtree is not univalue tree, then we 
        //return false. Because for current node, it is impossible to be 
        //a Uni-value tree
        //We need to use | instead of || here because | will check both 
        //sides even if the first statement is true
        if(!helper(node->left, node->val) | !helper(node->right, node->val))
            return false;
        
        cnt++;
        //we return the its parent and tell the parent whether this is a 
        //valid subtree (one of the two trees)
        return node->val == preVal;
    }
public:
    int countUnivalSubtrees(TreeNode* root) {
        if(!root) return 0;
        helper(root, root->val);
        return cnt;
    }
};


//366. Find Leaves of Binary Tree
//https://leetcode.com/problems/find-leaves-of-binary-tree/
//Implemented by me, took somewhile
class Solution {
private:
    int helper(TreeNode* node, vector<vector<int>>& res){
        if(!node) return 0;
        
        int leftLevel = helper(node->left, res);
        int rightLevel = helper(node->right, res);
        
        int level = max(leftLevel, rightLevel);
        if(level >= res.size()){
            vector<int> temp(1, node->val);
            res.emplace_back(temp);
        }else{
            res[level].push_back(node->val);
        }
        return level+1;
    }
public:
    vector<vector<int>> findLeaves(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        helper(root, res);
        return res;
    }
};



//285. Inorder Successor in BST
//https://leetcode.com/problems/inorder-successor-in-bst/
//Using a flag is key to success
//For this problem, we cannot use recursive version. It will be hard to
//track the right node.
class Solution {
public:
    TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
        if(!root || !p) return nullptr;
        stack<TreeNode*> st;
        TreeNode* pre = nullptr, *cur = root;
        bool isFoundP = false;

        while(cur || !st.empty()){
            while(cur){
                st.push(cur);
                cur = cur->left;
            }
            cur = st.top();
            st.pop();
            
            if(isFoundP) return cur;
            if(cur->val == p->val) isFoundP = true;
            
            //We need to move right now
            if(cur->right && cur->right != pre){
                //if(cur->val == p->val) return cur->right;
                cur = cur->right;
            }else{
                pre = cur;
                cur = nullptr;
            }
        }
        return nullptr;
    }
};


//510. Inorder Successor in BST II
//https://leetcode.com/problems/inorder-successor-in-bst-ii/
class Solution {
public:
    Node* inorderSuccessor(Node* node) {
        if(!node) return nullptr;
        //I did not get this if statement on my own. Note we need to 
        //traverse up until we find a node which is the left child of parent
        //node, that node will be the target node or nullptr
        if(!node->right) {
            //node is the right child of its parent
            //we need to go up
            //[5,3,6,2,4,null,null,1] with target to be 4
            //Try it yourself
            while(node->parent && node == node->parent->right)
                node = node->parent;
            return node->parent;
        }
        else{
            Node* temp = node->right;
            while(temp->left){
                temp = temp->left;
            }
            return temp;
        }
    }
};


//270. Closest Binary Search Tree Value
//https://leetcode.com/problems/closest-binary-search-tree-value/
//Naive approach is to use an extra array and do in-order traversal
//then find the closest element and return in a sorted array.
//Actually, we can do binary search and keep the closest element updated.
class Solution {
public:
    int closestValue(TreeNode* root, double target) {
        int res = root->val;
        double closest = DBL_MAX;
        TreeNode* ptr = root;
        while(ptr){
            double diff = abs(double(ptr->val) - target);
            if(diff < closest){
                closest = diff;
                res = ptr->val;
            }
            ptr = ptr->val > target ? ptr->left : ptr->right;
        }
        return res;
    }
};


//272. Closest Binary Search Tree Value II
//https://leetcode.com/problems/closest-binary-search-tree-value-ii/
// O(n)
//The general idea is to maintain a queue with size k, whenever we visit a 
//new element and the queue has k elements, then we compare the distance 
//between the new element with the distance between the first element. 
//If the distance is smaller, then we need to pop the first element, because
//this one is the smallest element (largest distance with target), and push
//this new element. If the current distance is larger, we can safely break!
//(because the distance will become larger and larger!)
class Solution {
public:
    vector<int> closestKValues(TreeNode* root, double target, int k) {
        vector<int> res;
        if(!root || k <= 0) return res;
        
        stack<TreeNode*> st;
        TreeNode* pre = nullptr, *cur = root;
        deque<int> dq;
        
        while(cur || !st.empty()){
            while(cur){
                st.push(cur);
                cur = cur->left;
            }
            cur = st.top();
            st.pop();
            if(dq.size() < k){
                dq.push_back(cur->val);
            }else{
                double diff = abs(double(cur->val) - target);
                double diff2 = abs(double(dq.front()) - target);
                if(diff < diff2){
                    dq.pop_front();
                    dq.push_back(cur->val);
                }else{
                    break;
                }
            }
            if(cur->right && cur->right != pre){
                cur = cur->right;
            }else{
                pre = cur;
                cur = nullptr;
            }
        }
        
        for(auto it = dq.begin(); it != dq.end(); ++it){
            res.push_back(*it);
        }
        return res;
    }
};


//314. Binary Tree Vertical Order Traversal
//https://leetcode.com/problems/binary-tree-vertical-order-traversal/
//Level order traversal and keep track of each column and its corresponding
//nodes. We can utilize a hash table to do this and keep track of the lower
//bound and upperbound of the column.
class Solution {
public:
    vector<vector<int>> verticalOrder(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        int lo = 0, hi = 0;
        //Q saves the current node and its coresponding column.
        queue<pair<TreeNode*, int>> Q;
        //uMap saves the corresponding column and the nodes belong to the 
        //coresponding column.
        unordered_map<int, vector<int>> uMap;
        Q.push({root, 0});
        while(!Q.empty()){
            TreeNode* node = Q.front().first;
            int col = Q.front().second;
            lo = min(lo, col);
            hi = max(hi, col);
            Q.pop();
            uMap[col].push_back(node->val);
            if(node->left){
                Q.push({node->left, col-1});
            }
            if(node->right){
                Q.push({node->right, col+1});
            }
        }
        for(int i = lo; i <= hi; ++i){
            if(uMap.count(i) > 0)
                res.emplace_back(uMap[i]);
        }
        return res;
        
    }
};



//1367. Linked List in Binary Tree
//https://leetcode.com/contest/weekly-contest-178/problems/linked-list-in-binary-tree/
//You did not get it right during the contest, what a shame!
class Solution {
private:
    bool isValid(ListNode* head, TreeNode* root){
        if(!head) return true;
        if(!root) return false;
        
        return head->val == root->val && (isValid(head->next, root->left) || isValid(head->next, root->right));
    }
public:
    bool isSubPath(ListNode* head, TreeNode* root) {
        //Tree can potentially be empty, especially we put it into recursive form
        if(!root) return false;
        
        return isValid(head, root) || isSubPath(head, root->left) || isSubPath(head, root->right);
    }
};


//1372. Longest ZigZag Path in a Binary Tree
//https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/
//You did not make it during bContest 21st
//Execellent solution, still a little bit confusing!
class Solution {
    //calZigZag will return the current valid nodes we can visit starting from
    //node
    
    //preLeft - previous comes from left branch
    //curMax - current max number of valid nodes starting from this node, 
    //since we do post order traversal. It will be 
    //updated when we search left and right node
    int calZigZag(TreeNode* node, bool preLeft, int& curMax){
        if(!node) return 0;
        //leftMax means explore the left branch of node. We plus 1 to include node!
        int leftMax = calZigZag(node->left, true, curMax) + 1;
        int rightMax = calZigZag(node->right, false, curMax) + 1;
        
        curMax = max(curMax, max(leftMax, rightMax));
        
        return preLeft ? rightMax : leftMax;
        
    }
public:
    int longestZigZag(TreeNode* root) {
        if(!root) return 0;
        int maxLen = 0;
        //This is a confusing staff, imagine we have a virtual node comes from 
        //left of root, actually, true or false does not really matter. Since they
        //will be the same!
        calZigZag(root, false, maxLen);
        
        //We need to -1 according to the definition
        return maxLen - 1;
        
    }
};


// Another perspective, much slower!
class Solution {
    //return value has 3 components:
    //0 - first Max
    //1 - current Max
    //2 - right Max
    vector<int> calZigZag(TreeNode* node){
        if(!node) return vector<int>({-1, -1, -1});
        
        vector<int> leftMax = calZigZag(node->left);
        vector<int> rightMax = calZigZag(node->right);
        
        vector<int> res (3, 0);
        //leftMax[2] represents the right optimal solution for left child,
        //we need to add 1 to include left child node. Node in order to make
        //sure that we are still in a chain, we need to gurantee include the 
        //sub-tree node by adding 1 here
        res[0] = leftMax[2] + 1;
        //final optimal solution!
        res[1] = max(max(leftMax[2], rightMax[0]) + 1, max(leftMax[1], rightMax[1]));
        //the optisite compared with above left
        res[2] = rightMax[0] + 1;
        
        return res;
    }
public:
    int longestZigZag(TreeNode* root) {
        if(!root) return 0;
        return calZigZag(root)[1];
    }
};


//1373. Maximum Sum BST in Binary Tree
//https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/
//You did not make it during the biweekly contest 21
// Post order traversal, however, you need some intuition to get
// it right! The idea is we first check all the children, if they
// are valid BST and with current node, they can still form a valid
// BST, then we can update our maximum sub tree sum.
class Solution {
private:
    int res;
    
    //The return value will always be a vector with 3 elements, the first
    //one is the maximum value from the tree, the second one is the 
    //smallest from the tree. Since we do a post order traversal, we
    //need to always maintain the BST property for all the sub tree.
    vector<int> checkValidBST(TreeNode* node){
        //As for the empty node, we always denote the sum of the tree to be 0
        //Null node is always a valid BST
        if(!node) return vector<int>({INT_MIN, INT_MAX, 0});
        
        auto left = checkValidBST(node->left);
        auto right = checkValidBST(node->right);
        
        //With the node, we can not form a valid BST any more
        if(left.empty() || right.empty() || left[0] >= node->val || right[1] <= node->val)
            return vector<int>();
        
        //If we do have a BST
        int curSum = node->val + left[2] + right[2];
        res = max(res, curSum);
        
        //Note we have always maintain the correct return value. 
        //right[0] is the maximum value from the right sub tree
        //left[1] is the minimum value from the left sub tree
        return vector<int>({max(right[0], node->val), min(left[1], node->val), curSum});
        
    }
public:
    int maxSumBST(TreeNode* root) {
        res = 0;
        
        checkValidBST(root);
        //We have all negative nodes
        return max(0, res);
        
    }
};


//Note the above solution is slow because of too many vector construction.
//Same idea but much simper data structure
class Solution {
    int dfs(TreeNode *root) {
        if (!root) return 0;
        int l = dfs(root->left), r = dfs(root->right);
        if (l == INT_MIN || r == INT_MIN || root->left && root->left->val >= root->val
            || root->right && root->right->val <= root->val)
            return INT_MIN;
        int sum = l + r + root->val;
        mx = max(mx, sum);
        return sum;
    }
    int mx;
public:
    int maxSumBST(TreeNode* root) {
        mx = 0;
        dfs(root);
        return mx;
    }
};


// 1483. Kth Ancestor of a Tree Node
// https://leetcode.com/problems/kth-ancestor-of-a-tree-node/
// Very interesting idea
// Concept of binary lifting
// From: https://leetcode.com/problems/kth-ancestor-of-a-tree-node/discuss/686362/JavaC%2B%2BPython-Binary-Lifting
// Interesting staff, good to know staff
class TreeAncestor {
private:
    // For each entry, we save the parent of curent node. 
    // v[i][0] means the parent of ith node, v[i][1] means the 2^1th parent of i
    // v[i][j] means 2^jth parent node of i
    vector<vector<int>> v;
public:
    TreeAncestor(int n, vector<int>& parent) {
        vector<vector<int>> temp(n, vector<int>(20, -1));
        for(int i = 0; i < parent.size(); ++i){
            temp[i][0] = parent[i];
        }
        for(int k = 1; k < 20; ++k){
            for(int i = 0; i < n; ++i){
                if(temp[i][k-1] == -1) continue;
                temp[i][k] = temp[temp[i][k-1]][k-1];
            }
        }
        swap(v, temp);
    }
    
    int getKthAncestor(int node, int k) {
        // The most tricky part is node=temp[node][i]
        // Note whenever we right shift 1 bit, we are actually moving 2^i steps up. 
        // if we want to find next valid parent, which is v[node][i]
        //
        for(int i = 0; i < 20; ++i){
            if((k >> i) & 1){
                node = v[node][i];
                if(node == -1) return -1;
            }
        }
        return node;
    }
};

/**
 * Your TreeAncestor object will be instantiated and called as such:
 * TreeAncestor* obj = new TreeAncestor(n, parent);
 * int param_1 = obj->getKthAncestor(node,k);
 */