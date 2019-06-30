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

//145. Binary Tree Postorder Traversal
//https://leetcode.com/problems/binary-tree-postorder-traversal/
/*
Basically, we juest reverse the order of preorder traversal. Note in preorder, we have root->left->right. It's easy to calculate. However, we can first calculate root->right->left. Then reverse the order of the output.
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
Note we are actually swapping the left and right children level by level. This holds true for both recursive and iterative version.
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
private:
    int dfs(TreeNode* node){
        if(!node) return 0;
        if(!node->left) return dfs(node->right) + 1;
        if(!node->right) return dfs(node->left) + 1;
        return max(dfs(node->left), dfs(node->right))+1;
    }
public:
    int maxDepth(TreeNode* root) {
        return dfs(root);
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
        //Note we only need to return the maximum path of the two, then we can from the valid path
        return max(left, right) + node->val;
    }
public:
    int maxPathSum(TreeNode* root) {
        rec(root);
        return maxVal;
    }
};




