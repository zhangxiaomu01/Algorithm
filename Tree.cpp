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
                //We need to update cur. because currentlt it's nullptr
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

