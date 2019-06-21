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
