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

using namespace std;

//155. Min Stack
//https://leetcode.com/problems/min-stack/
/*
Not hard if we allocate two stacks, we can optimize the space 
a little bit if we only push elements to minstack when its value
is smaller than current minvalue.
*/

class MinStack {
private:
    stack<int> container;
    stack<int> minSt;
public:
    /** initialize your data structure here. */
    MinStack() {
    }
    void push(int x) {
        container.push(x);
        if(minSt.empty() || x <= minSt.top()){
            minSt.push(x);
        }
    }
    
    void pop() {
        int temp = container.top();
        container.pop();
        if(temp == minSt.top()){
            minSt.pop();
        }
    }
    
    int top() {
        return container.top();
    }
    
    int getMin() {
        return minSt.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */

//232. Implement Queue using Stacks
//https://leetcode.com/problems/implement-queue-using-stacks/
/*
Solution is trivial if we allow O(n) push or pop, we can simply allocate two stacks, 
and get the front element; Here we have amortized O(1) solution, a little bit tricky.
Please note how we handle two stacks separately for checking solution.
*/
class MyQueue {
private:
    stack<int> container;
    stack<int> revContainer;
    int front;
public:
    /** Initialize your data structure here. */
    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        if(container.empty()){
            front = x;
        }
        container.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        if(revContainer.empty()){
            while(!container.empty()){
                int temp = container.top();
                container.pop();
                revContainer.push(temp);
            }
            
        }
        int res = revContainer.top();
            revContainer.pop();
            return res;
    }
    
    /** Get the front element. */
    int peek() {
        if(!container.empty() && revContainer.empty())
            return front;
        else
            return revContainer.top();
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return container.empty() && revContainer.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */


//225. Implement Stack using Queues
//https://leetcode.com/problems/implement-stack-using-queues/
/*
Using two queue. Either push will be O(n) or pop will be O(n)
Understand one method, the other one will be trivial...
*/
class MyStack {
private:
    queue<int> Q;
    queue<int> extraQ;
    int topVal;
public:
    /** Initialize your data structure here. */
    MyStack() {
        
    }
    
    /** Push element x onto stack. */
    void push(int x) {
        topVal = x;
        Q.push(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int temp;
        while(!Q.empty()){
            temp = Q.front();
            Q.pop();
            if(!Q.empty()){
                extraQ.push(temp);
                if(Q.size() == 1){
                    topVal = temp;
                }
            }
        }
        swap(Q, extraQ);
        return temp;
    }
    
    /** Get the top element. */
    int top() {
        return topVal;
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        return Q.empty();
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */

//388. Longest Absolute File Path
//https://leetcode.com/problems/longest-absolute-file-path/
/*
Hash table solution is fast, a little bit tricky to get to this solution. The general idea is easy to follow,
we first define a map which records the depth and length associate with that depth, then we check whether we 
find a file in a specific layer, if we find a file, then we update maxLen, else , we keep adding new layer.
Imagine the hierarchy structure as a tree, we will always check one subtree before moving to other sub trees.

We could also implement this using pair and stack. Just mimic the depth first search progress. We check each 
path until we find a file, then we update maxLen, then we back to parent node, and search another pass...
*/
//C++ Hash map solution
//Convert string to istringstream may make the problem easier.
class Solution {
public:
    int lengthLongestPath(string input) {
        istringstream ss(input);
        int len = input.size();
        //First int represents folder depth, second means current path length
        unordered_map<int, int> dict;
        int maxLen = 0;
        string s("");
        //We need to initialize the map to represent root directory
        dict[0] = 0;
        //getline will extract each string to s based on delimiter. 
        //Default delimiter is '\n'
        while(getline(ss, s)){
            //find_last_of finds the last '\t' character, seems like 
            //'\t' only counts as one character... Takes up 1 size
            size_t sPos = s.find_last_of('\t');
            string name = (sPos != string::npos) ? s.substr(sPos+1) : s;
            int depth = s.size() - name.size();
            //We calculate current length for each depth (folder hierarchy)
            //If we find a file, we need to update maxLen
            if(s.find('.') != string::npos){
                int l = dict[depth] + name.size();
                maxLen = max(maxLen, l);
            }
            else{
                //Note we need to include '/' in each level
                dict[depth+1] = dict[depth] + name.size() + 1;
            }
        }
        return maxLen;
    }
};

//Stack solution
class Solution {
public:
    int lengthLongestPath(string input) {
        if(input.size() == 0) return 0;
        istringstream ss(input);
        vector<pair<string, int>> container;
        string s("");
        int maxLen = 0;
        //Extract each string segments and its corresponding level to container.
        //Pay attention to how to using istringstream class
        while(getline(ss, s)){
            size_t p = s.find_last_of('\t');
            string name = (p == string::npos) ? s : s.substr(p+1);
            int level = s.size() - name.size();
            container.emplace_back(make_pair(name, level));
        }
        
        //Keep track of current length we already have
        int cur = 0;
        stack<pair<string, int>> st;
        //We use stack to keep track of each level (DFS), whenever we find leafs
        //We traceback cur value, whenever we push one element, we increase cur
        //by the string size. Whenever we find a file, we update maxLen.
        for(auto element : container){
            while(!st.empty() && st.top().second + 1 != element.second){
                cur -= (st.top().first.size() + 1);
                st.pop();
                
            }
            if(element.first.find('.') != string::npos){
                int l = cur + element.first.size();
                maxLen = max(maxLen, l);
            }
            else{
                st.push(element);
                cur += element.first.size() + 1;
            }
        }
        return maxLen;
    }
};
