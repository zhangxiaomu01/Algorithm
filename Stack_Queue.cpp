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
            int res = revContainer.top();
            revContainer.pop();
            return res;
        }
    }
    
    /** Get the front element. */
    int peek() {
        if(!container.empty() && revContainer.empty())
            return front;
        if(container.empty() && !revContainer.empty())
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

