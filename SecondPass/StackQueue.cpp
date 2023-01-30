/**
 * @file StackQueue.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2023-01-29
 * 
 * A quick second pass with the common 'Stack & Queue' related problems.
 * 
 */
 /*
    232. Implement Queue using Stacks
    https://leetcode.com/problems/implement-queue-using-stacks/
 
    Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

    Implement the MyQueue class:

    void push(int x) Pushes element x to the back of the queue.
    int pop() Removes the element from the front of the queue and returns it.
    int peek() Returns the element at the front of the queue.
    boolean empty() Returns true if the queue is empty, false otherwise.
    Notes:
    You must use only standard operations of a stack, which means only push to top, peek/pop from top, size, and is empty operations are valid.
    Depending on your language, the stack may not be supported natively. You may simulate a stack using a list or deque (double-ended queue) as long as you use only a stack's standard operations.
    

    Example 1:
    Input
    ["MyQueue", "push", "push", "peek", "pop", "empty"]
    [[], [1], [2], [], [], []]
    Output
    [null, null, null, 1, 1, false]

    Explanation
    MyQueue myQueue = new MyQueue();
    myQueue.push(1); // queue is: [1]
    myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
    myQueue.peek(); // return 1
    myQueue.pop(); // return 1, queue is [2]
    myQueue.empty(); // return false
    

    Constraints:
    1 <= x <= 9
    At most 100 calls will be made to push, pop, peek, and empty.
    All the calls to pop and peek are valid.

    Follow-up: Can you implement the queue such that each operation is amortized O(1) time complexity? In other words, performing n operations will take overall O(n) time even if one of those operations may take longer.
 */
/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
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

/*
    225. Implement Stack using Queues
    https://leetcode.com/problems/implement-stack-using-queues/
 
    Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

    Implement the MyStack class:
    void push(int x) Pushes element x to the top of the stack.
    int pop() Removes the element on the top of the stack and returns it.
    int top() Returns the element on the top of the stack.
    boolean empty() Returns true if the stack is empty, false otherwise.
    Notes:
    You must use only standard operations of a queue, which means that only push to back, peek/pop from front, size and is empty operations are valid.
    Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue) as long as you use only a queue's standard operations.
    

    Example 1:
    Input
    ["MyStack", "push", "push", "top", "pop", "empty"]
    [[], [1], [2], [], [], []]
    Output
    [null, null, null, 2, 2, false]

    Explanation
    MyStack myStack = new MyStack();
    myStack.push(1);
    myStack.push(2);
    myStack.top(); // return 2
    myStack.pop(); // return 2
    myStack.empty(); // return False
    

    Constraints:
    1 <= x <= 9
    At most 100 calls will be made to push, pop, top, and empty.
    All the calls to pop and top are valid.
    
    Follow-up: Can you implement the stack using only one queue?
 */
/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */
class MyStack {
private:
    queue<int> myQueue;
public:
    MyStack() {
        
    }
    
    void push(int x) {
        myQueue.push(x);
    }
    
    int pop() {
        int count = myQueue.size() - 1;
        while(count-- > 0) {
            int front = myQueue.front();
            myQueue.pop();
            myQueue.push(front);
        } 
        int res = myQueue.front();
        myQueue.pop();
        return res;
    }
    
    int top() {
        return myQueue.back();
    }
    
    bool empty() {
        return myQueue.empty();
    }
};

/*
    20. Valid Parentheses
    https://leetcode.com/problems/valid-parentheses/
 
    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

    An input string is valid if:
    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.
    Every close bracket has a corresponding open bracket of the same type.
    

    Example 1:
    Input: s = "()"
    Output: true

    Example 2:
    Input: s = "()[]{}"
    Output: true

    Example 3:
    Input: s = "(]"
    Output: false
    

    Constraints:
    1 <= s.length <= 104
    s consists of parentheses only '()[]{}'.
 */
class Solution {
public:
    bool isValid(string s) {
        // Creating a hashmap here will make code much cleaner!
        unordered_map<char,char> hash = {{')','('},{'}','{'},{']','['}};
        vector<char> pStack;
        int len = s.size();
        for(int i = 0; i< len; i++)
        {
            if(s[i] == '(' || s[i] == '{'||s[i] == '[')
                pStack.push_back(s[i]);
            if(s[i] == ')' || s[i] == '}'||s[i] == ']'){
                if(pStack.empty() || pStack.back() != hash[s[i]])
                    return false;
                else
                    pStack.pop_back();
            }                     
        }
        return pStack.empty();
    }
};

/*
    1047. Remove All Adjacent Duplicates In String
    https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/
 
    You are given a string s consisting of lowercase English letters. A duplicate removal consists of choosing two adjacent and equal letters and removing them.
    We repeatedly make duplicate removals on s until we no longer can.
    Return the final string after all such duplicate removals have been made. It can be proven that the answer is unique.

    
    Example 1:
    Input: s = "abbaca"
    Output: "ca"
    Explanation: 
    For example, in "abbaca" we could remove "bb" since the letters are adjacent and equal, and this is the only possible move.  The result of this move is that the string is "aaca", of which only "aa" is possible, so the final string is "ca".
    Example 2:
    Input: s = "azxxzy"
    Output: "ay"
    

    Constraints:
    1 <= s.length <= 105
    s consists of lowercase English letters.
 */
class Solution {
public:
    string removeDuplicates(string S) {
        string ans = "";
        for (const char &c : S) {
            if (ans.back() == c) ans.pop_back();
            else ans.push_back(c);
        }
        return ans;
    }
};