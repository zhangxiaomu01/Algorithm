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

/*
    150. Evaluate Reverse Polish Notation
    https://leetcode.com/problems/evaluate-reverse-polish-notation/
 
    You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.

    Evaluate the expression. Return an integer that represents the value of the expression.

    Note that:

    The valid operators are '+', '-', '*', and '/'.
    Each operand may be an integer or another expression.
    The division between two integers always truncates toward zero.
    There will not be any division by zero.
    The input represents a valid arithmetic expression in a reverse polish notation.
    The answer and all the intermediate calculations can be represented in a 32-bit integer.
    

    Example 1:
    Input: tokens = ["2","1","+","3","*"]
    Output: 9
    Explanation: ((2 + 1) * 3) = 9

    Example 2:
    Input: tokens = ["4","13","5","/","+"]
    Output: 6
    Explanation: (4 + (13 / 5)) = 6

    Example 3:
    Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
    Output: 22
    Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
    = ((10 * (6 / (12 * -11))) + 17) + 5
    = ((10 * (6 / -132)) + 17) + 5
    = ((10 * 0) + 17) + 5
    = (0 + 17) + 5
    = 17 + 5
    = 22
    

    Constraints:
    1 <= tokens.length <= 104
    tokens[i] is either an operator: "+", "-", "*", or "/", or an integer in the range [-200, 200].
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
class Solution {
private:
    bool isOperator(string& s) {
        return s == "+" || s == "*" || s == "-" || s == "/";
    }
public:
    int evalRPN(vector<string>& tokens) {
        vector<int> process;
        for (string& e : tokens) {
            if (isOperator(e)) {
                int temp = 0;
                if (e == "+") temp = process.back() + process[process.size() - 2];
                if (e == "-") temp = process[process.size() - 2] - process.back();
                if (e == "*") temp = process[process.size() - 2] * process.back();
                if (e == "/") temp = process[process.size() - 2] / process.back();
                process.pop_back();
                process.pop_back();
                process.emplace_back(temp);
            } else {
                process.emplace_back(stoi(e));
            }
        }
        return process[0];
    }
};

/*
    239. Sliding Window Maximum
    https://leetcode.com/problems/sliding-window-maximum/
 
    You are given an array of integers nums, there is a sliding window of size k which is 
    moving from the very left of the array to the very right. You can only see the k numbers 
    in the window. Each time the sliding window moves right by one position.
    Return the max sliding window.

    

    Example 1:
    Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
    Output: [3,3,5,5,6,7]
    Explanation: 
    Window position                Max
    ---------------               -----
    [1  3  -1] -3  5  3  6  7       3
    1 [3  -1  -3] 5  3  6  7       3
    1  3 [-1  -3  5] 3  6  7       5
    1  3  -1 [-3  5  3] 6  7       5
    1  3  -1  -3 [5  3  6] 7       6
    1  3  -1  -3  5 [3  6  7]      7

    Example 2:
    Input: nums = [1], k = 1
    Output: [1]
    

    Constraints:
    1 <= nums.length <= 105
    -104 <= nums[i] <= 104
    1 <= k <= nums.length
 */
// Maintain a maximum queue when sliding.
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        // We need to maintain a sliding maximum queue during the iteration.
        deque<int> maxQueue;
        vector<int> res;
        maxQueue.push_back(nums[0]);
        if (k == 1) {
            return nums;
        }
        for (int i = 1 ; i < nums.size(); ++i) {
            // Always saves the maximum value in the queue
            // Note: we need to check from the back, so we can get rid of any intermediate 
            // previous next maximum value.
            while (!maxQueue.empty() && maxQueue.back() < nums[i]) {
                maxQueue.pop_back();
            }
            maxQueue.push_back(nums[i]);

            if (i >= k - 1) {
                res.push_back(maxQueue.front());
            }
            // The sliding window is longer than k and our maximum is equal to the first element
            if (i >= k - 1 && maxQueue.front() == nums[i - (k-1)]) {
                maxQueue.pop_front();
            }
        }
        return res;
    }
};

// A better implementation: define a specific maximum queue!
// A data structure which maintains the running maximum during iteration.
class maximumQueue {
private:
    deque<int> dq;

public:
    // Pushes the `val` to the queue, removes element smaller than val from the back before pushing.
    void push(int val) {
        while (!dq.empty() && dq.back() < val) dq.pop_back();
        dq.push_back(val);
    }

    // Pops the front of queue if the value equals to `val`.
    void pop(int val) {
        if (!dq.empty() && dq.front() == val) dq.pop_front();
    }

    // Returns the maximum value in the queue.
    int top() {
        if (!dq.empty()) return dq.front();
        return INT_MIN;
    }
};


class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        maximumQueue mQ;
        vector<int> res;
        for (int i = 0; i < k; ++i) mQ.push(nums[i]);
        res.push_back(mQ.top());

        for (int i = k; i < nums.size(); ++i) {
            mQ.pop(nums[i-k]);
            mQ.push(nums[i]);
            res.push_back(mQ.top());
        }
        return res;
    }
};

/*
    347. Top K Frequent Elements
    https://leetcode.com/problems/top-k-frequent-elements/
 
    Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

    Example 1:
    Input: nums = [1,1,1,2,2,3], k = 2
    Output: [1,2]

    Example 2:
    Input: nums = [1], k = 1
    Output: [1]
    

    Constraints:
    1 <= nums.length <= 105
    -104 <= nums[i] <= 104
    k is in the range [1, the number of unique elements in the array].
    It is guaranteed that the answer is unique.
    

    Follow up: Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
 */
class Solution {
private:
    // maintain the minimum priority queue.
    struct comp {
        public:
            bool operator()(pair<int, int>& l, pair<int, int>& r) {
                return l.second > r.second;
            }
    };

public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> uMap;
        priority_queue<pair<int, int>, vector<pair<int, int>>, comp> pq;
        vector<int> res;
        for (int i : nums) uMap[i] ++;

        for (auto i = uMap.begin(); i != uMap.end(); ++i) {
            pq.push( {i->first, i->second} );
            if (pq.size() > k) pq.pop();
        }

        for (int i = 0; i < k; ++i) {
            res.push_back(pq.top().first);
            pq.pop();
        }

        return res;
    }
};

/*
    71. Simplify Path
    https://leetcode.com/problems/simplify-path/
 
    Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.
    In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'. For this problem, any other format of periods such as '...' are treated as file/directory names.
    The canonical path should have the following format:

    The path starts with a single slash '/'.
    Any two directories are separated by a single slash '/'.
    The path does not end with a trailing '/'.
    The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')
    Return the simplified canonical path.


    Example 1:
    Input: path = "/home/"
    Output: "/home"
    Explanation: Note that there is no trailing slash after the last directory name.

    Example 2:
    Input: path = "/../"
    Output: "/"
    Explanation: Going one level up from the root directory is a no-op, as the root level is the highest level you can go.

    Example 3:
    Input: path = "/home//foo/"
    Output: "/home/foo"
    Explanation: In the canonical path, multiple consecutive slashes are replaced by a single one.
    

    Constraints:
    1 <= path.length <= 3000
    path consists of English letters, digits, period '.', slash '/' or '_'.
    path is a valid absolute Unix path.
 */
class Solution {
public:
    string simplifyPath(string path) {
        //Note utilize string stream can greatly improve the efficiency,
        //since we can extrack each segment by a specified delimiter
        stringstream ss(path);
        vector<string> st;
        string temp;
        //Note we will discard delimiter '/' when extract characters
        //We continue when we encounter "." or "//"
        while(getline(ss, temp, '/')) {
            if (temp == "" || temp == ".") continue;
            if (temp == ".." && !st.empty()) st.pop_back();
            else if (temp != "..") st.emplace_back(temp);
        }

        string res;
        for (string s : st) {
            res.push_back('/');
            res.append(s);
        }

        return res.empty() ? "/" : res;
    }
};

