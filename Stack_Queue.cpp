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

//394. Decode String
//https://leetcode.com/problems/decode-string/
/*
Stack implementation: Note the tricky part is two stacks represents different level:
"3[a2[c]]"
We push 3 and "" in to level 1
then we push 2 and "a" into level 2
So when we meet first "]", we can get 2 from stack num (top), and complete the extension of "c", 
and concatenate back to "a" in stack st. Then we pop 2 and "a" out of stack.
Next we can pop 3 from num, and complete the extension of "acc", and apend the empty
string with "accaccacc" to res, then we get the solution.
Not so straight forward. May come back later.
*/
class Solution {
public:
    string decodeString(string s) {
        stack<int> num;
        stack<string> st;
        int len = s.size();
        if(len == 0) return "";
        int tempNum = 0;
        string res("");
        //Iterate the string, and handle different situation accordingly
        for(int i = 0; i < len; i++){
            if(s[i] == '['){
                num.push(tempNum);
                st.push(res);
                tempNum = 0;
                res = "";
            }
            else if(s[i] >= '0' && s[i] <= '9'){
                tempNum = tempNum*10 + (s[i] - '0');
            }
            else if(isalpha(s[i])){
                tempNum = 0;
                res.push_back(s[i]);
            }
            //The tricky part is we use res to store current repetitive string.
            //Then we add st.top() to res, this gives us the complete string in one [] 
            else if(s[i] == ']'){
                int rep = num.top();
                string temp = res;
                for(int j = 0; j < rep-1; j++){
                    res += temp;
                }
                if(!st.empty()){
                    res = st.top() + res;
                }
                num.pop();
                st.pop();
            } 
        }
        return res;
    }
};

/*
Recursive version: More simple and elegant, however, hard to get to the right insight... Basically,
the stack version is mimic the recursive calls 
*/
class Solution {
public:
    string decodeString(string s) {
        int i = 0;
        return doDecode(s, i);
    }
    //Note i passes by reference...
    string doDecode(const string& s, int& i){
        string res;
        
        while(i < s.size() && s[i] != ']'){
            if(isalpha(s[i])){
                res.push_back(s[i++]);
            }
            else if(isdigit(s[i])){
                int num = 0;
                //shouldn't put ++ to while isdigit[i++], we need to update i after we compute the num
                while(i < s.size() && isdigit(s[i])){
                    num = num*10 + (s[i++] - '0');
                }
                i++;//skip '['
                string t = doDecode(s, i);
                i++;//skip ']'
                
                for(int j = 0; j < num; j++)
                    res += t;
            }//All other symbols, we need to move forward
            else
                i++;
        }
        return res;
    }
};

//224. Basic Calculator
//https://leetcode.com/problems/basic-calculator/
/*
Not that hard. Using stack to mimic the parenthesis operation.
*/
class Solution {
public:
    int calculate(string s) {
        int num = 0;
        int sign = 1;
        int res = 0;
        stack<int> digits, signs;
        for(char c:s){
            if(isdigit(c)){
                num = num * 10 + (c - '0');
            }
            else{
                res = res + sign*num;
                num = 0;
                if(c == '+') sign = 1;
                else if(c == '-') sign = -1;
                else if(c == '('){
                    digits.push(res);
                    signs.push(sign);
                    res = 0;
                    sign = 1;
                }
                else if(c == ')'){
                    res = signs.top() * res + digits.top();
                    digits.pop();
                    signs.pop();
                }
            }
        }
        //Note we need to add the final num to result
        res = res + sign*num;
        return res;
    }
};

//227. Basic Calculator II
//https://leetcode.com/problems/basic-calculator-ii/
/*
A little bit tricky... 
*/
//The general idea is to calculate multiplication and division first,
//then add all the results together...
class Solution {
private:
    void doOperation(stack<int>& digit, char op, int num){
        if(op == '+')
            digit.push(num);
        else if(op == '-')
            digit.push(-num);
        else if(op == '*'){
            int temp = num * digit.top();
            digit.top() = temp;
        }
        else if(op == '/'){
            int temp = digit.top()/num;
            digit.top() = temp;
        }
    }

public:
    int calculate(string s) {
        char op = '+';
        int num = 0;
        int res = 0;
        stack<int> digit;
        for(char c:s){
            if(isdigit(c)){
                num = num*10 + (c - '0');
            }
            
            if(!isdigit(c) && !isspace(c)){
                doOperation(digit, op, num);
                num = 0;
                //We need to update op every iteration, note that op always 
                //comes after the two operands 
                op = c;
            }
        }
        doOperation(digit, op, num);
        //Add all the numbers together, since no * or / exists
        while(!digit.empty()){
            res += digit.top();
            digit.pop();
        }
        return res;
    }
};