/* Design pattern principles:
This is one of the most important, as well as overlooked, topics in software 
engineering.  Tech focuses a lot more on the "Patterns" during this talk than 
on "Principles".  These software design patterns offer ways to accomplish certain 
behaviors, within a program, and also serve to benefit programmer communication.  
They all stand upon and are justified by principles.

I think the most important principles are the following:

D.R.Y. - Don't Repeat Yourself - Pretty much self-explanatory.  If you are 
re-writing the same block(s) of code in various places then you are ensuring that 
you'll have multiple places to change it later.  This is probably the #1 warning 
flag that you should look into an alternative/pattern.

Y.A.G.N.I - You Ain't Gonna Need It - Don't code out every little detail that you 
can possibly think of because, in reality, you probably won't end up needing much 
of it.  So just implement what is absolutely necessary (save the GUI until later, 
for example!)

K.I.S.S. - Keep It Stupid Simple - When in doubt, make it easy to understand.  
Don't try to build in too much complexity.  If a particular design pattern overly 
complicates things (vs. an alternative or none at all) then don't implement it.  
This applies heavily to user interface design and APIs (application programming 
interfaces).

P.O.L.A. - Principle Of Least Astonishment - This has overlap with KISS...  
Do the least astonishing thing.  In other words: DON'T BE CLEVER.  It's nice that 
you can squeeze a ton of detail into a single line of code.  But your future self 
and other developers will judge it by the number of "F"-words per minute when they 
open your source file.  It should be relatively easy to jump into what your code is 
doing, even for an outsider who isn't quite as clever as you are.

Last, and most certainly NOT least...

S.O.L.I.D.  (I saved this for last because it's more specific, but it should 
probably land around #2 on this list)

Single Responsibility - A class should have only one reason to change;  Do one 
thing and do it well.
Open/Closed Principle - A class should be open for "extension" but closed for 
"modification".
L. Substitution Principle - Derived (sub) classes should fit anywhere that their 
base (super) class does.
Interface Segregation - Multiple specific interfaces are better than one 
general-purpose interface.
Dependency Inversion - Depend on abstractions NOT on concrete implementations.  
(Depend on abstract classes and interfaces, which can't be instantiated, rather 
then any specific instance)

1. @3:00 Break programs into logically independent components, allowing component 
dependencies where necessary. 
2. @3:10 Inspect your data objects and graph their interactions/where they flow. 
Keep classes simple. Keep one-way dataflow/interactions simple. 
@3:29 Simplicity -- keep a small number of class types (controller, views, data 
objects) avoid creating classes that aliases those classes (managers, coordinators, 
helpers, handlers, providers, executor). @4:02 -- Keep dataflow simple (e.g., don't 
let Views have business logic; views shouldn't need to communicate any data upstream). 
3. @4:33 Keep data model objects pruned of logic, except if the logic is part of the
 objects' states. Single Responsibility Principle. If you can't describe a class's 
 responsibility straightforwardly, it's too complex. 
4. @5:25 Inspect class composition ("follow ownership graph"). Follow the life of a 
child (composed) object to make sure that accessibility to and operations done on 
that object is managed. 
5. @5:55 Singletons ("basically globals floating in a system"). Use dependency 
injection (?) to 'scope singletons' and make them testable (?). Singletons often 
represent independent (uncoupled) objects which is good for representing independent
 execution flow. Too many inter-communicating singletons in a system make it 
 difficult to trace data flow and understand the overall transformation of the data 
 through the system. 
6. @6:50 Singleton communication patterns. Singletons expose Publisher/Subscriber 
interface (one publisher object, many subscriber objects listening to events). 
7. @7:41 Delegate communication pattern. (?)
8.@8:02 Chain of Responsibility Pattern. Hierarchical data flow -- unhandled events 
bubble upwards to parent objects who should handle them. Conflicts with (2) by 
allowing upstream dataflow. Use with discretion. 
9. @8:53 OOP's Inheritance can create tightly coupled hierarchical objects that are 
hard to refactor (object dependencies). Use composition pattern to allow flexibility
 in what the container object can do. 
10. @9:58 Lazy initialization. Startup performance boost. 
11. @10:06 Adapter Pattern. 
12. @10:26 Factory builder classes.

*/

//146. LRU Cache
//https://leetcode.com/problems/lru-cache/
/* Not memory efficient, can optimize a little bit
 The time complexity should be the same with other 
 approach. The insight to use map to keep track of 
 the last used element iterator is the key to solve
 this problem.*/
class LRUCache {
private:
    int m_Cap;
    //list to keep track of the least used items
    //the first item is the last updated (used)
    list<int> LRU;
    //keep track of the position of each key in LRU list
    unordered_map<int, list<int>::iterator> keyMap;
    //Build the key-value mapping, easy to retrieve 
    //the value
    unordered_map<int, int> dict;

    void updateKey(int key){
        auto it = keyMap[key];
        LRU.erase(it);
        LRU.push_front(key);
        //mark the key as the least used
        keyMap[key] = LRU.begin();
    }
    //Erase the key from dict and keyMap
    void evictKey(){
        dict.erase(LRU.back());
        keyMap.erase(LRU.back());
        LRU.pop_back();
    }
    
public:
    LRUCache(int capacity) : m_Cap(capacity) {
        
    }

    int get(int key) {
        int val = -1;
        if(dict.count(key) > 0){
            val = dict[key];
            updateKey(key);
        }
        return val;
    }
    
    void put(int key, int value) {
        //We need to insert a new key and value, however, we reach
        //the capacity limit
        if(LRU.size() == m_Cap && dict.count(key) == 0){
            evictKey();
        }
        if(dict.count(key) == 0){
            LRU.push_front(key);
            keyMap[key] = LRU.begin();
            dict[key] = value;
        }else{
            dict[key] = value;
            updateKey(key);
        }
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */


//284. Peeking Iterator
//https://leetcode.com/problems/peeking-iterator/
/* Solution 1 use the copy constructor. Solution 2 is more generic! */
//Solution 1
// Below is the interface for Iterator, which is already defined for you.
// **DO NOT** modify the interface for Iterator.

class Iterator {
    struct Data;
	Data* data;
public:
	Iterator(const vector<int>& nums);
	Iterator(const Iterator& iter);
	virtual ~Iterator();
	// Returns the next element in the iteration.
	int next();
	// Returns true if the iteration has more elements.
	bool hasNext() const;
};


class PeekingIterator : public Iterator {
public:
	PeekingIterator(const vector<int>& nums) : Iterator(nums) {
	    // Initialize  any member here.
	    // **DO NOT** save a copy of nums and manipulate it directly.
	    // You should only use the Iterator interface methods.
	    
	}

    // Returns the next element in the iteration without advancing the iterator.
	int peek() {
        //Copy the iterator object and return the next
        return Iterator(*this).next();
	}

	// hasNext() and next() should behave the same as in the Iterator interface.
	// Override them if needed.
	int next() {
	    return Iterator::next();
	}

	bool hasNext() const {
	    return Iterator::hasNext();
	}
};

//Solution 2
class PeekingIterator : public Iterator {
private:
    int m_next;
    int m_hasNext;
public:
	PeekingIterator(const vector<int>& nums) : Iterator(nums) {
	    // Initialize  any member here.
	    // **DO NOT** save a copy of nums and manipulate it directly.
	    // You should only use the Iterator interface methods.
	    m_hasNext = Iterator::hasNext();
        if(m_hasNext) m_next = Iterator::next();
	}

    // Returns the next element in the iteration without advancing the iterator.
	int peek() {
        //Copy the iterator object and return the next
        return m_next;
	}

	// hasNext() and next() should behave the same as in the Iterator interface.
	// Override them if needed.
	int next() {
        int t = m_next;
        m_hasNext = Iterator::hasNext();
        if(m_hasNext) m_next = Iterator::next();
	    return t;
	}

	bool hasNext() const {
	    return m_hasNext;
	}
};

//341. Flatten Nested List Iterator
//https://leetcode.com/problems/flatten-nested-list-iterator/
/* Use stack to unroll the nested list is key to success. */
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * class NestedInteger {
 *   public:
 *     // Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool isInteger() const;
 *
 *     // Return the single integer that this NestedInteger holds, if it holds a single integer
 *     // The result is undefined if this NestedInteger holds a nested list
 *     int getInteger() const;
 *
 *     // Return the nested list that this NestedInteger holds, if it holds a nested list
 *     // The result is undefined if this NestedInteger holds a single integer
 *     const vector<NestedInteger> &getList() const;
 * };
 */
class NestedIterator {
private:
    stack<NestedInteger*> st;
public:
    NestedIterator(vector<NestedInteger> &nestedList) {
        //Unroll the first level
        int len = nestedList.size();
        for(int i = len-1; i >= 0; --i){
            st.push(&nestedList[i]);
        }
    }

    int next() {
        //We are guranteed to call the hasNext() first,
        //so st.top() will always have integer (if not empty).
        int res = st.top()->getInteger();
        st.pop();
        return res;
    }

    bool hasNext() {
        if(st.empty()) return false;
        //While loop is necessary here, because we could have the
        //case [[],[],[1, []],2]
        while(!st.empty()){
           NestedInteger* p = st.top();
            if(p->isInteger()){
                return true;
            }
            st.pop();

            vector<NestedInteger>& iList = p->getList();
            int len = iList.size();
            for(int i = len-1; i >= 0; --i){
                st.push(&iList[i]);
            } 
        }
        return false;
    }
};

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i(nestedList);
 * while (i.hasNext()) cout << i.next();
 */


//295. Find Median from Data Stream
//https://leetcode.com/problems/find-median-from-data-stream/
/* Insertion sort Solution (O(n)) */
class MedianFinder {
private:
    vector<int> Con;
public:
    /** initialize your data structure here. */
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        auto it = lower_bound(Con.begin(), Con.end(), num);
        Con.insert(it, num);
    }
    
    double findMedian() {
        int len = Con.size();
        if(len == 0) return 0;
        else return len % 1 ? Con[len/2] : (Con[(len-1)/2] + Con[len/2]) * 0.5;
    }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */

/* Priority queue solution O(log n) */
class MedianFinder {
private:
    //The top element will be the largest of the queue
    priority_queue<int> maxQ;
    //The top element will be the smallest. We save larger half 
    //numbers in this queue, so the top will be close to median
    priority_queue<int, vector<int>, greater<int>> minQ;
public:
    /** initialize your data structure here. */
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        maxQ.push(num);
        //Balancing value, this step is critical!
        minQ.push(maxQ.top());
        maxQ.pop();
        //We always make maxQ have equal or more number of elements than minQ
        if(maxQ.size() < minQ.size()){
            maxQ.push(minQ.top());
            minQ.pop();
        }
        
    }
    
    double findMedian() {
        int len = maxQ.size() + minQ.size();
        if(len == 0) return 0;
        if(len & 1)
            return maxQ.top();
        else 
            return (minQ.top() + maxQ.top()) * 0.5;
    }
};

//Multiset solution. Easy to get the idea. 
//Very tricky when we handle the repetitive elements in the array.
//Pay attention to when length is even, how to handle the case.
//Still not very efficient
class MedianFinder {
private:
    multiset<int> mSet;
    multiset<int>::iterator loIt, hiIt;
public:
    /** initialize your data structure here. */
    MedianFinder() : loIt(mSet.end()), hiIt(mSet.end()) {
        
    }
    
    void addNum(int num) {
        //Get the length before inserting element
        int len = mSet.size();
        //When len is odd, after insert one element, the len will
        //be even.
        mSet.insert(num);
        
        if(len == 0){
            loIt = mSet.begin();
            hiIt = mSet.begin();
            return;
        }
        
        if(len & 1){
            if(num < *loIt)
                loIt--;
            else
                hiIt++;
        }else{
            //Note C++ will insert the new repetitive element in the 
            //end of all repetitive elements
            if(num > *loIt && num < *hiIt){
                loIt++;
                hiIt--;
            }
            else if(num >= *hiIt)
                loIt ++;
            else // num <= *loIt < *hiIt
                loIt = --hiIt; //insertion at the end of equal range spoils loIt
            //so we need loIt = --hiIt, instead of just hiIt--
                
        }
    }
    double findMedian() {
        if(loIt == mSet.end() && hiIt == mSet.end())
            return -1;
        return (*loIt + *hiIt) / 2.0;
    }
};


//352. Data Stream as Disjoint Intervals
//https://leetcode.com/problems/data-stream-as-disjoint-intervals/
/* The general idea is to use a set to efficiently remove and insert element.
This code is not the most efficient one, but it's easy to understand. O(nlogn)
Note the iterator to the set container follows the sorting order, which means 
the smallest element will be visited first. Then we can change the linear iteration
to binary search.*/
class SummaryRanges {
private:
    //Save all the potential intervals
    set<vector<int>> Intervals;
public:
    /** Initialize your data structure here. */
    SummaryRanges() {
        
    }
    
    void addNum(int val) {
        vector<int> newInterval = {val, val};
        for(auto it = Intervals.begin(); it != Intervals.end();){
            if(newInterval.back() + 1 < (it -> front()) || newInterval.front() - 1 > (it->back()))
                it++;
            else{
                //update newIntervals and erase the it object
                newInterval.front() = min(newInterval.front(), it->front());
                newInterval.back() = max(newInterval.back(), it->back());
                //We need to update it to the next element after the erased element
                it = Intervals.erase(it);
            }
        }
        Intervals.insert(newInterval);
    }
    
    vector<vector<int>> getIntervals() {
        return vector<vector<int>>(Intervals.begin(), Intervals.end());
    }
};

/**
 * Your SummaryRanges object will be instantiated and called as such:
 * SummaryRanges* obj = new SummaryRanges();
 * obj->addNum(val);
 * vector<vector<int>> param_2 = obj->getIntervals();
 */


//303. Range Sum Query - Immutable
//https://leetcode.com/problems/range-sum-query-immutable/
class NumArray {
    vector<int> prefixSum;
public:
    NumArray(vector<int>& nums) {
        if(nums.empty()) return;
        prefixSum.push_back(0);
        for(int i = 0; i < nums.size(); ++i){
            prefixSum.push_back(nums[i] + prefixSum.back());
        }
    }
    
    int sumRange(int i, int j) {
        return prefixSum[j+1] - prefixSum[i];
    }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * int param_1 = obj->sumRange(i,j);
 */


//307. Range Sum Query - Mutable
//https://leetcode.com/problems/range-sum-query-mutable/
/* deferred update approach, too slow. O(n) */
class NumArray {
private:
    unordered_map<int, int> updateDict;
    vector<int> prefixSum;
    vector<int*> updateNums;
public:
    NumArray(vector<int>& nums) {
        prefixSum.push_back(0);
        if(nums.empty()) return;
        int len = nums.size();
        for(int i = 0; i < len; ++i){
            prefixSum.push_back(nums[i] + prefixSum.back());
            updateNums.push_back(&nums[i]);
        }
    }
    
    void update(int i, int val) {
        updateDict[i] = val;
    }
    
    int sumRange(int i, int j) {
        int cumulativeSum = 0;
        for(int k = 0; k < updateNums.size(); ++k){
            if(updateDict.count(k) > 0){
                cumulativeSum += updateDict[k] - (*updateNums[k]); 
                *updateNums[k] = updateDict[k];
                updateDict.erase(k);
            }
            prefixSum[k+1] += cumulativeSum;
        }
        return prefixSum[j+1] - prefixSum[i];
    }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * obj->update(i,val);
 * int param_2 = obj->sumRange(i,j);
 */


/* Segment tree implementation: Very intresting!
A detailed explanation: 
https://leetcode.com/articles/a-recursive-approach-to-segment-trees-range-sum-queries-lazy-propagation/ */
class NumArray {
private:
    int* m_segTree;
    int m_len;
    void buildTree(vector<int>& nums){
        //Calculate leaves
        for(int i = m_len, j = 0; i < 2 * m_len; ++i, ++j){
            m_segTree[i] = nums[j];
        }
        for(int i = m_len-1; i >= 0; --i){
            m_segTree[i] = m_segTree[2*i] + m_segTree[2*i+1];
        }
    }
public:
    NumArray(vector<int>& nums) {
        m_len = nums.size();
        if(m_len > 0){
            m_segTree = new int[2 * m_len];
            buildTree(nums);
        }
    }
    
    //We will start updating leaf node and then go up until the 
    //root
    void update(int i, int val) {
        int pos = i + m_len;
        m_segTree[pos] = val;
        while(pos > 0){
            int left = pos;
            int right = pos;
            //Not easy to get this part right!
            if(pos % 2 == 0){
                right = pos + 1;
            }else
                left = pos - 1;
            m_segTree[pos/2] = m_segTree[left] + m_segTree[right];
            pos = pos/2;
        }
        
    }
    
    int sumRange(int i, int j) {
        //Get the leaf nodes of the range
        int posL = i + m_len;
        int posR = j + m_len;
        
        int sum = 0;
        while(posL <= posR){
            //left range is located in the right sub tree.
            //We need to include node posL to the sum instead
            //of its parent node
            if(posL % 2 == 1){
                sum += m_segTree[posL];
                posL ++;
            }
            //right range is located in the left sub tree
            //We need to include node posR to the sum instead
            //of its parent node
            if(posR % 2 == 0){
                sum += m_segTree[posR];
                posR --;
            }
            //When L == R, one of the above condition will meet
            //So we are guaranteed to break the loop
            posL /= 2;
            posR /= 2;
        }
        
        return sum;
    }
};

/* Binary Indexed Tree Solution: Good explanation */
//https://leetcode.com/problems/range-sum-query-mutable/discuss/75753/Java-using-Binary-Indexed-Tree-with-clear-explanation
//https://www.topcoder.com/community/competitive-programming/tutorials/binary-indexed-trees/


//382. Linked List Random Node
//https://leetcode.com/problems/linked-list-random-node/
/*
A good problem to refresh random library. Then default implementation is trivial.
Try follow up.
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
private:
    int m_size;
    ListNode* m_head;
public:
    /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
    Solution(ListNode* head) {
        m_head = head;
        m_size = 0;
        while(head){
            m_size++;
            head = head -> next;
        }
    }
    
    /** Returns a random node's value. */
    int getRandom() {
        default_random_engine seed((random_device())());
        uniform_int_distribution<int> distriIndex(0, m_size - 1);
        int pos = distriIndex(seed);
        ListNode* ptr = m_head;
        while(pos > 0){
            ptr = ptr->next;
            pos --;
        }
        return ptr->val;
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(head);
 * int param_1 = obj->getRandom();
 */

//Follow up implementation: Now list is large, you cannot know its size
//Reservoir sampling algorithm!!
//https://www.youtube.com/watch?time_continue=1&v=A1iwzSew5QY
class Solution {
private:
    ListNode* m_head;
public:
    /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
    Solution(ListNode* head) {
        m_head = head;
    }
    
    /** Returns a random node's value. */
    int getRandom() {
        int reservoirNum = 0;
        default_random_engine seed((random_device())());
        ListNode* ptr = m_head;
        int res = 0;
        while(ptr){
            uniform_int_distribution distr(0, reservoirNum);
            int index = distr(seed);
            if(index == 0)
                res = ptr->val;
            reservoirNum++;
            ptr = ptr->next;
        }
        return res;
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(head);
 * int param_1 = obj->getRandom();
 */


//398. Random Pick Index
//https://leetcode.com/problems/random-pick-index/
//Essentially the same idea as using 
//Reservoir sampling algorithm!!
//https://www.youtube.com/watch?time_continue=1&v=A1iwzSew5QY
//We cannot sort the array or the index will be wrong!
class Solution {
private:
    vector<int>* A;
public:
    Solution(vector<int>& nums) {
        A = &nums;
    }
    
    int pick(int target) {
        //We need to count how mant target value we have
        int countDuplicate = 0;
        int len = A->size();
        int res = -1;
        for(int i = 0; i < len; ++i){
            if((*A)[i] == target){
                // with prob 1/(n+1) to replace the previous index
                countDuplicate++; 
                //when countDuplicate == 1, we always pick the first one
                if(rand() % countDuplicate == 0)
                    res = i;
            }
        }
        return res;
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(nums);
 * int param_1 = obj->pick(target);
 */



//707. Design Linked List
//https://leetcode.com/problems/design-linked-list/
//define the node
struct node{
    int val;
    node* next;
    node(int x) : val(x), next(nullptr){}
};
class MyLinkedList {
private:
    int m_size;
    node* m_lhead;
    node* m_ltail;
    
public:
    /** Initialize your data structure here. */
    MyLinkedList() {
        m_size = 0;
        m_lhead = nullptr;
        m_ltail = nullptr;
    }
    
    void printList(){
        if(!m_lhead) return;
        node* temp = m_lhead;
        while(temp){
            cout << temp->val << " ";
            temp = temp->next;
        }
        cout << endl;
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    int get(int index) {
        if(index < 0 || index >= m_size)
            return -1;
        
        node* temp = m_lhead;
        while(index > 0){
            temp = temp->next;
            index--;
        }
        return temp->val;
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    void addAtHead(int val) {
        node* tempNode = new node(val);
        if(!m_lhead) {
            m_lhead = tempNode;
            m_ltail = tempNode;
            m_size++;
            //printList();
            return;
        }
        tempNode->next = m_lhead;
        m_lhead = tempNode;
        m_size++;
        //printList();
    }
    
    /** Append a node of value val to the last element of the linked list. */
    void addAtTail(int val) {
        if(!m_lhead){
            addAtHead(val);
            return;
        }
        node* tempNode = new node(val);
        m_ltail->next = tempNode;
        m_ltail = tempNode;
        m_size++;
        //printList();
    }
    
    node* getNodeBeforeI(int index){
        int i = index - 1;
        node* temp = m_lhead;
        while(i > 0){
            temp = temp->next;
            i--;
        }
        return temp;
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    void addAtIndex(int index, int val) {
        if(index > m_size) return;
        if(index <= 0) addAtHead(val);
        else if(index == m_size) addAtTail(val);
        else{
            node* temp = getNodeBeforeI(index);
            node* tempNode = new node(val);
            node* tempNextNode = temp->next;
            temp->next = tempNode;
            tempNode->next = tempNextNode;
            m_size++;
        }
        //printList();
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    void deleteAtIndex(int index) {
        if(index < 0 || index >= m_size) return;
        node* temp = getNodeBeforeI(index);
        if(index == 0){
            m_lhead = temp->next;
            m_size--;
            delete temp;
        }
        else{
            if(index == m_size - 1) m_ltail = temp;
            node* pendingDeleteNode = temp->next;
            temp->next = pendingDeleteNode->next;
            m_size--;
            delete pendingDeleteNode;            
        }
        //printList();
    }
};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */


//705. Design HashSet
//https://leetcode.com/problems/design-hashset/
typedef list<int> hl;
class MyHashSet {
private:
    /*
    //Get value from key
    unsigned int unhash(unsigned int x) {
        x = ((x >> 16) ^ x) * 0x119de1f3;
        x = ((x >> 16) ^ x) * 0x119de1f3;
        x = (x >> 16) ^ x;
        return x;
    }
    */
    //decent Hash function (without module)
    unsigned int hash(unsigned int x) {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x % 997;
    }
    vector<hl*> container;
    
public:
    /** Initialize your data structure here. */
    MyHashSet() {
        container = vector<hl*>(997, nullptr);
        
    }
    
    void add(int key) {
        unsigned int kIndex = hash(key);
        if(container[kIndex] == nullptr){
            container[kIndex] = new hl(1, key);
        }else{
            hl* tList = container[kIndex];
            auto it = tList->begin();
            for(; it != tList->end(); ++it){
                if(*it == key) return;
            }
            if(it == tList->end())
                container[kIndex]->emplace_back(key);
        }
    }
    
    void remove(int key) {
        unsigned int kIndex = hash(key);
        if(container[kIndex] == nullptr) return;
        else{
            hl* tList = container[kIndex];
            for(auto it = tList->begin(); it != tList->end(); ++it){
                if(*it == key){
                    tList->erase(it);
                    return;
                }
            }
        }
    }
    
    /** Returns true if this set contains the specified element */
    bool contains(int key) {
        unsigned int kIndex = hash(key);
        if(container[kIndex] == nullptr) return false;
        else{
            hl* tList = container[kIndex];
            for(auto it = tList->begin(); it != tList->end(); ++it){
                if(*it == key) return true;
            }
        }
        return false;
    }
};

/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet* obj = new MyHashSet();
 * obj->add(key);
 * obj->remove(key);
 * bool param_3 = obj->contains(key);
 */


//706. Design HashMap
//https://leetcode.com/problems/design-hashmap/
//hash list for each entry
//first is key, second is value
typedef list<pair<int, int>> hl;
class MyHashMap {
private:

    vector<hl*> hashContainer;
    //decent Hash function (without module)
    unsigned int hash(unsigned int x) {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x % 997;
    }
public:
    
    /** Initialize your data structure here. */
    MyHashMap() {
        hashContainer = vector<hl*>(997, nullptr);
    }
    
    /** value will always be non-negative. */
    void put(int key, int value) {
        int entry = hash(key);
        if(hashContainer[entry] == nullptr){
            hashContainer[entry] = new hl();
            hashContainer[entry]->push_back(make_pair(key, value));
        }else{
            for(auto it = hashContainer[entry]->begin(); it != hashContainer[entry]->end(); ++it){
                if((*it).first == key){
                    (*it).second = value;
                    return;
                }    
            }
            hashContainer[entry]->push_back(make_pair(key, value));
        }
    }
    
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {
        int entry = hash(key);
        if(hashContainer[entry] == nullptr) return -1;
        for(auto it = hashContainer[entry]->begin(); it != hashContainer[entry]->end(); ++it){
            if((*it).first == key)
                return (*it).second;
        }
        return -1;
    }
    
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {
        int entry = hash(key);
        if(hashContainer[entry] == nullptr) return;
        for(auto it = hashContainer[entry]->begin(); it != hashContainer[entry]->end(); ++it){
            if((*it).first == key){
                hashContainer[entry]->erase(it);
                return;
            }   
        }
    }
};

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap* obj = new MyHashMap();
 * obj->put(key,value);
 * int param_2 = obj->get(key);
 * obj->remove(key);
 */

//384. Shuffle an Array
//https://leetcode.com/problems/shuffle-an-array/
//Note how we use the time as the seed
class Solution {
private:
    vector<int> initNum;
    vector<int> m_nums;
    unsigned int m_size;
    
public:
    Solution(vector<int>& nums) {
        //We use time as the seed
        srand(time(NULL));
        m_size = nums.size();
        for(int n : nums){
            initNum.push_back(n);
            m_nums.push_back(n);
        }
    }
    
    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {
        m_nums = initNum;
        return m_nums;
    }
    
    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        if(m_nums.empty()) return vector<int>();
        int j = 0;
        for(int i = m_size-1; i >= 0; --i){
            //Fisher Yates algorithm
            j = rand() % (i+1);
            swap(m_nums[i], m_nums[j]);
        }
        return m_nums;
    }
};


//355. Design Twitter
//https://leetcode.com/problems/design-twitter/
/*My implementation, inefficient!*/
struct userData{
    int m_id;
    unordered_set<int> m_following;
    //user will always follow himself/herself
    userData(int id):m_id(id){
        m_following.insert(id);
    }
};
class Twitter {
private:
    list<pair<int, int>> TweetsPool;
    unordered_map<int, userData*> userPool;
    
public:
    /** Initialize your data structure here. */
    Twitter() {
        
    }
    
    /** Compose a new tweet. */
    void postTweet(int userId, int tweetId) {
        if(userPool.count(userId) == 0)
            userPool[userId] = new userData(userId);
        TweetsPool.push_front({tweetId, userId});
        //cout << "The user id is:" << userId << "Tweet ID is: " << tweetId << endl;
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    vector<int> getNewsFeed(int userId) {
        int total = 0;
        if(userPool.count(userId) == 0) return vector<int>();
        vector<int> res;
        for(auto it = TweetsPool.begin(); it != TweetsPool.end(); ++it){
            int followeeID = (*it).second;
            if(userPool[userId]->m_following.count(followeeID) > 0){
                res.push_back((*it).first);
                total++;
            }
            if(total >= 10) return res;
        }
        return res;
    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    void follow(int followerId, int followeeId) {
        if(userPool.count(followerId) == 0)
            userPool[followerId] = new userData(followerId);
        if(userPool.count(followeeId) == 0)
            userPool[followeeId] = new userData(followeeId);
        
        userPool[followerId]->m_following.insert(followeeId);
        //cout << "Now user: " << followerId << "followed user: " << followeeId << "with total number of followees: " << userPool[followerId]->m_following.size() << endl;
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    void unfollow(int followerId, int followeeId) {
        if(userPool.count(followerId) == 0)
            userPool[followerId] = new userData(followerId);
        if(userPool.count(followeeId) == 0)
            userPool[followeeId] = new userData(followeeId);
        //user cannot unfollow himself/herself
        if(followerId == followeeId) return;
        if(userPool.count(followerId) && userPool.count(followeeId)){
            userPool[followerId]->m_following.erase(followeeId);
        }
    }
};

/**
 * Your Twitter object will be instantiated and called as such:
 * Twitter* obj = new Twitter();
 * obj->postTweet(userId,tweetId);
 * vector<int> param_2 = obj->getNewsFeed(userId);
 * obj->follow(followerId,followeeId);
 * obj->unfollow(followerId,followeeId);
 */

//More optimized version, never implement by myself. I think the reason why it
//is faster is because getNewsFeed() function does the pruning before hand. In
//my implementation, I just search the list from front to end, which may involves
//many unnecessary check.
class Twitter {
public:
    /** Initialize your data structure here. */
    unordered_map<int, set<int> > followers;
    unordered_map<int, vector<pair<int, int>  > > users;
    int time = 1;
    Twitter() {
        
    }
    
    /** Compose a new tweet. */
    void postTweet(int userId, int tweetId) {
        users[userId].push_back({time++,tweetId});
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    vector<int> getNewsFeed(int userId) {
        priority_queue<pair<int, int> > pq; 
        vector<pair<int, int> > userTweets = users[userId];
        for(int i=userTweets.size()-1, c=0; c<10 && i>=0; i--, c++)
        {
            pq.push(userTweets[i]);
        }
        set<int> follow = followers[userId];
        for(auto it: follow)
        {
            vector<pair<int, int> > followTweets = users[it];
            for(int j=followTweets.size()-1, c=0; c<10 && j>=0; j--, c++)
            {
                pq.push(followTweets[j]);
            }
        }
        
        vector<int> ans;
        for(int i=0; i<10 && !pq.empty(); i++)
        {
            ans.push_back(pq.top().second);
            pq.pop();
        }
        return ans;
    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    void follow(int followerId, int followeeId) {
        if(followerId != followeeId)
          followers[followerId].insert(followeeId);
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    void unfollow(int followerId, int followeeId) {
        followers[followerId].erase(followeeId);
    }
};


//380. Insert Delete GetRandom O(1)
//https://leetcode.com/problems/insert-delete-getrandom-o1/
//The tricky part is how to delete an element in an array in O(1)
//A very useful techniques!
class RandomizedSet {
private:
    unordered_map<int, int> uMap;
    //Use the data to keep track of all the elements in the uMap
    //so we can implement random
    vector<int> data;
    default_random_engine seed;
public:
    /** Initialize your data structure here. */
    RandomizedSet() {
        seed = default_random_engine((random_device())());
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    bool insert(int val) {
        if(uMap.count(val) > 0) return false;
        //save the indices for element val in data
        int len = data.size();
        uMap[val] = len;
        data.push_back(val);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    bool remove(int val) {
        if(uMap.count(val) == 0) return false;
        if(data.back() == val){
            data.pop_back();
        }else{
            int index = uMap[val];
            int lastElement = data.back();
            //swap the to be deleted element with the last element of
            //the array, and then update the uMap
            swap(data[index], data.back());
            uMap[lastElement] = index;
            data.pop_back();
        }
        uMap.erase(val);
        return true;
    }
    
    /** Get a random element from the set. */
    int getRandom() {
        int len = data.size();
        uniform_int_distribution<int> distri(0, len-1);
        int index = distri(seed);
        return data[index];
    }
};

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet* obj = new RandomizedSet();
 * bool param_1 = obj->insert(val);
 * bool param_2 = obj->remove(val);
 * int param_3 = obj->getRandom();
 */

//460. LFU Cache
//https://leetcode.com/problems/lfu-cache/
//First Try: Wrong!
class LFUCache {
private:
    int m_capacity;
    std::list<int> keyOrder;
    std::unordered_map<int, int> uMap; 
    unordered_map<int, list<int>::iterator> uDictIt;
public:
    LFUCache(int capacity) {
        m_capacity = capacity;
    }
    
    int get(int key) {
        if(uMap.count(key) == 0 || m_capacity == 0) {
            return -1;
        }
        auto it = uDictIt[key];
        keyOrder.erase(it);
        keyOrder.emplace_front(key);
        uDictIt[key] = keyOrder.begin();
        //cout << uMap[key] << endl;
        return uMap[key];
    }
    
    void put(int key, int value) {
        if(m_capacity == 0) return;
        if(keyOrder.size() == m_capacity){
            int lastKey = keyOrder.back();
            uMap.erase(lastKey);
            uDictIt.erase(lastKey);
            keyOrder.pop_back();
        }
        uMap[key] = value;
        keyOrder.emplace_back(key);
        auto it = keyOrder.begin();
        for(; it != keyOrder.end(); ++it){
            if(std::next(it) == keyOrder.end())
                break;
        }
        uDictIt[key] = it;
        //for(int i : keyOrder)
        //    cout << i << " ";
        //cout << endl;
        
    }
};


//Right solution! Not an easy task to get it right! How to maintain the 
//information is critical here!
class LFUCache {
private:
    int m_lfu, m_capacity, m_size;
    //stores the key with the same frequency!
    unordered_map<int, list<int>> freq_keyMap;
    //stores the key, value - frequency map
    unordered_map<int, pair<int, int>> uMap;
    //stores the key - iterator map
    unordered_map<int, list<int>::iterator> key_Iter;
    
    void UpdateKey(int key){
        auto it = key_Iter[key];
        //original frequency
        int frequency = uMap[key].second;
        uMap[key].second++;
        freq_keyMap[frequency].erase(it);
        freq_keyMap[frequency+1].emplace_back(key);
        key_Iter[key] = --freq_keyMap[frequency+1].end();
        //If our original lfu is empty, we need to increase it by 1
        if(freq_keyMap[m_lfu].empty())
            m_lfu++;
    }
    
public:
    LFUCache(int capacity) {
        m_lfu = 0;
        m_capacity = capacity;
        m_size = 0;
    }
    
    int get(int key) {
        if(uMap.count(key) == 0) return -1;
        UpdateKey(key);
        return uMap[key].first;
    }
    
    void put(int key, int value) {
        if(!m_capacity) return;
        if(uMap.count(key) != 0){
            uMap[key].first = value;
            UpdateKey(key);
        }else{
            //The container reaches the limit
            if(m_size == m_capacity){
                int lastUsedKey = freq_keyMap[m_lfu].front();
                uMap.erase(lastUsedKey);
                key_Iter.erase(lastUsedKey);
                freq_keyMap[m_lfu].pop_front();
            }else
                m_size++;
            
            //Newly insert element with frequency to be 1
            uMap[key] = make_pair(value, 1);
            freq_keyMap[1].emplace_back(key);
            key_Iter[key] = --freq_keyMap[1].end();
            m_lfu = 1;
        }
    }
};

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache* obj = new LFUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */


//622. Design Circular Queue
//https://leetcode.com/problems/design-circular-queue/
//How to maintain the front and tail pointer is the key to success
class MyCircularQueue {
private:
    int m_capacity;
    int m_size;
    int* m_array;
    int m_front, m_tail;
    
public:
    /** Initialize your data structure here. Set the size of the queue to be k. */
    MyCircularQueue(int k) {
        m_array = new int[k];
        m_capacity = k;
        m_size = 0;
        m_front = m_tail = 0;
    }
    
    /** Insert an element into the circular queue. Return true if the operation is successful. */
    bool enQueue(int value) {
        if(isFull()) return false;
        m_array[m_tail] = value;
        m_size++;
        m_tail = (m_tail + 1) % m_capacity;
        return true;
    }
    
    /** Delete an element from the circular queue. Return true if the operation is successful. */
    bool deQueue() {
        if(isEmpty()) return false;
        m_size--;
        m_front = (m_front + 1) % m_capacity;
        return true;
    }
    
    /** Get the front item from the queue. */
    int Front() {
        if(isEmpty()) return -1;
        return m_array[m_front];
    }
    
    /** Get the last item from the queue. */
    int Rear() {
        if(isEmpty()) return -1;
        //We need to add m_capacity here in order to prevent the 
        //situation that m_tail - 1 < 0
        return m_array[(m_tail + m_capacity - 1) % m_capacity];
    }
    
    /** Checks whether the circular queue is empty or not. */
    bool isEmpty() {
        return m_size == 0;
    }
    
    /** Checks whether the circular queue is full or not. */
    bool isFull() {
        return m_size == m_capacity;
    }
};

/**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue* obj = new MyCircularQueue(k);
 * bool param_1 = obj->enQueue(value);
 * bool param_2 = obj->deQueue();
 * int param_3 = obj->Front();
 * int param_4 = obj->Rear();
 * bool param_5 = obj->isEmpty();
 * bool param_6 = obj->isFull();
 */




