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

    ~MyCircularQueue(){
        delete[] m_array;
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


//641. Design Circular Deque
//https://leetcode.com/problems/design-circular-deque/
//Exactly the same as problem 622.
class MyCircularDeque {
private:
    int m_capacity;
    int m_size;
    int m_front, m_tail;
    int* m_array;
public:
    /** Initialize your data structure here. Set the size of the deque to be k. */
    MyCircularDeque(int k) {
        m_array = new int[k];
        m_size = 0;
        m_capacity = k;
        m_front = m_tail = 0;
    }
    
    ~MyCircularDeque(){
        delete[] m_array;
    }
    
    /** Adds an item at the front of Deque. Return true if the operation is successful. */
    bool insertFront(int value) {
        if(isFull()) return false;
        m_front = (m_front + m_capacity - 1) % m_capacity;
        m_array[m_front] = value;
        m_size++;
        return true;
    }
    
    /** Adds an item at the rear of Deque. Return true if the operation is successful. */
    bool insertLast(int value) {
        if(isFull()) return false;
        m_array[m_tail] = value;
        m_tail = (m_tail + 1) % m_capacity;
        m_size++;
        return true;
    }
    
    /** Deletes an item from the front of Deque. Return true if the operation is successful. */
    bool deleteFront() {
        if(isEmpty()) return false;
        m_front = (m_front + 1) % m_capacity;
        m_size--;
        return true;
    }
    
    /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
    bool deleteLast() {
        if(isEmpty()) return false;
        //pop the last element, we need to add m_capacity to avoid
        //negative index
        m_tail = (m_tail + m_capacity - 1) % m_capacity;
        m_size--;
        return true;
    }
    
    /** Get the front item from the deque. */
    int getFront() {
        if(isEmpty()) return -1;
        return m_array[m_front];
    }
    
    /** Get the last item from the deque. */
    int getRear() {
        if(isEmpty()) return -1;
        return m_array[(m_tail + m_capacity - 1) % m_capacity];
    }
    
    /** Checks whether the circular deque is empty or not. */
    bool isEmpty() {
        return m_size == 0;
    }
    
    /** Checks whether the circular deque is full or not. */
    bool isFull() {
        return m_size == m_capacity;
    }
};

/**
 * Your MyCircularDeque object will be instantiated and called as such:
 * MyCircularDeque* obj = new MyCircularDeque(k);
 * bool param_1 = obj->insertFront(value);
 * bool param_2 = obj->insertLast(value);
 * bool param_3 = obj->deleteFront();
 * bool param_4 = obj->deleteLast();
 * int param_5 = obj->getFront();
 * int param_6 = obj->getRear();
 * bool param_7 = obj->isEmpty();
 * bool param_8 = obj->isFull();
 */



//855. Exam Room
//https://leetcode.com/problems/exam-room/
//A straightforward approach. O(n) for insertion, O(1) for deletion
class ExamRoom {
private:
    int m_length;
    list<int> seats;
    //maps current seat index with an iterator in seats list, for fast
    //deletion
    unordered_map<int, list<int>::iterator> seatMap;
public:
    ExamRoom(int N) {
        m_length = N;
    }
    
    int seat() {
        if(seats.empty()){
            seats.emplace_back(0);
            seatMap[0] = seats.begin();
            return 0;
        }
        //Note we need to check which distance is larger:
        // ....s........s_k......
        //which is befroe the first seat, between the first seat and last
        //seat and after the last seat
        // pre - previous seat in the list
        int pre = -1;
        //val - current seat index, dist - distance between pre and 
        //current
        int val = 0, dist = 0;
        //pos - the position we are going to insert the seat to our list
        list<int>::iterator pos;
        for(auto it = seats.begin(); it!=seats.end(); ++it){
            //handle the first case, now pre points to the first 
            //seat
            if(pre == -1){
                dist = *it;
                pos = it;
                val = 0;
            }
            //find a gap larger than previous one
            else if((*it - pre) / 2 > dist){
                dist = (*it - pre) / 2;
                val = (*it + pre) / 2;
                pos = it;
            }
            pre = *it;
        }
        //Now we need to check the distance after the last seat
        if(m_length - 1 - seats.back() > dist){
            val = m_length - 1;
            dist = m_length - 1 - seats.back();
            pos = seats.end();
        }
        //The return value will be the iterator points to new inserted 
        //element
        //insert before pos
        pos = seats.insert(pos, val);
        seatMap[val] = pos;
        return val;
    }
    
    void leave(int p) {
        auto it = seatMap[p];
        seatMap.erase(p);
        seats.erase(it);
    }
};

/**
 * Your ExamRoom object will be instantiated and called as such:
 * ExamRoom* obj = new ExamRoom(N);
 * int param_1 = obj->seat();
 * obj->leave(p);
 */



//900. RLE Iterator
//https://leetcode.com/problems/rle-iterator/
class RLEIterator {
private:
    //create a number - val mapping
    queue<pair<int, int>> Q;
public:
    RLEIterator(vector<int>& A) {
        for(int i = 0; i < A.size(); ++i){
            if(i % 2 == 0){
                if(A[i] == 0) continue;
                Q.push({A[i+1], A[i]});
            }
        }
    }
    
    int next(int n) {
        
        while(!Q.empty()){
            //We need to create a reference here, since we want to modify
            //queue on the fly!
            auto& p = Q.front();
            int num = p.second;
            int val = p.first;
            if(num >= n){
                p.second -= n;
                return val;
            }else{
                n -= num;
                Q.pop();
            }
        }
        return -1;
    }
};

/**
 * Your RLEIterator object will be instantiated and called as such:
 * RLEIterator* obj = new RLEIterator(A);
 * int param_1 = obj->next(n);
 */


//981. Time Based Key-Value Store
//https://leetcode.com/problems/time-based-key-value-store/
class TimeMap {
    //create a key - (val, time stamp) map
    unordered_map<string, vector<pair<int, string>>> uMap;
    int getIndex(vector<pair<int, string>>& v, int t){
        int l = 0, r = v.size()-1;
        while(l < r){
            int mid = l + (r - l) / 2;
            if(v[mid].first >= t)
                r = mid;
            else
                l = mid+1;
        }
        return l;
    }
    
public:
    /** Initialize your data structure here. */
    TimeMap() {
        
    }
    
    void set(string key, string value, int timestamp) {
        uMap[key].push_back({timestamp, value});
    }
    
    string get(string key, int timestamp) {
        if(uMap.count(key) == 0) return "";
        auto myComp = [](pair<int, string>& p1, int p2){
            return p1.first < p2;
        };

        int pos = getIndex(uMap[key], timestamp);
        if(pos == 0 && uMap[key][pos].first > timestamp) return "";
        else if(pos > 0 && uMap[key][pos].first > timestamp)
            return uMap[key][pos-1].second;
        else
            return uMap[key][pos].second;
    }
};

/**
 * Your TimeMap object will be instantiated and called as such:
 * TimeMap* obj = new TimeMap();
 * obj->set(key,value,timestamp);
 * string param_2 = obj->get(key,timestamp);
 */


//729. My Calendar I
//https://leetcode.com/problems/my-calendar-i/
class MyCalendar {
private:
    vector<pair<int, int>> bookings;
public:
    MyCalendar() {
        
    }
    
    bool book(int start, int end) {
        for(auto p : bookings){
            if(start < p.second && end > p.first)
                return false;
        }
        bookings.push_back({start, end});
        return true;
    }
};

/**
 * Your MyCalendar object will be instantiated and called as such:
 * MyCalendar* obj = new MyCalendar();
 * bool param_1 = obj->book(start,end);
 */

//Slow
class MyCalendar {
private:
    map<int, int> bookings;
public:
    MyCalendar() {
        
    }
    
    bool book(int start, int end) {
        bookings[start]++;
        bookings[end]--;
        int booked = 0;
        for(auto it : bookings){
            booked += it.second;
            if(booked == 2){
                bookings[start]--;
                bookings[end]++;
                return false;
            }
        }
        return true;
    }
};

//Binary tree, not bad
class MyCalendar {
private:
    //Alternatively, we can set as well like below:
    //set<pair<int, int>> books;
    //auto next = books.lower_bound({s, e});
    map<int, int> bookings;
public:
    MyCalendar() {
        
    }
    
    bool book(int start, int end) {
        // first element with key not go before k (i.e., either it is equivalent or goes after). 
        //Note how we use the lower_bound here to boost efficiency!
        auto next = bookings.lower_bound(start);
        if(next != bookings.end() && next->first < end)
            return false;
        if(next != bookings.begin() && (--next)->second > start)
            return false;
        bookings[start] = end;
        return true;
    }
};


//731. My Calendar II
//https://leetcode.com/problems/my-calendar-ii/
/*
Utilize the two vectors to save the information of booking and overlap. 
Not that bad.
*/
class MyCalendarTwo {
private:
    vector<pair<int, int>> bookings;
    vector<pair<int, int>> doubleBookings;
public:
    MyCalendarTwo() {
        
    }
    
    bool book(int start, int end) {
        for(auto p : doubleBookings){
            if(start < p.second && end > p.first)
                return false;
        }
        for(auto p : bookings){
            //Note we need to maintain the shorter interval!
            if(start < p.second && end > p.first)
                doubleBookings.push_back({max(start, p.first), min(end, p.second)});
        }
        bookings.push_back({start, end});
        return true;
    }
};

/**
 * Your MyCalendarTwo object will be instantiated and called as such:
 * MyCalendarTwo* obj = new MyCalendarTwo();
 * bool param_1 = obj->book(start,end);
 */

//Treemap. Similar to light on/off question!
//Add a tag to start and end position, and calculate the range sum.
//Slower than 2 vector solution!
class MyCalendarTwo {
private:
    map<int, int> bookings;
public:
    MyCalendarTwo() {
        
    }
    bool book(int start, int end) {
        bookings[start]++;
        bookings[end]--;
        int booked = 0;
        for(auto p : bookings){
            booked += p.second;
            if(booked == 3){
                bookings[start] --;
                bookings[end] ++;
                return false;
            }
        }
        return true;
    }
};

//Another option, not implemented by myself. 
class MyCalendar {
public:
    MyCalendar() {} 
    bool book(int start, int end) {
        int p=0,st=0,en=arr.size()-1;
        while(st<=en){
            if(start<arr[st].first){p=st;break;}
            if(start>arr[en].first){p=en+1;break;}
            int mid=(st+en)>>1;
            if(start==arr[mid].first)return false;
            if(start<arr[mid].first)en=mid-1;
            else st=mid+1;
        }
        if((p>0&&start<arr[p-1].second)||(p<arr.size()&&end>arr[p].first))return false;
        arr.insert(arr.begin()+p,pair<int,int>(start,end));
        return true;
    }
private:
    vector<pair<int,int> >arr;
};


//732. My Calendar III
//https://leetcode.com/problems/my-calendar-iii/
class MyCalendarThree {
private:
    map<int, int> bookings;
public:
    MyCalendarThree() {
        
    }
    
    int book(int start, int end) {
        bookings[start]++;
        bookings[end]--;
        int count = 0;
        int res = 0;
        for(auto p : bookings){
            count += p.second;
            res = max(res, count);
        }
        return res;
    }
};


/*Excellent idea from:
https://leetcode.com/problems/my-calendar-iii/discuss/176950/
Very tricky! Fast in practice
*/
class MyCalendarThree {
private:
    //map records the index and how many overlap we have for current index
    //Note we have to put -1 here because 
    //we can initialize bookings like below
    map<int, int> bookings = {{-1, 0}};
    //maintain a global res here
    int res = 0;
public:
    MyCalendarThree() {
        
    }
    int book(int start, int end) {
        auto sIndex = bookings.emplace(start, (--bookings.lower_bound(start))->second).first;
        auto eIndex = bookings.emplace(end, (--bookings.lower_bound(end))->second).first; 
        for(auto it = sIndex; it != eIndex; ++it){
            res = max(res, ++(it->second));
        }
        return res;
    }
};



//157. Read N Characters Given Read4
//https://leetcode.com/problems/read-n-characters-given-read4/
//Initial implementation, unoptimized!
// Forward declaration of the read4 API.
int read4(char *buf);

class Solution {
public:
    /**
     * @param buf Destination buffer
     * @param n   Number of characters to read
     * @return    The number of actual characters read
     */
    int read(char *buf, int n) {
        int cnt = 0;
        for(int i = 0; i < n/4; ++i){
            char localBuf[4];
            int x = read4(localBuf);
            for(int j = 0; j < x; ++j)
                buf[j+cnt] = localBuf[j];
            cnt += x;
            if(x == 0) break;
        }
        char localBuf[4];
        int x = read4(localBuf);
        for(int i = 0; i < min(n%4, x); ++i){
            buf[i+cnt] = localBuf[i];
        }
        cnt += min(n%4, x);
        
        return cnt;
    }
};


//Optimized version. Time complexity is the same. The code is more concise!
// Forward declaration of the read4 API.
int read4(char *buf);

class Solution {
public:
    /**
     * @param buf Destination buffer
     * @param n   Number of characters to read
     * @return    The number of actual characters read
     */
    int read(char *buf, int n) {
        int total = 0;
        char* bufPtr = buf;
        while(total < n){
            char tempBuf[4];
            int m = read4(tempBuf);
            //cout << m << endl;
            for(int i = 0; i < min(m, n - total); ++i){
                bufPtr[i + total] = tempBuf[i];  
            }
            total += min(m, n - total);
            if(m < 4) break;
        }
        return total;
    }
};



//158. Read N Characters Given Read4 II - Call multiple times
//https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/
/*
Extreme elegant solution! note how to save the temperate result to member 
variable and how to maintain two pointers is the key to success!
*/

// Forward declaration of the read4 API.
int read4(char *buf);

class Solution {
private:
    //internal buffer: since we will have mulptiple call, we need to save the
    //temporary result to a member variable.
    char tempChar[4];
    //how many characters previous read!
    int n4 = 0;
    //Current pointer to internal buffer
    int i4 = 0;
public:
    /**
     * @param buf Destination buffer
     * @param n   Number of characters to read
     * @return    The number of actual characters read
     */
    int read(char *buf, int n) {
        int total = 0;
        while(total < n){
            //when our internal buffer reaches boundry
            if(i4 >= n4){
                //reset i4
                i4 = 0;
                n4 = read4(tempChar);
                //no need for further update
                if(n4 == 0) break;
            }
            buf[total++] = tempChar[i4++];
        }
        return total;
    }
};


//1032. Stream of Characters
//https://leetcode.com/problems/stream-of-characters/
/* Trie! */
struct Trie{
    vector<shared_ptr<Trie>> children;
    bool isTerminated;
    
    Trie(){
        isTerminated = false;
        children = vector<shared_ptr<Trie>>(26, nullptr);
    }
    
};

class StreamChecker {
private:
    shared_ptr<Trie> root_;
    deque<char> letters_;
    int maxLen_;
public:
    //Build the trie of reverse order of s
    StreamChecker(vector<string>& words) {
        root_ = shared_ptr<Trie>(new Trie());
        for(string& s : words){
            int lenS = s.size();
            maxLen_ = max(maxLen_, lenS);
            shared_ptr<Trie> t = root_;
            for(int i = s.size()-1; i >= 0; --i){
                if(t->children[s[i]-'a'] == nullptr){
                    t->children[s[i]-'a'] = shared_ptr<Trie>(new Trie());
                }
                t = t->children[s[i]-'a'];
            }
            t->isTerminated = true;
        }
    }
    
    bool query(char letter) {
        letters_.push_back(letter);
        if(letters_.size() > maxLen_){
            letters_.pop_front();
        }
        shared_ptr<Trie> t = root_;
        for(auto it = letters_.rbegin(); it != letters_.rend(); ++it){
            if(t->children[(*it)-'a'] == nullptr) return false;
            t = t->children[(*it)-'a'];
            if(t->isTerminated) return true;
        }
        std::cout << std::endl;
        return false;
    }
};

/**
 * Your StreamChecker object will be instantiated and called as such:
 * StreamChecker* obj = new StreamChecker(words);
 * bool param_1 = obj->query(letter);
 */

