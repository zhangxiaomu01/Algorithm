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




