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


