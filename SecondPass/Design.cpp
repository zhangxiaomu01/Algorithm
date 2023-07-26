/**
 * @file Design.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2023-07-13
 * 
 * A quick second pass with the common 'Design' problems.
 */
 /*
    146. LRU Cache
    https://leetcode.com/problems/lru-cache/
    Design a data structure that follows the constraints of a Least Recently Used (LRU) 
    cache.

    Implement the LRUCache class:
    LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
    int get(int key) Return the value of the key if the key exists, otherwise return -1.
    void put(int key, int value) Update the value of the key if the key exists. Otherwise, 
    add the key-value pair to the cache. If the number of keys exceeds the capacity from 
    this operation, evict the least recently used key.
    The functions get and put must each run in O(1) average time complexity.

    Example 1:
    Input
    ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
    [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
    Output
    [null, null, null, 1, null, -1, null, -1, 3, 4]

    Explanation
    LRUCache lRUCache = new LRUCache(2);
    lRUCache.put(1, 1); // cache is {1=1}
    lRUCache.put(2, 2); // cache is {1=1, 2=2}
    lRUCache.get(1);    // return 1
    lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
    lRUCache.get(2);    // returns -1 (not found)
    lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
    lRUCache.get(1);    // return -1 (not found)
    lRUCache.get(3);    // return 3
    lRUCache.get(4);    // return 4
    

    Constraints:
    1 <= capacity <= 3000
    0 <= key <= 10^4
    0 <= value <= 10^5
    At most 2 * 105 calls will be made to get and put.
 */
/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
class LRUCache {
private:
    unordered_map<int, list<int>::iterator> uMap;
    // The first the element in the cache is the least used!
    list<int> cache;
    // Key-value pair
    unordered_map<int, int> uCache;
    int cap;

    // Make sure that we move the key-value to the least used position.
    void updateKey(int key) {
        auto it = uMap[key];
        cache.erase(it);
        cache.push_front(key);
        uMap[key] = cache.begin();
    }

    void evict() {
        if (cache.empty()) return;
        int key = cache.back();
        cache.pop_back();
        uMap.erase(key);
        uCache.erase(key);
    }

public:
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        if (uCache.find(key) != uCache.end()) {
            updateKey(key);
            return uCache[key];
        }
        return -1;
    }
    
    void put(int key, int value) {
        if (uCache.find(key) != uCache.end()) {
            uCache[key] = value;
            updateKey(key);
            return;
        }

        if (cache.size() == cap) {
            evict();
        }
        cache.push_front(key);
        uCache[key] = value;
        uMap[key] = cache.begin();
    }
};

 /*
    295. Find Median from Data Stream
    https://leetcode.com/problems/find-median-from-data-stream/
    The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle 
    value, and the median is the mean of the two middle values.

    For example, for arr = [2,3,4], the median is 3.
    For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
    Implement the MedianFinder class:

    MedianFinder() initializes the MedianFinder object.
    void addNum(int num) adds the integer num from the data stream to the data structure.
    double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be 
    accepted.
    

    Example 1:
    Input
    ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
    [[], [1], [2], [], [3], []]
    Output
    [null, null, null, 1.5, null, 2.0]

    Explanation
    MedianFinder medianFinder = new MedianFinder();
    medianFinder.addNum(1);    // arr = [1]
    medianFinder.addNum(2);    // arr = [1, 2]
    medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
    medianFinder.addNum(3);    // arr[1, 2, 3]
    medianFinder.findMedian(); // return 2.0
    

    Constraints:
    -10^5 <= num <= 10^5
    There will be at least one element in the data structure before calling findMedian.
    At most 5 * 10^4 calls will be made to addNum and findMedian.
 */
/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */
class MedianFinder {
private:
    priority_queue<int> maxQ;
    priority_queue<int, vector<int>, greater<int>> minQ;
    int count = 0;

public:
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        count++;
        maxQ.push(num);
        // Balancing the two queues!
        minQ.push(maxQ.top());
        maxQ.pop();
        if (maxQ.size() < minQ.size()) {
            maxQ.push(minQ.top());
            minQ.pop();
        }

    }
    
    double findMedian() {
        if (count == 0) return -1.0;
        if (count & 1) return (double) maxQ.top();
        return (minQ.top() + maxQ.top()) * 0.5;
    }
};
