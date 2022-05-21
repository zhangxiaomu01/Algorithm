/**
 * @file Array.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2022-05-15
 * 
 * A quick second pass with the common 'Array' algorithm problems.
 * 
 */

 // Category: Binary Search
 /*
    704. Binary Search
    https://leetcode.com/problems/binary-search/
 
    Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

    You must write an algorithm with O(log n) runtime complexity.
 */
// Solution 01: iterative binary search
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        int m = 0;
        while (l < r) {
            m = l + (r - l) / 2;
            if (nums[m] < target) l = m + 1;
            else if (nums[m] == target) return m;
            else r = m;
        }
        return nums[l] == target ? l : -1;
    }
};

// Solution 2: recursive binary search
class Solution {
public:
    int search(vector<int>& nums, int target) {
        return binarySearch(nums, target, 0, nums.size() - 1);
    }
    
    int binarySearch(vector<int>& nums, int target, int l, int r) {
        if (l > r) return -1;
        if (l == r) return nums[l] == target ? l : -1;
        int m = l + (r - l) / 2;
        if (nums[m] > target) return binarySearch(nums, target, l, m);
        else if (nums[m] == target) return m;
        else {
            return binarySearch(nums, target, m+1, r);
        }
        return -1;
    }
};

 /*
    35. Search Insert Position
    https://leetcode.com/problems/search-insert-position/
 
    Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

    You must write an algorithm with O(log n) runtime complexity.
 */
// Solution 01: iterative binary search
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        int m = 0;
        // We won't have l <= r here, because when that happens, we
        // either found nums[l] == target, or not, which is the same
        // index for us to insert the value.
        while (l <= r) {
            if (l == r) {
                return nums[l] >= target ? l  : l + 1;
            }
            m = l + (r - l) / 2;
            if (nums[m] == target) return m;
            else if (nums[m] < target) l = m + 1;
            else r = m - 1;
        }
        return l;
    }
};


