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

 /*
    34. Find First and Last Position of Element in Sorted Array
    https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
 
    Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

    If target is not found in the array, return [-1, -1].

    You must write an algorithm with O(log n) runtime complexity.

    Example 1:

    Input: nums = [5,7,7,8,8,10], target = 8
    Output: [3,4]

    Example 2:

    Input: nums = [5,7,7,8,8,10], target = 6
    Output: [-1,-1]
 */
// Solution 01: two pass iterative binary search
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        int m = 0;
        vector<int> res (2, -1);
        while (l <= r) {
            m = l + (r - l) / 2;
            if (nums[m] < target) l = m + 1;
            else if (nums[m] > target) r = m - 1;
            else {
                res[0] = m;
                r = m -1;
            }
        }
        // No target found, early return.
        if (res[0] == -1) return res;
        
        l = 0;
        r = nums.size() - 1;
        while (l <= r) {
            m = l + (r - l) / 2;
            if (nums[m] < target) l = m + 1;
            else if (nums[m] > target) r = m - 1;
            else {
                res[1] = m;
                l = m + 1;
            }
        }
        return res;
    }
};

 /*
    69. Sqrt(x)
    https://leetcode.com/problems/sqrtx/
 
    Given a non-negative integer x, compute and return the square root of x.

    Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.

    Note: You are not allowed to use any built-in exponent function or operator, such as pow(x, 0.5) or x ** 0.5.

    Example 1:

    Input: x = 4
    Output: 2
    Example 2:

    Input: x = 8
    Output: 2
    Explanation: The square root of 8 is 2.82842..., and since the decimal part is truncated, 2 is returned.
 */
// Solution: binary search.
class Solution {
public:
    int mySqrt(int x) {
        long l = 0, r = x;
        long m = 0;
        while (l < r) {
            m = l + (r - l) / 2;
            if (m * m == x) return m;
            else if (m * m < x) l = m + 1;
            else r = m - 1;
        }
        if (l * l > x) l = l - 1;
        return l;
    }
};

 /*
    367. Valid Perfect Square
    https://leetcode.com/problems/valid-perfect-square/
 
    Given a positive integer num, write a function which returns True if num is a perfect square else False.

    Follow up: Do not use any built-in library function such as sqrt.

    Example 1:

    Input: num = 16
    Output: true
    Example 2:

    Input: num = 14
    Output: false
    

    Constraints:

    1 <= num <= 2^31 - 1
 */
class Solution {
public:
    bool isPerfectSquare(int x) {
        long l = 0, r = x;
        long m = 0;
        while (l < r) {
            m = l + (r - l) / 2;
            if (m * m == x) return true;
            else if (m * m < x) l = m + 1;
            else r = m - 1;
        }
        if (l * l > x) l = l - 1;
        return l * l == x ? true : false;
    }
};
