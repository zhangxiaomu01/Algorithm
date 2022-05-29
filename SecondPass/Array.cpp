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

 /*
    27. Remove Element
    https://leetcode.com/problems/remove-element/
 
    Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. 
    The relative order of the elements may be changed.

    Since it is impossible to change the length of the array in some languages, you must instead 
    have the result be placed in the first part of the array nums. More formally, if there are k 
    elements after removing the duplicates, then the first k elements of nums should hold the 
    final result. It does not matter what you leave beyond the first k elements.

    Return k after placing the final result in the first k slots of nums.

    Do not allocate extra space for another array. You must do this by modifying the 
    input array in-place with O(1) extra memory.

    Custom Judge:

    The judge will test your solution with the following code:

    int[] nums = [...]; // Input array
    int val = ...; // Value to remove
    int[] expectedNums = [...]; // The expected answer with correct length.
                                // It is sorted with no values equaling val.

    int k = removeElement(nums, val); // Calls your implementation

    assert k == expectedNums.length;
    sort(nums, 0, k); // Sort the first k elements of nums
    for (int i = 0; i < actualLength; i++) {
        assert nums[i] == expectedNums[i];
    }
    If all assertions pass, then your solution will be accepted.

    

    Example 1:

    Input: nums = [3,2,2,3], val = 3
    Output: 2, nums = [2,2,_,_]
    Explanation: Your function should return k = 2, with the first two elements of nums being 2.
    It does not matter what you leave beyond the returned k (hence they are underscores).
    Example 2:

    Input: nums = [0,1,2,2,3,0,4,2], val = 2
    Output: 5, nums = [0,1,4,0,3,_,_,_]
    Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
    Note that the five elements can be returned in any order.
    It does not matter what you leave beyond the returned k (hence they are underscores).
    

    Constraints:

    0 <= nums.length <= 100
    0 <= nums[i] <= 50
    0 <= val <= 100
 */
// In-place element swap
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int kIndex = nums.size() - 1;
        for(int i = 0; i < nums.size(); ++i) {
            if (nums[i] == val) {
                while (kIndex >= 0 && nums[kIndex] == val) kIndex --;
                if (kIndex < i) break; 
                nums[i] = nums[kIndex];
                nums[kIndex] = val;
                kIndex--;
            }
        }
        return kIndex + 1;
    }
};

// Fast + slow pointer: similar approach
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slowIndex = 0;
        for (int fastIndex = 0; fastIndex < nums.size(); fastIndex++) {
            if (val != nums[fastIndex]) {
                nums[slowIndex++] = nums[fastIndex];
            }
        }
        return slowIndex;
    }
};

 /*
    26. Remove Duplicates from Sorted Array
    https://leetcode.com/problems/remove-duplicates-from-sorted-array/
 
    Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such 
    that each unique element appears only once. The relative order of the elements should be kept 
    the same.

    Since it is impossible to change the length of the array in some languages, you must instead 
    have the result be placed in the first part of the array nums. More formally, if there are k 
    elements after removing the duplicates, then the first k elements of nums should hold the final 
    result. It does not matter what you leave beyond the first k elements.

    Return k after placing the final result in the first k slots of nums.

    Do not allocate extra space for another array. You must do this by modifying the input array 
    in-place with O(1) extra memory.

    Custom Judge:

    The judge will test your solution with the following code:

    int[] nums = [...]; // Input array
    int[] expectedNums = [...]; // The expected answer with correct length

    int k = removeDuplicates(nums); // Calls your implementation

    assert k == expectedNums.length;
    for (int i = 0; i < k; i++) {
        assert nums[i] == expectedNums[i];
    }
    If all assertions pass, then your solution will be accepted.

    

    Example 1:

    Input: nums = [1,1,2]
    Output: 2, nums = [1,2,_]
    Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
    It does not matter what you leave beyond the returned k (hence they are underscores).
    Example 2:

    Input: nums = [0,0,1,1,1,2,2,3,3,4]
    Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
    Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
    It does not matter what you leave beyond the returned k (hence they are underscores).
    

    Constraints:

    1 <= nums.length <= 3 * 104
    -100 <= nums[i] <= 100
    nums is sorted in non-decreasing order.
 */
// In-place swap
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int previous = nums[0];
        int slow = 1;
        if (nums.size() == 1) return 1;
        for(int fast = 0; fast < nums.size(); ++fast) {
            if (nums[fast] != previous) {
                nums[slow ++] = nums[fast];
            }
            previous = nums[fast];
        }
        return slow;
    }
};

 /*
    283. Move Zeroes
    https://leetcode.com/problems/move-zeroes/
 
    Given an integer array nums, move all 0's to the end of it while maintaining the relative 
    order of the non-zero elements.

    Note that you must do this in-place without making a copy of the array.

    

    Example 1:

    Input: nums = [0,1,0,3,12]
    Output: [1,3,12,0,0]
    Example 2:

    Input: nums = [0]
    Output: [0]
    

    Constraints:

    1 <= nums.length <= 104
    -231 <= nums[i] <= 231 - 1
 */
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int slow = 0;
        int i = 0;
        while (i < nums.size() && nums[i] != 0) {
            i++;
            slow++;
        }
        for (; i < nums.size(); ++i) {
            if(nums[i] != 0) {
                nums[slow] = nums[i];
                nums[i] = 0;
                while(slow < i && nums[slow] != 0) slow ++;
            }
        }
    }
};


 /*
    844. Backspace String Compare (Follow up)
    https://leetcode.com/problems/backspace-string-compare/
 
    Given two strings s and t, return true if they are equal when both are typed into empty 
    text editors. '#' means a backspace character.

    Note that after backspacing an empty text, the text will continue empty.

    Example 1:

    Input: s = "ab#c", t = "ad#c"
    Output: true
    Explanation: Both s and t become "ac".
    Example 2:

    Input: s = "ab##", t = "c#d#"
    Output: true
    Explanation: Both s and t become "".
    Example 3:

    Input: s = "a#c", t = "b"
    Output: false
    Explanation: s becomes "c" while t becomes "b".
    

    Constraints:

    1 <= s.length, t.length <= 200
    s and t only contain lowercase letters and '#' characters.
    

    Follow up: Can you solve it in O(n) time and O(1) space?
 */
// The initial problem can easily be solved using stack. This is the solution with O(1) space.
class Solution {
public:
    bool backspaceCompare(string s, string t) {
        int i = s.size() - 1, j = t.size() - 1;
        while(i >= 0 || j >= 0) {
            int count = 0;
            while(i >= 0 && (s[i] == '#' || count > 0)) {
                if (s[i] == '#')
                    count ++;
                else
                    count --;
                i--;
            }
            count = 0;
            while(j >= 0 && (t[j] == '#' || count > 0)) {
                if (t[j] == '#')
                    count ++;
                else
                    count --;
                j--;
            }
            if(i < 0 && j < 0) return true;
            else if((i < 0 && j >= 0) || (j < 0 && i >= 0)) return false;
            if(s[i] != '#' && t[j] != '#' && s[i] != t[j]) return false;
            i--;
            j--;
        }
        return true;
    }
};