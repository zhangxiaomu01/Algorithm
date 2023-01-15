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

 /*
    977. Squares of a Sorted Array (follow up)
    https://leetcode.com/problems/squares-of-a-sorted-array/
 
    Given an integer array nums sorted in non-decreasing order, return an array of the squares of 
    each number sorted in non-decreasing order.

    Example 1:
    Input: nums = [-4,-1,0,3,10]
    Output: [0,1,9,16,100]
    Explanation: After squaring, the array becomes [16,1,0,9,100].
    After sorting, it becomes [0,1,9,16,100].

    Example 2:
    Input: nums = [-7,-3,2,3,11]
    Output: [4,9,9,49,121]
    
    Constraints:
    1 <= nums.length <= 104
    -104 <= nums[i] <= 104
    nums is sorted in non-decreasing order.

    Follow up: Squaring each element and sorting the new array is very trivial, could you find 
    an O(n) solution using a different approach?
 */
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        int index = -1;
        int previous = abs(nums[0]);
        vector<int> res(1, 0);
        for (int i = 1; i < nums.size(); ++i) {
            int num = abs(nums[i]);
            if (num > previous) {
                index = i - 1;
                break;
            }
            previous = num;
        }
        index = index == -1 ? nums.size() - 1 : index;
        res[0] = nums[index] * nums[index];
        int l = index - 1, r = index + 1;
        while (l >= 0 && r < nums.size()) {
            if (abs(nums[l]) < nums[r]){
                res.push_back(nums[l] * nums[l]);
                l--;
            } else {
                res.push_back(nums[r] * nums[r]);
                r++;
            }
        }
        while (l >= 0) {
            res.push_back(nums[l] * nums[l]);
            l--;
        }
        
        while(r < nums.size()) {
            res.push_back(nums[r] * nums[r]);
            r++;
        }
        return res;
    }
};

// Similar idea, but more concise code
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        int previous = abs(nums[0]);
        int i = 0, j = nums.size() - 1;
        vector<int> res(nums.size(), 0);
        for(int k=nums.size()-1; k >= 0; k--) {
            if(abs(nums[i]) > abs(nums[j])) {
                res[k] = nums[i] * nums[i];
                i++;
            } else {
                res[k] = nums[j] * nums[j];
                j--;
            }
        }
        return res;
    }
};

 /*
    209. Minimum Size Subarray Sum
    https://leetcode.com/problems/minimum-size-subarray-sum/
 
    Given an array of positive integers nums and a positive integer target, return the minimal length 
    of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which the sum is greater than 
    or equal to target. If there is no such subarray, return 0 instead.

    Example 1:

    Input: target = 7, nums = [2,3,1,2,4,3]
    Output: 2
    Explanation: The subarray [4,3] has the minimal length under the problem constraint.
    Example 2:

    Input: target = 4, nums = [1,4,4]
    Output: 1
    Example 3:

    Input: target = 11, nums = [1,1,1,1,1,1,1,1]
    Output: 0
    

    Constraints:

    1 <= target <= 109
    1 <= nums.length <= 105
    1 <= nums[i] <= 104
 */
// Two pointers
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int res = nums.size();
        long long sum = 0;
        int l = 0, r = 0;
        
        
        while (l < nums.size()) {
            if (r < nums.size()) {
                sum += nums[r];
                r++;
            }
            while (sum >= target) {
                res = min(res, r - l);
                if (r > l) {
                    sum -= nums[l];
                    l++;
                }
            }
            if (r == nums.size() && sum < target) break;
        }
        // If we found that the sum of all elements is still less than target,
        // return 0.
        return l == 0 && sum < target ? 0 : res;
    }
};

// More concise version
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        int count = 0;
        int minC = INT_MAX;
        int sum = 0;
        int j = 0;
        for(int i = 0; i < len; ++i){
            sum += nums[i];
            count++;
            while(sum >= s){
                minC = min(minC, count);
                sum -= nums[j++];
                count --;
            }
            
        }
        return minC == INT_MAX ? 0 : minC;
    }
};

 /*
    904. Fruit Into Baskets
    https://leetcode.com/problems/fruit-into-baskets/description/
 
    You are visiting a farm that has a single row of fruit trees arranged from left to right. The trees are represented by an integer array fruits where fruits[i] is the type of fruit the ith tree produces.

    You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:

    You only have two baskets, and each basket can only hold a single type of fruit. There is no limit on the amount of fruit each basket can hold.
    Starting from any tree of your choice, you must pick exactly one fruit from every tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
    Once you reach a tree with fruit that cannot fit in your baskets, you must stop.
    Given the integer array fruits, return the maximum number of fruits you can pick.

    Example 1:

    Input: fruits = [1,2,1]
    Output: 3
    Explanation: We can pick from all 3 trees.
    Example 2:

    Input: fruits = [0,1,2,2]
    Output: 3
    Explanation: We can pick from trees [1,2,2].
    If we had started at the first tree, we would only pick from trees [0,1].
    Example 3:

    Input: fruits = [1,2,3,2,2]
    Output: 4
    Explanation: We can pick from trees [2,3,2,2].
    If we had started at the first tree, we would only pick from trees [1,2].
    

    Constraints:

    1 <= fruits.length <= 105
    0 <= fruits[i] < fruits.length
 */
// My solution:
class Solution {
public:
    int totalFruit(vector<int>& fruits) {
        map<int, int> basket;
        basket[fruits[0]] = 1;
        int finalMax = 1;
        int runningMax = 1;
        int i = 0;
        for(int j = 1; j < fruits.size(); ++j) {
            if (basket.find(fruits[j]) != basket.end() || (basket.find(fruits[j]) == basket.end() && basket.size() < 2) ) {
                basket[fruits[j]] ++;
                runningMax ++;
                finalMax = max(finalMax, runningMax);
            } else if (basket.size() == 2 && basket.find(fruits[j]) == basket.end()) {
                while (i < j) {
                    runningMax --;
                    basket[fruits[i]]--;
                    if (basket[fruits[i]] == 0) { 
                        basket.erase(fruits[i]);
                        i++;
                        break;
                    }
                    i++;
                }
                basket[fruits[j]] ++;
                runningMax ++;
                finalMax = max(finalMax, runningMax);
            }
        }
        return finalMax;
    }
};

// More elegant and trickier impl
//Sliding window. The first pointer always includes new elements
//When our basket is full, we need to move the second pointer 
//forward. In general, the time complexity is O(n)
//Exactly the same idea! However, this implementation is elegant!
//Note we do not need to move i to the designated position and 
//update each time, since i and j move with the same speed, we can
//update i only when count.size() > 2
class Solution {
public:
    int totalFruit(vector<int> &tree) {
        unordered_map<int, int> count;
        int i, j;
        for (i = 0, j = 0; j < tree.size(); ++j) {
            count[tree[j]]++;
            if (count.size() > 2) {
                if (--count[tree[i]] == 0)count.erase(tree[i]);
                i++;
            }
        }
        return j - i;
    }
};

 /*
    76. Minimum Window Substring
    https://leetcode.com/problems/minimum-window-substring/
 
    Given two strings s and t of lengths m and n respectively, return the minimum window 
    substring
    of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

    The testcases will be generated such that the answer is unique.

    

    Example 1:

    Input: s = "ADOBECODEBANC", t = "ABC"
    Output: "BANC"
    Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
    Example 2:

    Input: s = "a", t = "a"
    Output: "a"
    Explanation: The entire string s is the minimum window.
    Example 3:

    Input: s = "a", t = "aa"
    Output: ""
    Explanation: Both 'a's from t must be included in the window.
    Since the largest window of s only has one 'a', return empty string.
    

    Constraints:

    m == s.length
    n == t.length
    1 <= m, n <= 105
    s and t consist of uppercase and lowercase English letters.
    

    Follow up: Could you find an algorithm that runs in O(m + n) time?
 */
// My solution
class Solution {
public:
    string minWindow(string s, string t) {
        if (s.size() < t.size()) return "";
        unordered_map<int, int> uMap;
        for (char c : t) {
            uMap[c] ++;
        }
        int meetExp = uMap.size();
        int i = 0, j = 0;
        int res = s.size() + 1;
        int minStartIndex = -1;
        cout << meetExp << endl;
        while(i <= j && j < s.size()) {
            if (uMap.find(s[j]) != uMap.end()) {
                uMap[s[j]]--;
                if (uMap[s[j]] == 0) meetExp--;
                while (meetExp == 0 && i <= j) {
                    if (uMap.find(s[i]) != uMap.end()) {
                        uMap[s[i]]++;
                        if (uMap[s[i]] > 0) {
                            meetExp++;
                            if (res > j - i + 1) {
                                res = j - i + 1;
                                minStartIndex = i;
                            }
                            i++;
                            break;
                        }
                    }
                    i++;
                }
            }
            j++;
        }
        return minStartIndex == -1 ? "" : s.substr(minStartIndex, res);
    }
};

 /*
    54. Spiral Matrix
    https://leetcode.com/problems/spiral-matrix/
 
    Given an m x n matrix, return all elements of the matrix in spiral order.

    Example 1:
    Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
    Output: [1,2,3,6,9,8,7,4,5]

    Example 2:
    Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    Output: [1,2,3,4,8,12,11,10,9,5,6,7]
    
    Constraints:

    m == matrix.length
    n == matrix[i].length
    1 <= m, n <= 10
    -100 <= matrix[i][j] <= 100
 */
// Simulation
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = m ? matrix[0].size() : 0;
        int l = 0, r = n - 1, u = 0, b = m - 1;
        vector<int> res;
        while (true) {       
            for(int col = l; col <= r; col++) res.push_back(matrix[u][col]);
            if(++u > b) break;
           
            for(int row = u; row <= b; row++) res.push_back(matrix[row][r]);
            if(--r < l) break;

            for(int col = r; col >= l; col--) res.push_back(matrix[b][col]);
            if(--b < u) break;
            
            for(int row = b; row >= u; row--) res.push_back(matrix[row][l]);
            if(++l > r) break;
             
        }
        return res;
    }
};

 /*
    59. Spiral Matrix II
    https://leetcode.com/problems/spiral-matrix-ii/
 
    Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order.

    Example 1:
    Input: n = 3
    Output: [[1,2,3],[8,9,4],[7,6,5]]

    Example 2:
    Input: n = 1
    Output: [[1]]
    

    Constraints:
    1 <= n <= 20
 */
// Simulation: same idea!
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> finalRes (n, vector<int>(n));
        int res = 1;
        int l = 0, r = n-1, u = 0, d = n-1; 
        while(true){
            for(int col = l; col <= r; col++ ) {
                finalRes[l][col] = res;
                res++;
            }
            if(++u > d) break;;
            for(int row = u; row <= d; row++) {
                finalRes[row][r] = res;
                res++;
            }
            if(--r < l) break;
            for(int col = r; col >= l; col --){
                finalRes[d][col] = res;
                res++;
            }
            if(--d < u) break;
            for(int row = d; row>= u; row --){
                finalRes[row][l] = res;
                res++;
            }
            if(++l > r) break;
        }
        
        return finalRes;
    }
};

