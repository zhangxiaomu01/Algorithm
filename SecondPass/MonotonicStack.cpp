/**
 * @file MonotonicStack.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2023-07-13
 * 
 * A quick second pass with the common 'MonotonicStack' algorithm problems.
 */
 /*
    739. Daily Temperatures
    https://leetcode.com/problems/daily-temperatures/
    Given an array of integers temperatures represents the daily temperatures, return an array 
    answer such that answer[i] is the number of days you have to wait after the ith day to get 
    a warmer temperature. If there is no future day for which this is possible, keep 
    answer[i] == 0 instead.


    Example 1:
    Input: temperatures = [73,74,75,71,69,72,76,73]
    Output: [1,1,4,2,1,1,0,0]

    Example 2:
    Input: temperatures = [30,40,50,60]
    Output: [1,1,1,0]

    Example 3:
    Input: temperatures = [30,60,90]
    Output: [1,1,0]
    
    Constraints:
    1 <= temperatures.length <= 10^5
    30 <= temperatures[i] <= 100
 */
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& t) {
        int n = t.size();
        vector<int> res(n, 0);
        stack<int> st;
        st.push(n-1);
        for (int i = n - 2; i >= 0; --i) {
            while(!st.empty() && t[i] >= t[st.top()]) {
                st.pop();
            } 
            if (!st.empty()) res[i] = st.top() - i;
            st.push(i);
        }
        return res;
    }
};

 /*
    42. Trapping Rain Water
    https://leetcode.com/problems/trapping-rain-water/
    Given n non-negative integers representing an elevation map where the width of each bar is 1, 
    compute how much water it can trap after raining.

    Example 1:
    Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
    Output: 6
    Explanation: The above elevation map (black section) is represented by array 
    [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.

    Example 2:
    Input: height = [4,2,0,3,2,5]
    Output: 9
    
    Constraints:
    n == height.length
    1 <= n <= 2 * 10^4
    0 <= height[i] <= 10^5
 */
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        // Maintains the decreasing stack (the top element is the smallest).
        // Stack will saves the index of the curresponding height.
        stack<int> st;
        int res = 0;
        st.push(0);

        for (int i = 1; i < n; ++i) {
            while(!st.empty() && height[i] > height[st.top()]) {
                int index = st.top();
                int currentHeight = height[index];
                st.pop();
                if (!st.empty()) {
                    
                    int heightLeft = height[st.top()];
                    int currentArea = 
                    (min(heightLeft, height[i]) - currentHeight) * (i - st.top() - 1);
                    res += currentArea;
                }
            }
            if (!st.empty() && height[i] == height[st.top()]) {
                st.pop();
            }
            st.push(i);
        }
        return res;
    }
};

// Two pointers
class Solution {
public:
    int trap(vector<int>& height) {
        if(height.size() <= 1) return 0;
        int len = height.size();
        int left[len] = {0}, right[len] = {0};
        int maxLeft = 0, maxRight = 0;
        for(int i = 0; i < len; ++i){
            maxLeft = max(maxLeft, height[i]);
            left[i] = maxLeft;
            maxRight = max(maxRight, height[len-1-i]);
            right[len-1 -i] = maxRight; 
        }
        int res = 0;
        for(int i = 1; i < len-1; ++i){
            res += max(0, min(left[i], right[i]) - height[i]);
        }
        return res;
    }
};

 /*
    84. Largest Rectangle in Histogram
    https://leetcode.com/problems/largest-rectangle-in-histogram/
    Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return 
    the area of the largest rectangle in the histogram.

    Example 1:
    Input: heights = [2,1,5,6,2,3]
    Output: 10
    Explanation: The above is a histogram where width of each bar is 1.
    The largest rectangle is shown in the red area, which has an area = 10 units.

    Example 2:
    Input: heights = [2,4]
    Output: 4
    

    Constraints:
    1 <= heights.length <= 10^5
    0 <= heights[i] <= 10^4
 */
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        // Adds the placeholder.
         heights.insert(heights.begin(), 0);
         heights.push_back(0);

         // increasing stack, note that it saves the indices.
         stack<int> st;
         st.push(0);
         int n = heights.size();
         int res = 0;

         for (int i = 1; i < n; ++i) {
             while(!st.empty() && heights[i] < heights[st.top()]) {
                int index = st.top();
                int currentHeight = heights[index];
                st.pop();
                if (!st.empty()) {
                    int left = st.top();
                    int right = i;
                    int width = right - left - 1;
                    res = max(res, currentHeight * width);
                }
             }
             // We can safely ignore the heights[i] == heights[st.top()] condition
             // Because in the end, we always hit height == 0, then we will calculate
             // res correctly!
             st.push(i);
         }
        return res;
    }
};

// Two pointer: not my solution / Not easy to implement!
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        vector<int> minLeftIndex(heights.size());
        vector<int> minRightIndex(heights.size());
        int size = heights.size();

        // 记录每个柱子 左边第一个小于该柱子的下标
        minLeftIndex[0] = -1; // 注意这里初始化，防止下面while死循环
        for (int i = 1; i < size; i++) {
            int t = i - 1;
            // 这里不是用if，而是不断向左寻找的过程
            while (t >= 0 && heights[t] >= heights[i]) t = minLeftIndex[t];
            minLeftIndex[i] = t;
        }
        // 记录每个柱子 右边第一个小于该柱子的下标
        minRightIndex[size - 1] = size; // 注意这里初始化，防止下面while死循环
        for (int i = size - 2; i >= 0; i--) {
            int t = i + 1;
            // 这里不是用if，而是不断向右寻找的过程
            while (t < size && heights[t] >= heights[i]) t = minRightIndex[t];
            minRightIndex[i] = t;
        }
        // 求和
        int result = 0;
        for (int i = 0; i < size; i++) {
            int sum = heights[i] * (minRightIndex[i] - minLeftIndex[i] - 1);
            result = max(sum, result);
        }
        return result;
    }
};

 /*
    496. Next Greater Element I
    https://leetcode.com/problems/next-greater-element-i/
    The next greater element of some element x in an array is the first greater element that is to the right of x in the 
    same array.

    You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.

    For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element 
    of nums2[j] in nums2. If there is no next greater element, then the answer for this query is -1.

    Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.

    
    Example 1:
    Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
    Output: [-1,3,-1]
    Explanation: The next greater element for each value of nums1 is as follows:
    - 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
    - 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
    - 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.

    Example 2:
    Input: nums1 = [2,4], nums2 = [1,2,3,4]
    Output: [3,-1]
    Explanation: The next greater element for each value of nums1 is as follows:
    - 2 is underlined in nums2 = [1,2,3,4]. The next greater element is 3.
    - 4 is underlined in nums2 = [1,2,3,4]. There is no next greater element, so the answer is -1.
    

    Constraints:
    1 <= nums1.length <= nums2.length <= 1000
    0 <= nums1[i], nums2[i] <= 10^4
    All integers in nums1 and nums2 are unique.
    All the integers of nums1 also appear in nums2.

    Follow up: Could you find an O(nums1.length + nums2.length) solution?
 */
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        vector<int> res(m, -1);
        // the key is the element, the value is the index in nums1.
        unordered_map<int, int> uMap;
        for (int i = 0; i < m; ++i) uMap[nums1[i]] = i;
        // st saves the index of the decreasing stack (the top is the smallest)
        stack<int> st;
        st.push(n-1);
        for (int i = n - 2; i >= 0; --i) {
            while (!st.empty() && nums2[i] >= nums2[st.top()]) {
                st.pop();
            }
            // We know that nums[i] in in nums1.
            if (!st.empty() && uMap.find(nums2[i]) != uMap.end()) {
                int index = uMap[nums2[i]];
                res[index] = nums2[st.top()];
            }
            st.push(i);
        }
        return res;
    }
};

 /*
    503. Next Greater Element II
    https://leetcode.com/problems/next-greater-element-ii/
    Given a circular integer array nums (i.e., the next element of nums[nums.length - 1] is nums[0]), return the next 
    greater number for every element in nums.

    The next greater number of a number x is the first greater number to its traversing-order next in the array, which 
    means you could search circularly to find its next greater number. If it doesn't exist, return -1 for this number.

    

    Example 1:
    Input: nums = [1,2,1]
    Output: [2,-1,2]
    Explanation: The first 1's next greater number is 2; 
    The number 2 can't find next greater number. 
    The second 1's next greater number needs to search circularly, which is also 2.

    Example 2:
    Input: nums = [1,2,3,4,3]
    Output: [2,3,4,-1,4]
    

    Constraints:
    1 <= nums.length <= 10^4
    -10^9 <= nums[i] <= 10^9
 */
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        vector<int> nums1 = nums;
        nums.insert(nums.end(), nums1.begin(), nums1.end());
        vector<int> res(nums.size(), -1);
        // st records the index of the element
        stack<int> st;
        st.push(0);
        for (int i = 1; i < nums.size(); ++i) {
            while (!st.empty() && nums[i] > nums[st.top()]) {
                res[st.top()] = nums[i];
                st.pop();
            }
            st.push(i);
        }

        res.resize(nums1.size());
        return res;
    }
};

// Space optimize solution: not mine!
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        vector<int> result(nums.size(), -1);
        if (nums.size() == 0) return result;
        stack<int> st;
        for (int i = 0; i < nums.size() * 2; i++) {
            while (!st.empty() && nums[i % nums.size()] > nums[st.top()]) {
                result[st.top()] = nums[i % nums.size()];
                st.pop();
            }
            st.push(i % nums.size());
        }
        return result;
    }
};

