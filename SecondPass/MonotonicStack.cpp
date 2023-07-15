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
