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
