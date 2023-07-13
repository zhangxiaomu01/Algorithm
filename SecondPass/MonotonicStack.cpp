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
