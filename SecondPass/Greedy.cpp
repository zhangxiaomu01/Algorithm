/**
 * @file Greedy.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2022-07-19
 * 
 * A quick second pass with the common 'Greedy' algorithm problems.
 * 
 */

 /*
    455. Assign Cookies
    https://leetcode.com/problems/assign-cookies/
    Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most 
    one cookie.

    Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with; and 
    each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will be 
    content. Your goal is to maximize the number of your content children and output the maximum number.

    
    Example 1:
    Input: g = [1,2,3], s = [1,1]
    Output: 1
    Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
    And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
    You need to output 1.

    Example 2:
    Input: g = [1,2], s = [1,2,3]
    Output: 2
    Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2. 
    You have 3 cookies and their sizes are big enough to gratify all of the children, 
    You need to output 2.
    

    Constraints:
    1 <= g.length <= 3 * 10^4
    0 <= s.length <= 3 * 10^4
    1 <= g[i], s[j] <= 2^31 - 1
 */
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        int i = g.size() - 1;
        int j = s.size() - 1;
        int res = 0;
        while (i >= 0) {
            if (j >= 0 && g[i] <= s[j]) {
                res++;
                j--;
            }
            i--;
        }
        return res;
    }
};

 /*
    376. Wiggle Subsequence
    https://leetcode.com/problems/wiggle-subsequence/
    A wiggle sequence is a sequence where the differences between successive numbers strictly alternate between positive 
    and negative. The first difference (if one exists) may be either positive or negative. A sequence with one element 
    and a sequence with two non-equal elements are trivially wiggle sequences.

    For example, [1, 7, 4, 9, 2, 5] is a wiggle sequence because the differences (6, -3, 5, -7, 3) alternate between positive 
    and negative. In contrast, [1, 4, 7, 2, 5] and [1, 7, 4, 5, 5] are not wiggle sequences. The first is not because its 
    first two differences are positive, and the second is not because its last difference is zero.
    A subsequence is obtained by deleting some elements (possibly zero) from the original sequence, leaving the remaining 
    elements in their original order.

    Given an integer array nums, return the length of the longest wiggle subsequence of nums.

    
    Example 1:
    Input: nums = [1,7,4,9,2,5]
    Output: 6
    Explanation: The entire sequence is a wiggle sequence with differences (6, -3, 5, -7, 3).

    Example 2:
    Input: nums = [1,17,5,10,13,15,10,5,16,8]
    Output: 7
    Explanation: There are several subsequences that achieve this length.
    One is [1, 17, 10, 13, 10, 16, 8] with differences (16, -7, 3, -3, 6, -8).

    Example 3:
    Input: nums = [1,2,3,4,5,6,7,8,9]
    Output: 2
    

    Constraints:
    1 <= nums.length <= 1000
    0 <= nums[i] <= 1000
 */
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        // By default, we assume that the rightmost of the sequnce have a valley or
        // peak.
        int res = 1;
        int curDiff = 0;
        int preDiff = 0;
        // The idea is to keep the extremes(peak or valley), then we will get the
        // result.
        for (int i = 0; i < nums.size() - 1; ++i) {
            curDiff = nums[i+1] - nums[i];
            if ((preDiff <= 0 && curDiff > 0) || (preDiff >= 0 && curDiff < 0)) {
                res++;
                preDiff = curDiff;
            }
        }
        return res;
    }
};
