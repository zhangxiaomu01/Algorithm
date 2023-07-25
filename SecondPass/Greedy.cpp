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

 /*
    55. Jump Game
    https://leetcode.com/problems/jump-game/
    You are given an integer array nums. You are initially positioned at the array's first index, and each element in the 
    array represents your maximum jump length at that position.

    Return true if you can reach the last index, or false otherwise.


    Example 1:
    Input: nums = [2,3,1,1,4]
    Output: true
    Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

    Example 2:
    Input: nums = [3,2,1,0,4]
    Output: false
    Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
    

    Constraints:
    1 <= nums.length <= 10^4
    0 <= nums[i] <= 10^5
 */
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int maxRange = 0;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            if (maxRange < i) return false;
            maxRange = max(maxRange, i + nums[i]);
            if (maxRange >= n-1) return true;
        }
        return maxRange >= n-1;
    }
};

 /*
    45. Jump Game II
    https://leetcode.com/problems/jump-game-ii/
    You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].

    Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], 
    you can jump to any nums[i + j] where:
    0 <= j <= nums[i] and
    i + j < n
    Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].

    

    Example 1:
    Input: nums = [2,3,1,1,4]
    Output: 2
    Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the 
    last index.

    Example 2:
    Input: nums = [2,3,0,1,4]
    Output: 2
    

    Constraints:
    1 <= nums.length <= 10^4
    0 <= nums[i] <= 1000
    It's guaranteed that you can reach nums[n - 1].
 */
class Solution {
public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        // We only increase the # of steps if we reaches the current max range
        int curMaxRange = 0;
        int nextMaxRange = 0;
        int res = 0;
        if (nums.size() <= 1) return 0;
        for (int i = 0; i < n; ++i) {   
            if (i > nextMaxRange) return 0;
            nextMaxRange = max(nextMaxRange, nums[i] + i);
            // We reached the maximum range we can currently reach, starting with 0
            if (i == curMaxRange) {
                res++;
                // Record the next step that the distance we can reach.
                curMaxRange = nextMaxRange;
                if (curMaxRange >= n-1) return res;
            }
        } 
        return res;
    }
};

 /*
    1005. Maximize Sum Of Array After K Negations
    https://leetcode.com/problems/maximize-sum-of-array-after-k-negations/
    Given an integer array nums and an integer k, modify the array in the following way:

    choose an index i and replace nums[i] with -nums[i].
    You should apply this process exactly k times. You may choose the same index i multiple times.

    Return the largest possible sum of the array after modifying it in this way.


    Example 1:
    Input: nums = [4,2,3], k = 1
    Output: 5
    Explanation: Choose index 1 and nums becomes [4,-2,3].

    Example 2:
    Input: nums = [3,-1,0,2], k = 3
    Output: 6
    Explanation: Choose indices (1, 2, 2) and nums becomes [3,1,0,2].

    Example 3:
    Input: nums = [2,-3,-1,5,-4], k = 2
    Output: 13
    Explanation: Choose indices (1, 4) and nums becomes [2,3,-1,5,4].
    

    Constraints:
    1 <= nums.length <= 10^4
    -100 <= nums[i] <= 100
    1 <= k <= 10^4
 */
class Solution {
public:
    int largestSumAfterKNegations(vector<int>& nums, int k) {
        // Sort the array from largest to the smallest based on the abs(nums[i]).
        auto comp = [](int x, int y) {
            return abs(x) > abs(y);
        };
        sort(nums.begin(), nums.end(), comp);
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] < 0 && k > 0) {
                nums[i] = -nums[i];
                k--;
            }
        }
        if (k > 0 && k % 2 == 1) nums[nums.size() - 1] = -nums[nums.size() - 1];
        int res = 0;
        for (int i : nums) res += i;
        return res; 
    }
};

 /*
    134. Gas Station
    https://leetcode.com/problems/gas-station/
    There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

    You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th 
    station. You begin the journey with an empty tank at one of the gas stations.

    Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in 
    the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique


    Example 1:
    Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
    Output: 3
    Explanation:
    Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
    Travel to station 4. Your tank = 4 - 1 + 5 = 8
    Travel to station 0. Your tank = 8 - 2 + 1 = 7
    Travel to station 1. Your tank = 7 - 3 + 2 = 6
    Travel to station 2. Your tank = 6 - 4 + 3 = 5
    Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
    Therefore, return 3 as the starting index.

    Example 2:
    Input: gas = [2,3,4], cost = [3,4,3]
    Output: -1
    Explanation:
    You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
    Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
    Travel to station 0. Your tank = 4 - 3 + 2 = 3
    Travel to station 1. Your tank = 3 - 3 + 3 = 3
    You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
    Therefore, you can't travel around the circuit once no matter where you start.
    

    Constraints:
    n == gas.length == cost.length
    1 <= n <= 10^5
    0 <= gas[i], cost[i] <= 10^4
 */
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();
        int curSum = 0;
        int totalSum = 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            curSum += gas[i] - cost[i];
            totalSum += gas[i] - cost[i];
            if (curSum < 0) {
                // if curSum < 0, we know [0..i] can't be the starting point.
                res = i + 1;
                curSum = 0;
            }
        }
        // No way to finish the loop!
        if (totalSum < 0) return -1;
        return res;
    }
};

 /*
    135. Candy
    https://leetcode.com/problems/candy/
    There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.

    You are giving candies to these children subjected to the following requirements:
    Each child must have at least one candy.
    Children with a higher rating get more candies than their neighbors.
    Return the minimum number of candies you need to have to distribute the candies to the children.

    
    Example 1:
    Input: ratings = [1,0,2]
    Output: 5
    Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.

    Example 2:
    Input: ratings = [1,2,2]
    Output: 4
    Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
    The third child gets 1 candy because it satisfies the above two conditions.
    

    Constraints:
    n == ratings.length
    1 <= n <= 2 * 10^4
    0 <= ratings[i] <= 2 * 10^4
 */
class Solution {
public:
    int candy(vector<int>& ratings) {
        int n = ratings.size();
        ratings.insert(ratings.begin(), INT_MAX);
        ratings.push_back(INT_MAX);
        vector<int> res(n+2, 1);
        for (int i = 1; i < ratings.size() - 1; ++i) {
            if (ratings[i] > ratings[i-1]) res[i] = res[i-1] + 1;
        }
        for (int i = ratings.size() - 2; i >= 1; --i) {
            if (ratings[i] > ratings[i+1]) res[i] = max(res[i], res[i+1] + 1);
        }
        res.erase(res.begin());
        res.pop_back();
        int numOfCandies = 0;
        for (int e : res) numOfCandies += e;
        return numOfCandies;
    }
};
