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

 /*
    860. Lemonade Change
    https://leetcode.com/problems/lemonade-change/
    At a lemonade stand, each lemonade costs $5. Customers are standing in a queue to buy from 
    you and order one at a time (in the order specified by bills). Each customer will only buy 
    one lemonade and pay with either a $5, $10, or $20 bill. You must provide the correct change 
    to each customer so that the net transaction is that the customer pays $5.

    Note that you do not have any change in hand at first.

    Given an integer array bills where bills[i] is the bill the ith customer pays, return true 
    if you can provide every customer with the correct change, or false otherwise.

    

    Example 1:
    Input: bills = [5,5,5,10,20]
    Output: true
    Explanation: 
    From the first 3 customers, we collect three $5 bills in order.
    From the fourth customer, we collect a $10 bill and give back a $5.
    From the fifth customer, we give a $10 bill and a $5 bill.
    Since all customers got correct change, we output true.

    Example 2:
    Input: bills = [5,5,10,10,20]
    Output: false
    Explanation: 
    From the first two customers in order, we collect two $5 bills.
    For the next two customers in order, we collect a $10 bill and give back a $5 bill.
    For the last customer, we can not give the change of $15 back because we only have two $10 bills.
    Since not every customer received the correct change, the answer is false.
    
    Constraints:
    1 <= bills.length <= 10^5
    bills[i] is either 5, 10, or 20.
 */
class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        // money i represents # of 5 / 10 / 20 dollars
        int money[3] = {0, 0, 0};
        for (int i = 0; i < bills.size(); ++i) {
            if (bills[i] == 5) money[0] ++;
            else if (bills[i] == 10) {
                if (money[0] <= 0) return false;
                money[0]--;
                money[1]++;
            } else {
                if (money[0] <= 0) return false;
                if (money[1] > 0) {
                    money[1]--;
                    money[0]--;
                    money[2]++;
                } else {
                    money[0] -= 3;
                    if (money[0] < 0) return false;
                }
            }
        }
        return true;
    }
};

 /*
    406. Queue Reconstruction by Height
    https://leetcode.com/problems/queue-reconstruction-by-height/
    You are given an array of people, people, which are the attributes of some people in a queue
     (not necessarily in order). Each people[i] = [hi, ki] represents the ith person of height hi
      with exactly ki other people in front who have a height greater than or equal to hi.

    Reconstruct and return the queue that is represented by the input array people. The returned
    queue should be formatted as an array queue, where queue[j] = [hj, kj] is the attributes of
    the jth person in the queue (queue[0] is the person at the front of the queue).


    Example 1:
    Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
    Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    Explanation:
    Person 0 has height 5 with no other people taller or the same height in front.
    Person 1 has height 7 with no other people taller or the same height in front.
    Person 2 has height 5 with two persons taller or the same height in front, which is person 0 and 1.
    Person 3 has height 6 with one person taller or the same height in front, which is person 1.
    Person 4 has height 4 with four people taller or the same height in front, which are people 0, 1, 2, and 3.
    Person 5 has height 7 with one person taller or the same height in front, which is person 1.
    Hence [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] is the reconstructed queue.

    Example 2:
    Input: people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
    Output: [[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
    
    Constraints:
    1 <= people.length <= 2000
    0 <= hi <= 10^6
    0 <= ki < people.length
    It is guaranteed that the queue can be reconstructed.
 */
class Solution {
public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        auto comp = [](vector<int>& p1, vector<int>& p2) {
            if (p1[0] == p2[0]) return p1[1] < p2[1];
            return p1[0] > p2[0];
        };
        sort(people.begin(), people.end(), comp);
        list<vector<int>> res;
        // We sort the array based on height, we can safely insert the higher person to the 
        // given index when evaluated!
        for (int i = 0; i < people.size(); ++i) {
            int index = people[i][1];
            auto it = res.begin();
            while (index > 0) {
                index--;
                it++;
            }
            res.insert(it, people[i]);
        }
        vector<vector<int>> result(res.begin(), res.end());
        return result;
    }
};

 /*
    452. Minimum Number of Arrows to Burst Balloons
    https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/
    There are some spherical balloons taped onto a flat wall that represents the XY-plane. 
    The balloons are represented as a 2D integer array points where points[i] = [xstart, xend] 
    denotes a balloon whose horizontal diameter stretches between xstart and xend. You do not 
    know the exact y-coordinates of the balloons.

    Arrows can be shot up directly vertically (in the positive y-direction) from different points 
    along the x-axis. A balloon with xstart and xend is burst by an arrow shot at x if 
    xstart <= x <= xend. There is no limit to the number of arrows that can be shot. A shot arrow 
    keeps traveling up infinitely, bursting any balloons in its path.

    Given the array points, return the minimum number of arrows that must be shot to burst all 
    balloons.


    Example 1:
    Input: points = [[10,16],[2,8],[1,6],[7,12]]
    Output: 2
    Explanation: The balloons can be burst by 2 arrows:
    - Shoot an arrow at x = 6, bursting the balloons [2,8] and [1,6].
    - Shoot an arrow at x = 11, bursting the balloons [10,16] and [7,12].

    Example 2:
    Input: points = [[1,2],[3,4],[5,6],[7,8]]
    Output: 4
    Explanation: One arrow needs to be shot for each balloon for a total of 4 arrows.

    Example 3:
    Input: points = [[1,2],[2,3],[3,4],[4,5]]
    Output: 2
    Explanation: The balloons can be burst by 2 arrows:
    - Shoot an arrow at x = 2, bursting the balloons [1,2] and [2,3].
    - Shoot an arrow at x = 4, bursting the balloons [3,4] and [4,5].
    

    Constraints:
    1 <= points.length <= 10^5
    points[i].length == 2
    -2^31 <= xstart < xend <= 2^31 - 1
 */
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        auto comp = [](vector<int>& p1, vector<int>& p2) {
            return p1[0] < p2[0];
        };
        sort(points.begin(), points.end(), comp);
        int res = 1;
        int minLength = points[0][1];
        for (int i = 1; i < points.size(); ++i) {
            // If no overlap with the minimum x_end from previous checked groups
            if (points[i][0] > minLength) {
                res++;
                minLength = points[i][1];
                continue;
            }
            minLength = min(minLength, points[i][1]);
        }
        return res;
    }
};

 /*
    435. Non-overlapping Intervals
    https://leetcode.com/problems/non-overlapping-intervals/
    Given an array of intervals intervals where intervals[i] = [starti, endi], return the 
    minimum number of intervals you need to remove to make the rest of the intervals 
    non-overlapping.


    Example 1:
    Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
    Output: 1
    Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

    Example 2:
    Input: intervals = [[1,2],[1,2],[1,2]]
    Output: 2
    Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.

    Example 3:
    Input: intervals = [[1,2],[2,3]]
    Output: 0
    Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
    

    Constraints:
    1 <= intervals.length <= 10^5
    intervals[i].length == 2
    -5 * 10^4 <= starti < endi <= 5 * 10^4
 */
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        auto comp  = [](vector<int>& p1, vector<int>& p2) {
            if (p1[0] == p2[0]) return p1[1] < p2[1];
            return p1[0] < p2[0];
        };
        sort(intervals.begin(), intervals.end(), comp);
        int minLength = intervals[0][1];
        int res = 0;
        for (int i = 1; i < intervals.size(); ++i) {
            if (intervals[i][0] >= minLength) {
                minLength = intervals[i][1];
                continue;
            }
            minLength = min(minLength, intervals[i][1]);
            res++;
        }
        return res;
    }
};

 /*
    763. Partition Labels
    https://leetcode.com/problems/partition-labels/
    You are given a string s. We want to partition the string into as many parts as possible so 
    that each letter appears in at most one part.

    Note that the partition is done so that after concatenating all the parts in order, the 
    resultant string should be s.

    Return a list of integers representing the size of these parts.


    Example 1:
    Input: s = "ababcbacadefegdehijhklij"
    Output: [9,7,8]
    Explanation:
    The partition is "ababcbaca", "defegde", "hijhklij".
    This is a partition so that each letter appears in at most one part.
    A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits s into less parts.
    
    Example 2:
    Input: s = "eccbbbbdec"
    Output: [10]
    
    Constraints:
    1 <= s.length <= 500
    s consists of lowercase English letters.
 */
class Solution {
public:
    vector<int> partitionLabels(string s) {
        int hash[26] = {0};
        int n = s.size();
        // We save the furthest index that a character can appear.
        for (int i = 0; i < n; ++i) {
            hash[s[i] - 'a'] = i;
        }
        // Use two pointers to keep track of the result.
        int l = 0, r = 0;
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            r = max(r, hash[s[i]-'a']);
            // If we found one furthest bound, we need to save the result.
            if (i == r) {
                res.push_back(r - l + 1);
                l = i + 1;
            }
        }
        return res;
    }
};

 /*
    738. Monotone Increasing Digits
    https://leetcode.com/problems/monotone-increasing-digits/
    An integer has monotone increasing digits if and only if each pair of adjacent digits x and y 
    satisfy x <= y.

    Given an integer n, return the largest number that is less than or equal to n with monotone 
    increasing digits.

    
    Example 1:
    Input: n = 10
    Output: 9

    Example 2:
    Input: n = 1234
    Output: 1234

    Example 3:
    Input: n = 332
    Output: 299
    
    Constraints:
    0 <= n <= 10^9
 */
class Solution {
public:
    int monotoneIncreasingDigits(int n) {
        string digits = to_string(n);
        if (digits.size() <= 1) return stoi(digits);
        cout << digits << endl;
        int len = digits.size();
        int startOfNine = INT_MAX;
        for (int i = len - 1; i >= 1; --i) {
            // Whenever we detect y > x, then we decrement x and make y to be 9 (the largest)
            if (digits[i] < digits[i-1]) {
                startOfNine = i;
                digits[i] = '9';
                digits[i-1] --;
            }
        }
        // For special case 1000 etc.
        for (int i = startOfNine; i < len; ++i) digits[i] = '9';

        return stoi(digits);
    }
};
