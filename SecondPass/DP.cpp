/**
 * @file DP.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2022-06-18
 * 
 * A quick second pass with the common 'Dynamic programming' algorithm problems.
 * 
 */

 /*
    509. Fibonacci Number
    https://leetcode.com/problems/fibonacci-number/
    The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci 
    sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. 
    That is,

    F(0) = 0, F(1) = 1
    F(n) = F(n - 1) + F(n - 2), for n > 1.
    Given n, calculate F(n).

    

    Example 1:
    Input: n = 2
    Output: 1
    Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.

    Example 2:
    Input: n = 3
    Output: 2
    Explanation: F(3) = F(2) + F(1) = 1 + 1 = 2.

    Example 3:
    Input: n = 4
    Output: 3
    Explanation: F(4) = F(3) + F(2) = 2 + 1 = 3.
    

    Constraints:
    0 <= n <= 30
 */
class Solution {
public:
    int fib(int n) {
        if (n<=1) return n;
        vector<int> dp(n+1, 0);
        dp[1] = 1;
        for (int i = 2; i <= n; ++i) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
};

 /*
    70. Climbing Stairs
    https://leetcode.com/problems/climbing-stairs/
    You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?


    Example 1:
    Input: n = 2
    Output: 2
    Explanation: There are two ways to climb to the top.
    1. 1 step + 1 step
    2. 2 steps

    Example 2:
    Input: n = 3
    Output: 3
    Explanation: There are three ways to climb to the top.
    1. 1 step + 1 step + 1 step
    2. 1 step + 2 steps
    3. 2 steps + 1 step
    

    Constraints:
    1 <= n <= 45
 */
class Solution {
public:
    int climbStairs(int n) {
        if (n <= 1) return n;
        vector<int> dp(n+1, 0);
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 2; i <= n; ++i) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
};

 /*
    746. Min Cost Climbing Stairs
    https://leetcode.com/problems/min-cost-climbing-stairs/
    You are given an integer array cost where cost[i] is the cost of ith step on a staircase. 
    Once you pay the cost, you can either climb one or two steps.

    You can either start from the step with index 0, or the step with index 1.

    Return the minimum cost to reach the top of the floor.

 

    Example 1:
    Input: cost = [10,15,20]
    Output: 15
    Explanation: You will start at index 1.
    - Pay 15 and climb two steps to reach the top.
    The total cost is 15.

    Example 2:
    Input: cost = [1,100,1,1,1,100,1,1,100,1]
    Output: 6
    Explanation: You will start at index 0.
    - Pay 1 and climb two steps to reach index 2.
    - Pay 1 and climb two steps to reach index 4.
    - Pay 1 and climb two steps to reach index 6.
    - Pay 1 and climb one step to reach index 7.
    - Pay 1 and climb two steps to reach index 9.
    - Pay 1 and climb one step to reach the top.
    The total cost is 6.
    

    Constraints:
    2 <= cost.length <= 1000
    0 <= cost[i] <= 999
 */
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        if (n <= 1) return cost[0];
        vector<int> dp(n, 0);
        dp[0] = cost[0];
        dp[1] = cost[1];
        for (int i = 2; i < n; ++i) {
            dp[i] = min(dp[i-1] + cost[i], dp[i-2] + cost[i]);
        }
        // Note we need to compare the situation that we land at the last stair 
        // and the step before last stair.
        return min(dp[n-2], dp[n-1]);
    }
};

 /*
    62. Unique Paths
    https://leetcode.com/problems/unique-paths/
    There is a robot on an m x n grid. The robot is initially located at the top-left corner 
    (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

    Given the two integers m and n, return the number of possible unique paths that the robot can 
    take to reach the bottom-right corner.

    The test cases are generated so that the answer will be less than or equal to 2 * 109.

    

    Example 1:
    Input: m = 3, n = 7
    Output: 28

    Example 2:
    Input: m = 3, n = 2
    Output: 3
    Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
    1. Right -> Down -> Down
    2. Down -> Down -> Right
    3. Down -> Right -> Down
    

    Constraints:
    1 <= m, n <= 100
 */
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 0));
        for (int i = 0; i < n; ++i) dp[0][i] = 1;
        for (int i = 0; i < m; ++i) dp[i][0] = 1;

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};

class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> dp(n,  1);

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[j] += dp[j-1];
            }
        }
        return dp[n-1];
    }
};

 /*
    63. Unique Paths II
    https://leetcode.com/problems/unique-paths-ii/
    You are given an m x n integer array grid. There is a robot initially located at the top-left 
    corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner 
    (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

    An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes 
    cannot include any square that is an obstacle.

    Return the number of possible unique paths that the robot can take to reach the bottom-right 
    corner.

    The testcases are generated so that the answer will be less than or equal to 2 * 109.

    

    Example 1:
    Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
    Output: 2
    Explanation: There is one obstacle in the middle of the 3x3 grid above.
    There are two ways to reach the bottom-right corner:
    1. Right -> Right -> Down -> Down
    2. Down -> Down -> Right -> Right

    Example 2:
    Input: obstacleGrid = [[0,1],[0,0]]
    Output: 1
    
    Constraints:
    m == obstacleGrid.length
    n == obstacleGrid[i].length
    1 <= m, n <= 100
    obstacleGrid[i][j] is 0 or 1.
 */
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        if (obstacleGrid[0][0] == 1) return 0;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        for (int i = 0; i < m; ++i) {
            if (obstacleGrid[i][0] == 1) break;
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; ++i) {
            if (obstacleGrid[0][i] == 1) break;
            dp[0][i] = 1;
        }

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (obstacleGrid[i][j] == 1) dp[i][j] = 0;
                else {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }
        return dp[m-1][n-1];
    }
};

class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        if (obstacleGrid[0][0] == 1) return 0;
        vector<int> dp(n, 0);
        for (int i = 0; i < n; ++i) {
            if (obstacleGrid[0][i] == 1) break;
            dp[i] = 1;
        }

        for (int i = 1; i < m; ++i) {
            // We need to start with 0 to handle the case that obstacleGrid[i][0] == 1
            for (int j = 0; j < n; ++j) {
                if (obstacleGrid[i][j] == 1) dp[j] = 0;
                else if (j > 0) {
                    dp[j] += dp[j-1];
                }
            }
        }
        return dp[n-1];
    }
};


 /*
    343. Integer Break
    https://leetcode.com/problems/integer-break/
    Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize 
    the product of those integers.

    Return the maximum product you can get.

    
    Example 1:
    Input: n = 2
    Output: 1
    Explanation: 2 = 1 + 1, 1 × 1 = 1.

    Example 2:
    Input: n = 10
    Output: 36
    Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
    
    Constraints:
    2 <= n <= 58
 */
class Solution {
public:
    int integerBreak(int n) {
        // dp[i] represents the maximum possible product results of integer i.
        // It also implies we at least divide i into 2 smaller integers.
        vector<int> dp(n+1, 0);
        dp[1] = 1;
        dp[2] = 1;
        for (int i = 3; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                // Given dp[i-j] represents the maximum possible product of i-j, it also means
                // we divide i-j into at least 2 integers, so we need to check j*(i-j) to get final
                // result.
                dp[i] = max(dp[i], max(j*(i - j), j * dp[i-j]));
            }
        }
        return dp[n];
    }
};

 /*
    96. Unique Binary Search Trees
    https://leetcode.com/problems/unique-binary-search-trees/
    Given an integer n, return the number of structurally unique BST's (binary search trees) which 
    has exactly n nodes of unique values from 1 to n.

 
    Example 1:
    Input: n = 3
    Output: 5

    Example 2:
    Input: n = 1
    Output: 1
    

    Constraints:
    1 <= n <= 19
 */
class Solution {
public:
    int numTrees(int n) {
        // Dp[i] means the maximum unique BST's with node number i.
         vector<int> dp(n+1, 0);
         // Initialize dp[0] = 1, so when we do dp[left] * dp[right], we won't get 0;
         dp[0] = 1;
         dp[1] = 1;

         for (int i = 2; i <= n; ++i) {
             // i represents the number of nodes from the left tree;
             for (int j = 0; j <= i - 1; ++j) {
                 dp[i] += dp[j] * dp[i-1-j];
             }
         }

         return dp[n];
    }
};

 /*
    152. Maximum Product Subarray
    https://leetcode.com/problems/unique-binary-search-trees/
    Given an integer array nums, find a subarray that has the largest product, and 
    return the product.

    The test cases are generated so that the answer will fit in a 32-bit integer.

    

    Example 1:
    Input: nums = [2,3,-2,4]
    Output: 6
    Explanation: [2,3] has the largest product 6.

    Example 2:
    Input: nums = [-2,0,-1]
    Output: 0
    Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
    

    Constraints:
    1 <= nums.length <= 2 * 104
    -10 <= nums[i] <= 10
    The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
 */
// DP: Saves the previous minimum product and maximum product, then derive the results on the fly.
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        // We need to save the previous minimum negative and maximum positive result,
        // then we can derive the maximum with current nums[i].
        int preMin = nums[0], preMax = nums[0];
        int res = nums[0];
        for (int i = 1; i < nums.size(); ++i) {
            // Once we encounter 0, we need to reset state and start over again.
            if (nums[i] == 0) {
                res= max(nums[i], res);
                preMin = 0; 
                preMax = 0;
            } else {
                int curMin = INT_MAX, curMax = INT_MIN;
                if (nums[i] > 0) {
                    // Note once the current nums[i] is less than the preMin or greater than preMax,
                    // we need to make sure to only include it.
                    curMin = min(preMin * nums[i], nums[i]);
                    curMax = max(preMax * nums[i], nums[i]);

                } else {
                    curMin = min(preMax * nums[i], nums[i]);
                    curMax = max(preMin * nums[i], nums[i]);
                }
                res = max (res, curMax);
                preMin = curMin;
                preMax = curMax;
            } 
        }
        return res;
    }
};

// Find the invariance.
/*
We only need to consider 3 situations:
There is no 0 in the array:
1. if we contains even number of negative numbers, basically, the max product will be the product of all elements;
2. If we have odd number of negative numbers, we need to consider whether we 
drop the first negative number or the last.
3.With 0, we only need to update the result to be 1 after comparison
Then the general idea is to product from both end and handle 0 separately!
*/
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int forwardProduct = 1, backwardProduct = 1;
        int res = INT_MIN;
        int len = nums.size();
        for(int i = 0; i < len; ++i){
            forwardProduct *= nums[i];
            backwardProduct *= nums[len- 1 - i];
            res = max(res, max(forwardProduct, backwardProduct));
            forwardProduct = forwardProduct ? forwardProduct : 1;
            backwardProduct = backwardProduct ? backwardProduct : 1;
        }
        return res;
    }
};

 /*
    416. Partition Equal Subset Sum
    https://leetcode.com/problems/partition-equal-subset-sum/
    Given an integer array nums, return true if you can partition the array into two subsets 
    such that the sum of the elements in both subsets is equal or false otherwise.


    Example 1:
    Input: nums = [1,5,11,5]
    Output: true
    Explanation: The array can be partitioned as [1, 5, 5] and [11].

    Example 2:
    Input: nums = [1,2,3,5]
    Output: false
    Explanation: The array cannot be partitioned into equal sum subsets.
    

    Constraints:
    1 <= nums.length <= 200
    1 <= nums[i] <= 100
 */
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for (int e : nums) sum += e;
        if (sum % 2 == 1) return false;
        int target = sum / 2;
        vector<vector<bool>> dp(nums.size() + 1, vector<bool>(target + 1, false));
        dp[0][0] = true;

        for (int i = 1; i <= nums.size(); ++i) {
            for (int j = target; j >= 0; --j) {
                if (j < nums[i-1]) dp[i][j] = dp[i-1][j];
                // Note we need to consider two situation:
                // 1. with nums[i-1], which means from previous i-1 number, we can get to j-nums[i-1].
                // 2. without nums[i-1], we have already got to j;
                else dp[i][j] = dp[i-1][j-nums[i-1]] || dp[i-1][j];
            }
        }
        return dp[nums.size()][target];
    }
};

// Optimized version
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for (int e : nums) sum += e;
        if (sum % 2 == 1) return false;
        int target = sum / 2;
        vector<bool> dp(target + 1, false);
        dp[0] = true;

        for (int i = 1; i <= nums.size(); ++i) {
            // We needs info from dp[i-1][j] & dp[i-1][j-nums[i-1]], thus we need to iterate from
            // the back.
            for (int j = target; j >= nums[i-1]; --j) {
                dp[j] = dp[j-nums[i-1]] || dp[j];
            }
        }
        return dp[target];
    }
};

 /*
    1049. Last Stone Weight II
    https://leetcode.com/problems/last-stone-weight-ii/
    You are given an array of integers stones where stones[i] is the weight of the ith stone.

    We are playing a game with the stones. On each turn, we choose any two stones and smash them 
    together. Suppose the stones have weights x and y with x <= y. The result of this smash is:

    If x == y, both stones are destroyed, and
    If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
    At the end of the game, there is at most one stone left.

    Return the smallest possible weight of the left stone. If there are no stones left, return 0.

    

    Example 1:
    Input: stones = [2,7,4,1,8,1]
    Output: 1
    Explanation:
    We can combine 2 and 4 to get 2, so the array converts to [2,7,1,8,1] then,
    we can combine 7 and 8 to get 1, so the array converts to [2,1,1,1] then,
    we can combine 2 and 1 to get 1, so the array converts to [1,1,1] then,
    we can combine 1 and 1 to get 0, so the array converts to [1], then that's the optimal value.

    Example 2:
    Input: stones = [31,26,33,21,40]
    Output: 5
    

    Constraints:
    1 <= stones.length <= 30
    1 <= stones[i] <= 100
 */
class Solution {
public:
    int lastStoneWeightII(vector<int>& stones) {
        int sums = 0;
        for (int e : stones) sums += e;
        int target = sums / 2;
        vector<vector<int>> dp(stones.size() + 1, vector<int>(target + 1, 0));

        for (int i = 1; i <= stones.size(); ++i) {
            for (int j = 1; j <= target; ++j) {
                if (j < stones[i-1]) dp[i][j] = dp[i-1][j];
                else dp[i][j] = max(dp[i-1][j], dp[i-1][j-stones[i-1]] + stones[i-1]);
            }
        }
        return abs(sums - dp[stones.size()][target] - dp[stones.size()][target]);
    }
};

// Optimized version!
class Solution {
public:
    int lastStoneWeightII(vector<int>& stones) {
        int sums = 0;
        for (int e : stones) sums += e;
        int target = sums / 2;
        vector<int> dp(target + 1, 0);

        for (int i = 1; i <= stones.size(); ++i) {
            for (int j = target; j >= stones[i-1]; --j) {
                dp[j] = max(dp[j], dp[j-stones[i-1]] + stones[i-1]);
            }
        }
        return abs(sums - dp[target] - dp[target]);
    }
};

 /*
    494. Target Sum
    https://leetcode.com/problems/target-sum/
    You are given an integer array nums and an integer target.

    You want to build an expression out of nums by adding one of the symbols '+' and '-' before 
    each integer in nums and then concatenate all the integers.

    For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate 
    them to build the expression "+2-1".
    Return the number of different expressions that you can build, which evaluates to target.

    

    Example 1:
    Input: nums = [1,1,1,1,1], target = 3
    Output: 5
    Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
    -1 + 1 + 1 + 1 + 1 = 3
    +1 - 1 + 1 + 1 + 1 = 3
    +1 + 1 - 1 + 1 + 1 = 3
    +1 + 1 + 1 - 1 + 1 = 3
    +1 + 1 + 1 + 1 - 1 = 3

    Example 2:
    Input: nums = [1], target = 1
    Output: 1
    

    Constraints:
    1 <= nums.length <= 20
    0 <= nums[i] <= 1000
    0 <= sum(nums[i]) <= 1000
    -1000 <= target <= 1000
 */
// We can also do backtracking, which will cause TLE.
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        // We can put all positive expressions together, so we will try to calculate 
        // Left subset - Right subset == target. We know that Left + Right = Sum, then
        // we know we are trying to get Left = (target + sum) / 2. What we want to do
        // is to figure how many possible ways we can get Left by picking the elements from
        // nums.
        int sum = 0;
        for (int e : nums) sum += e;
        // No way to get target. Note that target can be negative!
        if (abs(target) > sum || (sum + target) % 2 != 0) return 0; 
        int left = (sum + target) / 2;
        vector<vector<int>> dp(nums.size() + 1, vector<int>(left + 1, 0));
        dp[0][0] = 1;

        for (int i = 1; i <= nums.size(); ++i) {
            // Note we need to start with j = 0 to handle the situation we have multiple 0 in the array
            // [0,0,0,0,0,0,0,0,1], target  = 1. In this case, dp[i][0] = dp[i-1][0] + dp[i-1][0-0].
            // If we start with j = 1, we will not update j == 0 correctly!
            for (int j = 0; j <= left; ++j) {
                if (j < nums[i-1]) dp[i][j] = dp[i-1][j];
                else dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]];
            }
        }

        return dp[nums.size()][left];
    }
};

// Optimized version
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        // We can put all positive expressions together, so we will try to calculate 
        // Left subset - Right subset == target. We know that Left + Right = Sum, then
        // we know we are trying to get Left = (target + sum) / 2. What we want to do
        // is to figure how many possible ways we can get Left by picking the elements from
        // nums.
        int sum = 0;
        for (int e : nums) sum += e;
        // No way to get target. Note that target can be negative!
        if (abs(target) > sum || (sum + target) % 2 != 0) return 0; 
        int left = (sum + target) / 2;
        vector<int> dp(left + 1, 0);
        dp[0] = 1;

        for (int i = 1; i <= nums.size(); ++i) {
            for (int j = left; j >= nums[i-1]; --j) {
                dp[j] = dp[j] + dp[j-nums[i-1]];
            }
        }

        return dp[left];
    }
};

 /*
    474. Ones and Zeroes
    https://leetcode.com/problems/ones-and-zeroes/
    You are given an array of binary strings strs and two integers m and n.

    Return the size of the largest subset of strs such that there are at most m 0's and n 1's in 
    the subset.

    A set x is a subset of a set y if all elements of x are also elements of y.

    

    Example 1:
    Input: strs = ["10","0001","111001","1","0"], m = 5, n = 3
    Output: 4
    Explanation: The largest subset with at most 5 0's and 3 1's is {"10", "0001", "1", "0"}, so the answer is 4.
    Other valid but smaller subsets include {"0001", "1"} and {"10", "1", "0"}.
    {"111001"} is an invalid subset because it contains 4 1's, greater than the maximum of 3.

    Example 2:
    Input: strs = ["10","0","1"], m = 1, n = 1
    Output: 2
    Explanation: The largest subset is {"0", "1"}, so the answer is 2.
    

    Constraints:
    1 <= strs.length <= 600
    1 <= strs[i].length <= 100
    strs[i] consists only of digits '0' and '1'.
    1 <= m, n <= 100
 */
class Solution {
private:
    pair<int, int> getMN(string& s) {
        pair<int, int> res(0, 0);
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] == '0') res.first ++;
            else if (s[i] == '1') res.second ++;
        }
        return res;
    }
public:
    int findMaxForm(vector<string>& strs, int m, int n) {
       vector<vector<int>> dp(m+1, vector<int>(n+1, 0));

       for (int i = 0; i < strs.size(); ++i) {
           // Need to start from the back of the array, because we need the information of dp[i-1][j][k].
           for(int j = m; j >= 0; --j) {
               for (int k = n; k >= 0; --k) {
                   pair<int, int> temp = getMN(strs[i]);
                   if (j >= temp.first && k >= temp.second) {
                       dp[j][k] = max(dp[j-temp.first][k-temp.second] + 1, dp[j][k]);
                   }
               }
           }
       }
        return dp[m][n];
    }
};

 /*
    518. Coin Change II
    https://leetcode.com/problems/coin-change-ii/
    You are given an integer array coins representing coins of different denominations and an 
    integer amount representing a total amount of money.

    Return the number of combinations that make up that amount. If that amount of money cannot 
    be made up by any combination of the coins, return 0.

    You may assume that you have an infinite number of each kind of coin.

    The answer is guaranteed to fit into a signed 32-bit integer.

    

    Example 1:
    Input: amount = 5, coins = [1,2,5]
    Output: 4
    Explanation: there are four ways to make up the amount:
    5=5
    5=2+2+1
    5=2+1+1+1
    5=1+1+1+1+1

    Example 2:
    Input: amount = 3, coins = [2]
    Output: 0
    Explanation: the amount of 3 cannot be made up just with coins of 2.

    Example 3:
    Input: amount = 10, coins = [10]
    Output: 1
    

    Constraints:
    1 <= coins.length <= 300
    1 <= coins[i] <= 5000
    All the values of coins are unique.
    0 <= amount <= 5000
 */
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<vector<int>> dp(coins.size() + 1, vector<int>(amount + 1, 0));
        // If amount is 0, then we always have one way to make up it (pick nothing).
        for (int i = 0; i < coins.size(); ++i) dp[i][0] = 1;

        // The inner / outer loop order does not matter. Given we can get the correct dp[i-1][j].
        // Please note, if we are using 1d array, the same dp function will be wrong. It will
        // calculate the permutation instead of combination!!!
        // for(int j = 0; j <= amount; ++j) {
        //     for(int i = 1; i <= coins.size(); ++i) {
        //         if (j < coins[i-1]) dp[i][j] = dp[i-1][j];
        //         else dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]];
        //     }
        // }

        for(int i = 1; i <= coins.size(); ++i) {
            for(int j = 0; j <= amount; ++j) {
                if (j < coins[i-1]) dp[i][j] = dp[i-1][j];
                // Please note we need to value from dp[i][j-coins[i-1]], not dp[i-1][j-coins[i-1]];
                // It means that we have included coins[i-1] in the result multiple times.
                // Which also means we can't reverse the iteration of order of amount when using 1d
                // dp table.
                else dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]];
            }
        }
        return dp[coins.size()][amount];
    }
};

// Slightly optimized version
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount + 1, 0);
        // Always have one way to make up the amount to be 0.
        dp[0] = 1;

        // We cannot swap the inner / outer loop when using 1d dp table. If we swap it, we are 
        // computing the permutation instead of combination!!!
        for(int i = 1; i <= coins.size(); ++i) {
            for(int j = 0; j <= amount; ++j) {
                // Please note we need to value from dp[i][j-coins[i-1]], not dp[i-1][j-coins[i-1]];
                // It means that we have included coins[i-1] in the result multiple times.
                // Which also means we can't reverse the iteration of order of amount when using 1d
                // dp table.
                if (j >= coins[i-1]) dp[j]  = dp[j] + dp[j-coins[i-1]];
            }
        }
        return dp[amount];
    }
};

 /*
    322. Coin Change
    https://leetcode.com/problems/coin-change/
    You are given an integer array coins representing coins of different denominations and an 
    integer amount representing a total amount of money.

    Return the fewest number of coins that you need to make up that amount. If that amount of 
    money cannot be made up by any combination of the coins, return -1.

    You may assume that you have an infinite number of each kind of coin.

    

    Example 1:
    Input: coins = [1,2,5], amount = 11
    Output: 3
    Explanation: 11 = 5 + 5 + 1

    Example 2:
    Input: coins = [2], amount = 3
    Output: -1

    Example 3:
    Input: coins = [1], amount = 0
    Output: 0
    

    Constraints:

    1 <= coins.length <= 12
    1 <= coins[i] <= 231 - 1
    0 <= amount <= 10^4
 */
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
       vector<vector<int>> dp(coins.size() + 1, vector<int>(amount + 1, amount + 1));

       for (int i = 0; i < coins.size(); ++i) dp[i][0] = 0;

        // Swap the inner / outer loops do not affect the results.
        for(int i = 1; i <= coins.size(); ++i) {
            for(int j = 0; j <= amount; ++j) {
                if (j < coins[i-1]) dp[i][j] = dp[i-1][j];
                // Note we need to include dp[i][j-coins[i-1]], not dp[i-1][j-coins[i-1]].
                // Which represents we can pick up multiple coins[i-1].
                else dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i-1]] + 1);
            }
        }
        return dp[coins.size()][amount] == amount + 1 ? -1 : dp[coins.size()][amount];
    }
};

// Optimized solution
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
       vector<int> dp(amount + 1, amount + 1);

       dp[0] = 0;

        // Given it's 
        for(int i = 1; i <= coins.size(); ++i) {
            for(int j = 0; j <= amount; ++j) {
                if (j >= coins[i-1]) 
                    dp[j] = min(dp[j], dp[j-coins[i-1]] + 1);
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
};

 /*
    377. Combination Sum IV
    https://leetcode.com/problems/combination-sum-iv/
    Given an array of distinct integers nums and a target integer target, return the number of 
    possible combinations that add up to target.

    The test cases are generated so that the answer can fit in a 32-bit integer.

    

    Example 1:
    Input: nums = [1,2,3], target = 4
    Output: 7
    Explanation:
    The possible combination ways are:
    (1, 1, 1, 1)
    (1, 1, 2)
    (1, 2, 1)
    (1, 3)
    (2, 1, 1)
    (2, 2)
    (3, 1)
    Note that different sequences are counted as different combinations.

    Example 2:
    Input: nums = [9], target = 3
    Output: 0
    

    Constraints:
    1 <= nums.length <= 200
    1 <= nums[i] <= 1000
    All the elements of nums are unique.
    1 <= target <= 1000
    

    Follow up: What if negative numbers are allowed in the given array? 
    How does it change the problem? 
    What limitation we need to add to the question to allow negative numbers?
 */
// 2D dp solution! Please note the leetcode can't pass the solution because we got integer overflow
// during the calculation!
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        int n = nums.size();
        // Note we need to use unsigned int to prevent integer overflow.
        vector<vector<unsigned int>> dp(n+1, vector<unsigned int>(target+1, 0));
        for (int i = 0; i <= n; ++i) dp[i][0] = 1;

        for (int j = 0; j <= target; ++j) {
            for(int i = 1; i <= n; ++i) {
                if (j < nums[i-1]) dp[i][j] = dp[i-1][j];
                // dp[n][j-nums[i-1]] is the tricky part. It means we have used all the coins to 
                // get to j-nums[i-1] target. It will include all the possible permutations we 
                // get to previous n, then we simply add one more to the result.
                else dp[i][j] = dp[i-1][j] + dp[n][j-nums[i-1]];

            }
        }
        return dp[n][target];
    }
};

// Optimized 1d array
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        int n = nums.size();
        vector<unsigned int> dp(target+1, 0);
        dp[0] = 1;

        // Please note we have to start with target as the outer loop.
        for (int j = 0; j <= target; ++j) {
            for(int i = 1; i <= n; ++i) {
                if (j >= nums[i-1])
                    dp[j] += dp[j-nums[i-1]];
            }
        }
        return dp[target];
    }
};

// Top-down memorization solution
class Solution {
private:

    int dfs(vector<int>& nums, int target, vector<unsigned int>& memo){
        //Any invalid combination sum we return 0 to indicate it's invalid
        //Else we find a new combination sum, we return 1
        if(nums.empty() || target < 0) return 0;
        if(target == 0) return 1;
        if(memo[target]!= -1) return memo[target];
        unsigned int count = 0;
        for(int i = 0; i < nums.size(); i++){
            //We are actually checking each possible combination, first we reduce nums[0], then nums[1]... and so on
            count += dfs(nums, target - nums[i], memo);
        }
        return memo[target] = count;
    }
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<unsigned int> memo(target+1, -1);
        return dfs(nums, target, memo);
    }
};

 /*
    279. Perfect Squares
    https://leetcode.com/problems/perfect-squares/
    Given an integer n, return the least number of perfect square numbers that sum to n.

    A perfect square is an integer that is the square of an integer; in other words, it is the 
    product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 3 
    and 11 are not.

    

    Example 1:
    Input: n = 12
    Output: 3
    Explanation: 12 = 4 + 4 + 4.

    Example 2:
    Input: n = 13
    Output: 2
    Explanation: 13 = 4 + 9.
    

    Constraints:
    1 <= n <= 10^4
 */
class Solution {
public:
    int numSquares(int n) {
        int k = sqrt(n);
        vector<vector<int>> dp(k+1, vector<int>(n+1, n+1));
        // Need to initialize the defaults here.
        for(int i = 0; i <= k; ++i) dp[i][0] = 0;

        for(int i = 1; i <= k; ++i) {
            for (int j = 1; j <= n; ++j) {
                dp[i][j] = min(dp[i][j],  dp[i-1][j]);
                if (j >= i * i) {
                    dp[i][j] = min(dp[i-1][j], dp[i][j-i*i] + 1);
                }
            }
        } 
        return dp[k][n];
    }
};

// Optimized 1d array
class Solution {
public:
    int numSquares(int n) {
        int k = sqrt(n);
        vector<int> dp(n+1, n+1);
        dp[0] = 0;

        for(int i = 1; i <= k; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (j >= i * i) {
                    dp[j] = min(dp[j], dp[j-i*i] + 1);
                }
            }
        } 
        return dp[n];
    }
};

 /*
    139. Word Break
    https://leetcode.com/problems/word-break/
    Given a string s and a dictionary of strings wordDict, return true if s can be segmented into 
    a space-separated sequence of one or more dictionary words.

    Note that the same word in the dictionary may be reused multiple times in the segmentation.

    

    Example 1:
    Input: s = "leetcode", wordDict = ["leet","code"]
    Output: true
    Explanation: Return true because "leetcode" can be segmented as "leet code".

    Example 2:
    Input: s = "applepenapple", wordDict = ["apple","pen"]
    Output: true
    Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
    Note that you are allowed to reuse a dictionary word.

    Example 3:
    Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
    Output: false
    

    Constraints:
    1 <= s.length <= 300
    1 <= wordDict.length <= 1000
    1 <= wordDict[i].length <= 20
    s and wordDict[i] consist of only lowercase English letters.
    All the strings of wordDict are unique.
 */
// Complete knapsack problem: we need to find the **permutation** from the wordDict to formulate s.
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        vector<bool> dp(n+1, false);
        dp[0] = true;

        // Complete knapsack problem:
        // We are checking the permutations from the wordDict here, so we need to iterate
        // the knapsack value first, then the item (each word).
        for (int i = 1; i <= s.size(); ++i) {
            for(int j = 0; j < wordDict.size(); ++j) {  
                string word = wordDict[j];
                // Check whether we can formulate the sequence.
                if (i >= word.size() && dp[i - word.size()] && word == s.substr(i-word.size(), word.size())) 
                    dp[i] = true;
            }
        }
        return dp[s.size()];
    }
};

// 2D dp array: trickier to implement.
// Important for us to fully unstand the problem.
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = wordDict.size();
        vector<vector<bool>> dp(n+1, vector<bool>(s.size() + 1, false));
        // Need to initialize the defaults.
        for (int i = 0; i <= n; ++i) dp[i][0] = true;

        // Complete backtrack
        for (int j = 1; j <= s.size(); ++j) {
            for(int i = 1; i <= n; ++i) {  
                string word = wordDict[i-1];
                // If j < word.size(), we will not pick up wordDict[i]. Please note that we are
                // using 2d array, we need to manually set it. If we are using 1d dp table, we 
                // can derive automatically from the previous update.
                dp[i][j] = dp[i-1][j];
                // We are checking dp[n][j-word.size()], which represents if we include all words,
                // whether we can formulate string[0: j-word.size()]
                if (j >= word.size() && dp[n][j-word.size()] && s.substr(j - word.size(), word.size()) == word)
                    dp[i][j] = true;
            }
        }
        return dp[n][s.size()];
    }
};

// Another variation of the impl. I like the first one more.
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        vector<bool> dp(n+1, false);
        dp[0] = true;
        unordered_set<string> uSet(wordDict.begin(), wordDict.end());

        // Complete knapsack problem:
        // We are checking the permutations from the wordDict here, so we need to iterate
        // the knapsack value first, then the item (each word).
        for (int i = 1; i <= s.size(); ++i) {
            // We check each possible combinations and see whether we can find a good fit.
            for(int j = 0; j < i; ++j) {  
                string word = s.substr(j, i - j);
                // Note we need to check dp[j] here given the word start with index j,
                // dp[j] means the sequence before word[j, i).
                if (uSet.count(word) > 0 && dp[j]) 
                    dp[i] = true;
            }
        }
        return dp[s.size()];
    }
};

// Recursive + memo
class Solution {
private:
    bool helper(string& s, unordered_set<string>& uSet, vector<int>& memo, int start) {
        if (start >= s.size()) return true;

        if (memo[start] != -1) return memo[start] == 0 ? false : true;

        string word = "";
        for (int i = start; i < s.size(); ++i) {
            word.push_back(s[i]);
            if (uSet.count(word) > 0 && helper(s, uSet, memo, i + 1)) return memo[start] = 1;
        }
        memo[start] = 0;
        return false;
    }
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        vector<int> memo(n, -1);
        unordered_set<string> uSet(wordDict.begin(), wordDict.end());
        return helper(s, uSet, memo, 0);
    }
};

// BFS solution: omit.

 /*
    198. House Robber
    https://leetcode.com/problems/house-robber/
    You are a professional robber planning to rob houses along a street. Each house has a certain 
    amount of money stashed, the only constraint stopping you from robbing each of them is that 
    adjacent houses have security systems connected and it will automatically contact the police 
    if two adjacent houses were broken into on the same night.

    Given an integer array nums representing the amount of money of each house, return the maximum 
    amount of money you can rob tonight without alerting the police.

    
    Example 1:
    Input: nums = [1,2,3,1]
    Output: 4
    Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
    Total amount you can rob = 1 + 3 = 4.

    Example 2:
    Input: nums = [2,7,9,3,1]
    Output: 12
    Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
    Total amount you can rob = 2 + 9 + 1 = 12.
    

    Constraints:
    1 <= nums.length <= 100
    0 <= nums[i] <= 400
 */
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        // Adds one more value to unify the for loop behavior.
        vector<int> dp(n + 1, 0);
        dp[0] = 0;
        dp[1] = nums[0];

        for(int i = 2; i <= n; ++i) {
            dp[i] = dp[i-1];
            if (i > 1) dp[i] = max(dp[i], dp[i-2] + nums[i-1]);
        }
        return dp[n];
    }
};

// Optimized version: save space
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        // represents dp[i-2]
        int pre = 0;
        // represents dp[i-1]
        int cur = nums[0];

        for(int i = 2; i <= n; ++i) {
            int temp = cur;
            if (i > 1) temp = max(temp, pre + nums[i-1]);
            pre = cur;
            cur = temp;
            // dp[i] = dp[i-1];
            // if (i > 1) dp[i] = max(dp[i], dp[i-2] + nums[i-1]);
        }
        return cur;
    }
};

 /*
    213. House Robber II
    https://leetcode.com/problems/house-robber-ii/
    You are a professional robber planning to rob houses along a street. Each house has a certain 
    amount of money stashed. All houses at this place are arranged in a circle. That means the 
    first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system 
    connected, and it will automatically contact the police if two adjacent houses were broken 
    into on the same night.

    Given an integer array nums representing the amount of money of each house, return the maximum 
    amount of money you can rob tonight without alerting the police.

    

    Example 1:
    Input: nums = [2,3,2]
    Output: 3
    Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
    
    Example 2:
    Input: nums = [1,2,3,1]
    Output: 4
    Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
    Total amount you can rob = 1 + 3 = 4.
    
    Example 3:
    Input: nums = [1,2,3]
    Output: 3
    

    Constraints:
    1 <= nums.length <= 100
    0 <= nums[i] <= 1000
 */
/* The general idea is similar to House Robber I. 
We break the circle and consider the two situations separately.
1. We rob the first house, and ignore the last house
2. We start from the second house, and iterate to the last one
We can merge the code together, however, it will be hard to read*/
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        // Need to check the corner case.
        if (n == 1) return nums[0]; 

        vector<int> dp(n+1, 0);
        vector<int> dp1(n+1, 0);
        // Rob the first house
        dp[0] = 0;
        dp[1] = nums[0];

        // Rob the second house
        dp1[0] = 0;
        dp1[1] = 0;

        // Because we rob the first house, then we can't rob house n-1. Thus < n is used.
        for (int i = 2; i < n; ++i) {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i-1]);
        }
        // Because we rob the second house, then we can rob house n-1. Thus <= n is used.
        for (int i = 2; i <=n; ++i) {
            dp1[i] = max(dp1[i-1], dp1[i-2] + nums[i-1]);
        }

        return max(dp[n-1], dp1[n]);
    }
};

 /*
    337. House Robber III
    https://leetcode.com/problems/house-robber-iii/
    The thief has found himself a new place for his thievery again. There is only one entrance 
    to this area, called root.

    Besides the root, each house has one and only one parent house. After a tour, the smart thief 
    realized that all houses in this place form a binary tree. It will automatically contact the 
    police if two directly-linked houses were broken into on the same night.

    Given the root of the binary tree, return the maximum amount of money the thief can rob without 
    alerting the police.

    

    Example 1:
    Input: root = [3,2,3,null,3,null,1]
    Output: 7
    Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.

    Example 2:
    Input: root = [3,4,5,1,3,null,1]
    Output: 9
    Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
    

    Constraints:
    The number of nodes in the tree is in the range [1, 104].
    0 <= Node.val <= 10^4
 */
// Bruet force approach: we have a lot of duplicate calculations! Will get TLE in leetcode.
class Solution {
private:
    // helper returns the maximum possible profit we have when starting with Node root 
    int helper(TreeNode* root) {
        if(!root) return 0;

        // If we rob the parent
        int robParent = root->val;
        int left = 0, right = 0;
        // Skip left & right children
        if (root->left) left = helper(root->left->left) + helper(root->left->right);
        if (root->right) right = helper(root->right->left) + helper(root->right->right);
        // We need to add all the possible combination together.
        robParent += left + right;

        // If we skip the parent. Then we need to add both left & right.
        int nRobParent = helper(root->left) + helper(root->right);

        return max(robParent, nRobParent);
    }
public:
    int rob(TreeNode* root) {
        if (!root) return 0;

        return helper(root);
    }
};

// Using a pair to record the maximum possible outcome with / without robbing the node i.
class Solution {
private:
    // The first element in the pair represents the maximum possible outcome 
    // if we do not rob the child;
    // The second element in the pair represents the maximum possible outcome if we do 
    // rob the child.
    pair<int, int> helper(TreeNode* root) {
        if (!root) return {0, 0};

        auto left = helper(root->left);
        auto right = helper(root->right);

        // Rob the current root, we need to include the maximum without robbing left / right
        // children.
        int robCurrent = root->val + left.first + right.first;
        // Do not rob the current.
        int nRobCurrent = max(left.first, left.second) + max(right.first, right.second);
        
        return {nRobCurrent, robCurrent};
    }
public:
    int rob(TreeNode* root) {
        auto res = helper(root);
        return max(res.first, res.second);
    }
};

 /*
    121. Best Time to Buy and Sell Stock
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    You are given an array prices where prices[i] is the price of a given stock on the ith day.

    You want to maximize your profit by choosing a single day to buy one stock and choosing a 
    different day in the future to sell that stock.

    Return the maximum profit you can achieve from this transaction. If you cannot achieve any 
    profit, return 0.

    

    Example 1:
    Input: prices = [7,1,5,3,6,4]
    Output: 5
    Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
    Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
    
    Example 2:
    Input: prices = [7,6,4,3,1]
    Output: 0
    Explanation: In this case, no transactions are done and the max profit = 0.
    

    Constraints:
    1 <= prices.length <= 10^5
    0 <= prices[i] <= 10^4
 */
// DP
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        // dp[i][0] means the maximum possible profit on day i while holding one stock
        // dp[i][1] means the maximum possible profit on day i while not holding the stock
        vector<vector<int>> dp(n, vector<int>(2, INT_MIN));

        // Critical: needs to initialize the defaults correctly!
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        
        for(int i = 1; i < n; ++i) {
            // Note for dp[i][0], we need to use -prices[i-1] to represent that we buy
            // at day i, we should not include dp[i-1][1] because we can only buy & sell
            // once.
            dp[i][0] = max(-prices[i], dp[i-1][0]);
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i]);
        }

        return dp[n-1][1];
    }
};

// Record the premin.
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int preMin = prices[0];
        int res = 0;

        for(int i = 1; i < prices.size(); ++i) {
            preMin = min(preMin, prices[i]);
            res = max(res, prices[i] - preMin);
        }
        return res;
    }
};

 /*
    122. Best Time to Buy and Sell Stock II
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    You are given an integer array prices where prices[i] is the price of a given stock on the 
    ith day.

    On each day, you may decide to buy and/or sell the stock. You can only hold at most one 
    share of the stock at any time. However, you can buy it then immediately sell it on the same day.

    Find and return the maximum profit you can achieve.


    Example 1:
    Input: prices = [7,1,5,3,6,4]
    Output: 7
    Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
    Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
    Total profit is 4 + 3 = 7.

    Example 2:
    Input: prices = [1,2,3,4,5]
    Output: 4
    Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
    Total profit is 4.

    Example 3:
    Input: prices = [7,6,4,3,1]
    Output: 0
    Explanation: There is no way to make a positive profit, so we never buy the stock to achieve the maximum profit of 0.
    

    Constraints:
    1 <= prices.length <= 3 * 10^4
    0 <= prices[i] <= 10^4
 */
// DP
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        // dp[i][0] means the maximum possible profit on day i while holding one stock
        // dp[i][1] means the maximum possible profit on day i while not holding the stock
        vector<vector<int>> dp(n, vector<int>(2, INT_MIN));

        // Critical: needs to initialize the defaults correctly!
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        
        for(int i = 1; i < n; ++i) {
            // Compared with problem 121, we need to include the previous day (not holding)
            // stock here. dp[i-1][1] - prices[i]. Given we can buy and sell multiple times.
            dp[i][0] = max(dp[i-1][1]-prices[i], dp[i-1][0]);
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i]);
        }

        return dp[n-1][1];
    }
};

// Greedy
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.empty()) return 0;
        int len_p = prices.size();
        int maxP = 0;
        for(int i = 1; i < len_p; i++){
            if(prices[i]>prices[i-1])
                maxP += prices[i]-prices[i-1];
        }
        return maxP;
    }
};

 /*
    123. Best Time to Buy and Sell Stock III
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
    You are given an array prices where prices[i] is the price of a given stock on the ith day.

    Find the maximum profit you can achieve. You may complete at most two transactions.

    Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the 
    stock before you buy again).

    

    Example 1:
    Input: prices = [3,3,5,0,0,3,1,4]
    Output: 6
    Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
    Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.

    Example 2:
    Input: prices = [1,2,3,4,5]
    Output: 4
    Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
    Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging 
    multiple transactions at the same time. You must sell before buying again.

    Example 3:
    Input: prices = [7,6,4,3,1]
    Output: 0
    Explanation: In this case, no transaction is done, i.e. max profit = 0.
    

    Constraints:
    1 <= prices.length <= 10^5
    0 <= prices[i] <= 10^5
 */
// DP
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        // dp[i][0] represents the max profit that the first time we hold a stock;
        // dp[i][1] represents the max profit that the first time we do not hold 
        // a stock (we may never buy any stock or we buy and sell only once);
        // dp[i][2] represents the max profit the second time we hold a stcok;
        // dp[i][3] represents the max profit that the second time we do not have a stock
        vector<vector<int>> dp(n, vector<int>(4, INT_MIN));
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        dp[0][2] = -prices[0];
        dp[0][3] = 0;

        for (int i = 1; i < n; ++i) {
            // We either buy at day i or have the stock state from previous day;
            dp[i][0] = max(-prices[i], dp[i-1][0]);
            // We either sell at day i or have sold the stock before day i.
            dp[i][1] = max(dp[i-1][0] + prices[i], dp[i-1][1]);
            // We hold the stock at day i for the second time. Note we have to derive this
            // from the dp[i][1], where we sold the stock once.
            dp[i][2] = max(dp[i][1] - prices[i], dp[i-1][2]);
            // We do not hold the stock at day i for the second time.
            dp[i][3] = max(dp[i-1][2] + prices[i], dp[i-1][3]);
        }

        // Find the max of the two trasactions.
        return max(dp[n-1][1], dp[n-1][3]);
    }
};

 /*
    188. Best Time to Buy and Sell Stock IV
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
    You are given an integer array prices where prices[i] is the price of a given stock on the 
    ith day, and an integer k.

    Find the maximum profit you can achieve. You may complete at most k transactions: i.e. you 
    may buy at most k times and sell at most k times.

    Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the 
    stock before you buy again).

    

    Example 1:
    Input: k = 2, prices = [2,4,1]
    Output: 2
    Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.

    Example 2:
    Input: k = 2, prices = [3,2,6,5,0,3]
    Output: 7
    Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4. Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
    

    Constraints:
    1 <= k <= 100
    1 <= prices.length <= 1000
    0 <= prices[i] <= 1000
 */
// DP
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        // dp[i][k%2==0] represents the k+1 th trasaction while we hold stock;
        // dp[i][k%2==1] represents the k+1 th trasaction while we do not hold stock.
        vector<vector<int>> dp(n, vector<int>(2*k, INT_MIN));
        for (int i = 0; i < 2*k; ++i) {
            if (i % 2 == 0) {
                dp[0][i] = -prices[0];
            } else {
                dp[0][i] = 0;
            }
        }

        for (int i = 1; i < n; ++i) {
            dp[i][0] = max(-prices[i], dp[i-1][0]);
            dp[i][1] = max(dp[i-1][0] + prices[i], dp[i-1][1]);
            for (int j = 2; j < 2 * k - 1; j += 2) {
                // i + 1 the trasaction that we hold stock.
                // For dp[i][j], we need to derive from dp[i][j-1], basically we only buy
                // k+1 th stock after we have make sure that in the ith day, 
                // we do not hold hay stocks && have sold kth stock.
                dp[i][j] = max(dp[i][j-1] - prices[i], dp[i-1][j]);
                dp[i][j+1] = max(dp[i-1][j] + prices[i], dp[i-1][j+1]);
            }
        }

        // The maximum profit equals to the sell of **at most** k stocks.
        return dp[n-1][2*k-1];
    }
};

 /*
    309. Best Time to Buy and Sell Stock with Cooldown
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
    You are given an array prices where prices[i] is the price of a given stock on the ith day.

    Find the maximum profit you can achieve. You may complete as many transactions as you like 
    (i.e., buy one and sell one share of the stock multiple times) with the following 
    restrictions:

    After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
    Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the 
    stock before you buy again).

    
    Example 1:
    Input: prices = [1,2,3,0,2]
    Output: 3
    Explanation: transactions = [buy, sell, cooldown, buy, sell]

    Example 2:
    Input: prices = [1]
    Output: 0
    

    Constraints:
    1 <= prices.length <= 5000
    0 <= prices[i] <= 1000
 */
// DP
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();

        vector<vector<int>> dp(n, vector<int>(4, INT_MIN));
        // dp[i][0] represents the maximum profit while hold stock at day i.
        // dp[i][1] represents the maximum profit while not hold stock from prevous day i-1.
        // dp[i][2] represents the maximum profit while we sell stock at day i (special
        // situation of dp[i][1]).
        // dp[i][3] represents the maximum profit on cool down day i.
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        dp[0][2] = 0;
        dp[0][3] = 0;

        for (int i = 1; i < n; ++i) {
            // We either derive the same profit from previous hold day, or buy today.
            dp[i][0] = max(max(dp[i-1][0], dp[i-1][1] - prices[i]), dp[i-1][3] - prices[i]);
            // dp[i][1] is valid if we do not hold stock from previous day or we are at cool
            // down state from previous day.
            dp[i][1] = max(dp[i-1][1], dp[i-1][3]);
            // We will sell stock at day i, only one situation
            dp[i][2] = dp[i-1][0] + prices[i];
            // We are at cool down state, so previous day we must have sold the stock.
            dp[i][3] = dp[i-1][2];
        }

        return max(max(dp[n-1][1], dp[n-1][2]), dp[n-1][3]);
    }
};

 /*
    714. Best Time to Buy and Sell Stock with Transaction Fee
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
    You are given an array prices where prices[i] is the price of a given stock on the ith day, 
    and an integer fee representing a transaction fee.

    Find the maximum profit you can achieve. You may complete as many transactions as you like, 
    but you need to pay the transaction fee for each transaction.

    Note:
    You may not engage in multiple transactions simultaneously (i.e., you must sell the stock 
    before you buy again).
    The transaction fee is only charged once for each stock purchase and sale.
    

    Example 1:
    Input: prices = [1,3,2,8,4,9], fee = 2
    Output: 8
    Explanation: The maximum profit can be achieved by:
    - Buying at prices[0] = 1
    - Selling at prices[3] = 8
    - Buying at prices[4] = 4
    - Selling at prices[5] = 9
    The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.

    Example 2:
    Input: prices = [1,3,7,5,10,3], fee = 3
    Output: 6
    

    Constraints:
    1 <= prices.length <= 5 * 10^4
    1 <= prices[i] < 5 * 10^4
    0 <= fee < 5 * 10^4
 */
// DP
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int n = prices.size();
        if (n <= 1) return 0;

        // dp[i][0] means the maximum profit for we hold a stock at day i;
        // dp[i][1] means the maximum profit for we do not hold a stick at day i.
        vector<vector<int>> dp(n, vector<int>(2, INT_MIN));
        dp[0][0] = -prices[0];
        // Note dp[0][1] should be 0 instead of -fee, given we need to make it the 
        // maximum profit for day 0.
        dp[0][1] = 0;

        for(int i = 1; i < n; ++i) {
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i]);
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i] - fee);
        }

        return dp[n-1][1];
    }
};

 /*
    1143. Longest Common Subsequence
    https://leetcode.com/problems/longest-common-subsequence/
    Given two strings text1 and text2, return the length of their longest common subsequence. 
    If there is no common subsequence, return 0.

    A subsequence of a string is a new string generated from the original string with some 
    characters (can be none) deleted without changing the relative order of the remaining 
    characters.

    For example, "ace" is a subsequence of "abcde".
    A common subsequence of two strings is a subsequence that is common to both strings.

    
    Example 1:
    Input: text1 = "abcde", text2 = "ace" 
    Output: 3  
    Explanation: The longest common subsequence is "ace" and its length is 3.

    Example 2:
    Input: text1 = "abc", text2 = "abc"
    Output: 3
    Explanation: The longest common subsequence is "abc" and its length is 3.

    Example 3:
    Input: text1 = "abc", text2 = "def"
    Output: 0
    Explanation: There is no such common subsequence, so the result is 0.
    

    Constraints:
    1 <= text1.length, text2.length <= 1000
    text1 and text2 consist of only lowercase English characters.
 */
// DP + print path!
class Solution {
private:
    void printLCS(string& s1, vector<vector<int>>& path, int i, int j) {
        if (i < 0 || j < 0) return;

        if (path[i][j] == 1) {
            printLCS(s1, path, i-1, j-1);
            cout << s1[i] << " ";
        } else if (path[i][j] == 2) printLCS(s1, path, i, j-1);
        else printLCS(s1, path, i-1, j);
    }
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size(), n = text2.size();
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        // We would like to print the sequence, we have to save the relationship in
        // an array.
        vector<vector<int>> path(m, vector<int>(n, 0));
        
        for(int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (text1[i-1] == text2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                    // Save path
                    path[i-1][j-1] = 1;
                }
                else {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);

                    // Save path
                    if (dp[i-1][j] < dp[i][j-1]) path[i-1][j-1] = 2;
                    else path[i-1][j-1] = 3;
                }
            }
        }
        // Print the LCS.
        printLCS(text1, path, m-1, n-1);
        return dp[m][n];
    }
};

 /*
    718. Maximum Length of Repeated Subarray
    https://leetcode.com/problems/maximum-length-of-repeated-subarray/
    Given two integer arrays nums1 and nums2, return the maximum length of a subarray that appears 
    in both arrays.


    Example 1:
    Input: nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
    Output: 3
    Explanation: The repeated subarray with maximum length is [3,2,1].

    Example 2:
    Input: nums1 = [0,0,0,0,0], nums2 = [0,0,0,0,0]
    Output: 5
    Explanation: The repeated subarray with maximum length is [0,0,0,0,0].
    

    Constraints:
    1 <= nums1.length, nums2.length <= 1000
    0 <= nums1[i], nums2[i] <= 100
 */
// DP
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        int res = 0;
        for(int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (nums1[i-1] == nums2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                    res = max(dp[i][j], res);
                }
            }
        }

        return res;
    }
};

// DP optimized space!
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        vector<int> dp(n+1, 0);
        int res = 0;
        for(int i = 1; i <= m; ++i) {
            for (int j = n; j >= 1; --j) {
                if (nums1[i-1] == nums2[j-1]) {
                    dp[j] = dp[j-1] + 1;
                    res = max(dp[j], res);
                } else dp[j] = 0; // For 1d table, we need to reset value!
            }
        }

        return res;
    }
};

 /*
    300. Longest Increasing Subsequence
    https://leetcode.com/problems/longest-increasing-subsequence/
    Given an integer array nums, return the length of the longest strictly increasing 
    subsequence.


    Example 1:
    Input: nums = [10,9,2,5,3,7,101,18]
    Output: 4
    Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

    Example 2:
    Input: nums = [0,1,0,3,2,3]
    Output: 4

    Example 3:
    Input: nums = [7,7,7,7,7,7,7]
    Output: 1
    

    Constraints:
    1 <= nums.length <= 2500
    -104 <= nums[i] <= 104
    

    Follow up: Can you come up with an algorithm that runs in O(n log(n)) time complexity?
 */
// DP + print path!
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        // dp[i] means the longest increasing subsequence util i.
        vector<int> dp(n+1, 1);
        dp[0] = 0;
        int res = 1;
        for (int i = 2; i <= nums.size(); ++i) {
            for (int j = 1; j < i; ++j) {
                if (nums[i - 1] > nums[j - 1]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            res = max(res, dp[i]);
        }
        return res;
    }
};

// O(nlogn) solution: not easy to figure it out.
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int len = nums.size();
        if(!len) return 0;
        vector<int> dp;
        for(int i = 0; i < len; ++i){
            auto it = lower_bound(dp.begin(), dp.end(), nums[i]);
            if(it == dp.end()) dp.push_back(nums[i]);
            else *it = nums[i];
        }
        return dp.size();
    }
};

 /*
    674. Longest Continuous Increasing Subsequence
    https://leetcode.com/problems/longest-continuous-increasing-subsequence/
    Given an unsorted array of integers nums, return the length of the longest continuous 
    increasing subsequence (i.e. subarray). The subsequence must be strictly increasing.

    A continuous increasing subsequence is defined by two indices l and r (l < r) such that 
    it is [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] and for each l <= i < r, 
    nums[i] < nums[i + 1].

    

    Example 1:
    Input: nums = [1,3,5,4,7]
    Output: 3
    Explanation: The longest continuous increasing subsequence is [1,3,5] with length 3.
    Even though [1,3,5,7] is an increasing subsequence, it is not continuous as elements 5 and 
    7 are separated by element 4.

    Example 2:
    Input: nums = [2,2,2,2,2]
    Output: 1
    Explanation: The longest continuous increasing subsequence is [2] with length 1. 
    Note that it must be strictly increasing.
    

    Constraints:
    1 <= nums.length <= 10^4
    -109 <= nums[i] <= 10^9
 */
// DP
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        int n = nums.size();

        vector<int> dp(n, 1);
        int res = 1;
        for (int i = 1; i < n; ++i) {
            if (nums[i] > nums[i-1]) {
                dp[i] = dp[i-1] + 1;
                res = max(dp[i], res);
            }   
        }
        return res;
    }
};

// Greedy is omit, it's trivial to implement.

 /*
    1035. Uncrossed Lines
    https://leetcode.com/problems/uncrossed-lines/
    You are given two integer arrays nums1 and nums2. We write the integers of nums1 and nums2 
    (in the order they are given) on two separate horizontal lines.

    We may draw connecting lines: a straight line connecting two numbers nums1[i] and nums2[j] 
    such that: nums1[i] == nums2[j], and the line we draw does not intersect any other connecting 
    (non-horizontal) line.
    Note that a connecting line cannot intersect even at the endpoints (i.e., each number can 
    only belong to one connecting line).

    Return the maximum number of connecting lines we can draw in this way.

    

    Example 1:
    Input: nums1 = [1,4,2], nums2 = [1,2,4]
    Output: 2
    Explanation: We can draw 2 uncrossed lines as in the diagram.
    We cannot draw 3 uncrossed lines, because the line from nums1[1] = 4 to nums2[2] = 4 will intersect the line from nums1[2]=2 to nums2[1]=2.
    
    Example 2:
    Input: nums1 = [2,5,1,2,5], nums2 = [10,5,2,1,5,2]
    Output: 3
    
    Example 3:
    Input: nums1 = [1,3,7,1,7,5], nums2 = [1,9,2,5,1]
    Output: 2
    

    Constraints:
    1 <= nums1.length, nums2.length <= 500
    1 <= nums1[i], nums2[j] <= 2000
 */
// DP: equivalent to find the longest common subsequence
class Solution {
public:
    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        // This problem is equivalent to find the longest common subsequence
        // from nums1 & nums2.
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        int res = 0;
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (nums1[i-1] == nums2[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
                else dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                res = max(res, dp[i][j]);
            }
        }
        return res;
    }
};

 /*
    53. Maximum Subarray
    https://leetcode.com/problems/maximum-subarray/
    Given an integer array nums, find the subarray with the largest sum, and return its sum.


    Example 1:
    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: The subarray [4,-1,2,1] has the largest sum 6.

    Example 2:
    Input: nums = [1]
    Output: 1
    Explanation: The subarray [1] has the largest sum 1.

    Example 3:
    Input: nums = [5,4,-1,7,8]
    Output: 23
    Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
    

    Constraints:
    1 <= nums.length <= 10^5
    -10^4 <= nums[i] <= 10^4
 */
// DP:
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN;
        int n = nums.size();
        vector<int> dp(n+1, 0);
        for (int i = 1; i <= n; ++i) {
            // Similar idea as greedy
            dp[i] = max(dp[i-1] + nums[i-1], nums[i-1]);
            res = max(res, dp[i]);
        }

        return res;
    }
};

// Greedy
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN;
        int sum = 0;
        for (int i = 0; i < nums.size(); ++i) {
            sum += nums[i];
            res= max(res, sum);
            if (sum < 0) sum = 0;
        }
        return res;
    }
};

 /*
    392. Is Subsequence
    https://leetcode.com/problems/is-subsequence/
    Given two strings s and t, return true if s is a subsequence of t, or false otherwise.
    A subsequence of a string is a new string that is formed from the original string by 
    deleting some (can be none) of the characters without disturbing the relative positions 
    of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

    
    Example 1:
    Input: s = "abc", t = "ahbgdc"
    Output: true

    Example 2:
    Input: s = "axc", t = "ahbgdc"
    Output: false
    
    Constraints:
    0 <= s.length <= 100
    0 <= t.length <= 104
    s and t consist only of lowercase English letters.
 */
// We could also rely on hash map for this problem. It's too trival to implement it. Omit.
// DP:
class Solution {
public:
    bool isSubsequence(string s, string t) {
        if (s.size() > t.size()) return false;
        // DP[i][j] means the maximum subsequence between s and t.
        vector<vector<int>> dp(s.size() + 1, vector<int>(t.size() + 1, 0));
        for (int i = 1; i <= s.size(); ++i) {
            for (int j = 1; j <= t.size(); ++j) {
                // If we find a match, then we need to incremental the dp[i][j] by one
                if (s[i-1] == t[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
                // If we cannot find the correct match, we need to "delete" one character
                // from t.
                else dp[i][j] = dp[i][j-1];
            }
        }
        return dp[s.size()][t.size()] == s.size();
    }
};

 /*
    115. Distinct Subsequences
    https://leetcode.com/problems/distinct-subsequences/
    Given two strings s and t, return the number of distinct 
    subsequences of s which equals t.
    The test cases are generated so that the answer fits on a 32-bit signed integer.

    
    Example 1:
    Input: s = "rabbbit", t = "rabbit"
    Output: 3

    Example 2:
    Input: s = "babgbag", t = "bag"
    Output: 5
    

    Constraints:
    1 <= s.length, t.length <= 1000
    s and t consist of English letters.
 */
// DP:
class Solution {
public:
    int numDistinct(string s, string t) {
        int n = s.size(), m = t.size();
        if (m > n) return 0;

        // dp[i][j] means from t[0..i] and s[0..j], what is the maximum number of 
        // distinct subsequence.
        vector<vector<unsigned int>> dp(m+1, vector<unsigned int>(n+1, 0));
        for (int i = 0; i <= n; ++i) dp[0][i] = 1;

        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                // When s[j-1] == t[i-1], we can match this two character, we can also
                // ignore this match.
                if (s[j-1] == t[i-1]) dp[i][j] = dp[i-1][j-1] + dp[i][j-1];
                else dp[i][j] = dp[i][j-1];
            }
        }

        return dp[m][n];
    }
};

 /*
    583. Delete Operation for Two Strings
    https://leetcode.com/problems/delete-operation-for-two-strings/
    Given two strings word1 and word2, return the minimum number of steps required to make word1 
    and word2 the same.
    In one step, you can delete exactly one character in either string.

    
    Example 1:
    Input: word1 = "sea", word2 = "eat"
    Output: 2
    Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".

    Example 2:
    Input: word1 = "leetcode", word2 = "etco"
    Output: 4
    

    Constraints:
    1 <= word1.length, word2.length <= 500
    word1 and word2 consist of only lowercase English letters.
 */
// DP:
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        vector<vector<unsigned int>> dp(m+1, vector<unsigned int>(n+1, INT_MAX));
        for (int i = 0; i <= m; ++i) dp[i][0] = i;
        for (int j = 0; j <= n; ++j) dp[0][j] = j;

        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (word1[i-1] == word2[j-1]) dp[i][j] = dp[i-1][j-1];
                else dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1;
            }
        }

        return dp[m][n];
    }
};
