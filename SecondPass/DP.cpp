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
    Given an integer array nums, find a 
    subarray
    that has the largest product, and return the product.

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
