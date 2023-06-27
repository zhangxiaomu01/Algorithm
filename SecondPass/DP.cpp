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

