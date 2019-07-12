//70. Climbing Stairs
//https://leetcode.com/problems/climbing-stairs/
/* Basic problem */
class Solution {
public:
    int climbStairs(int n) {
        if(n < 2) return 1;
        vector<int> dp(n+1, 0);
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        
        for(int i = 3; i <= n; i++){
            dp[i] = dp[i-2] + dp[i-1];
        }
        return dp[n];
    }
};

class Solution {
private:
    int helper(int n, vector<int>& memo){
        if(memo[n]!= -1) return memo[n];
        memo[n] = helper(n-1, memo) + helper(n-2, memo);
        return memo[n];
    }
public:
    int climbStairs(int n) {
        if(n <= 1) return 1;
        vector<int> memo(n+1, -1);
        memo[0] = memo[1] = 1;
        memo[2] = 2;
        return helper(n, memo);
    }
};

//62. Unique Paths
//https://leetcode.com/problems/unique-paths/
/* Pretty standard! */
class Solution {
private:
    int helper(int m, int n, vector<vector<int>>& memo){
        if(m < 0 || n < 0) return 0;
        if(m == 0 && n == 0) return 1;
        if(memo[m][n] != -1) return memo[m][n];
        memo[m][n] = helper(m-1, n, memo) + helper(m, n-1, memo);
        return memo[m][n];
    }
public:
    int uniquePaths(int m, int n) {
        if(m <= 0 || n <= 0) return 0;
        if(m == 1 || n == 1) return 1;
        vector<vector<int>> memo(m, vector<int>(n, -1));
        memo[0][0] = 1;
        return helper(m-1, n-1, memo);
    }
};

//Iterative version
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 0));
        for(int i = 0; i < m; i++)
            dp[i][0] = 1;
        for(int j = 0; j < n; j++)
            dp[0][j] = 1;
        
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};

//



