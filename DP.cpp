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

//63. Unique Paths II
//https://leetcode.com/problems/unique-paths-ii/
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size(), n = (m ? obstacleGrid[0].size() : 0);
        vector<vector<long>> dp(m+1, vector<long>(n+1, 0));
        /*
        //The following code is wrong, we can no longer assume that the first row and colunm to be 1
        for(int i = 0; i <= m; i++){
            dp[i][0] = 1;
        }
        for(int j = 0; j <= n; j++){
            dp[0][j] = 1;
        }
        */
        dp[0][1] = 1;
        
        for(int i = 1; i <= m; i++){
            for(int j = 1; j <= n; j++){
                if(!obstacleGrid[i-1][j-1])
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m][n];
    }
};


//63. Unique Paths II
//https://leetcode.com/problems/unique-paths-ii/
//Iterative solution:
/*
The tricky part is how to set dp[1][1] to be 1. Note we allocate one more row and column to 
handle the situation i-1 and j-1. We need to gurantee when we reach dp[1][1] (destination),
we we have 1 possible solution.
We use long instead of int to prevent integer overflow.
*/
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size(), n = (m ? obstacleGrid[0].size() : 0);
        vector<vector<long>> dp(m+1, vector<long>(n+1, 0));
        if(obstacleGrid[0][0] == 1) return 0;
        dp[1][1] = 1;
        for(int i = 1; i <= m; i++){
            for(int j = 1; j <= n; j++){
                if(i == 1 && j == 1) continue;
                if(!obstacleGrid[i-1][j-1])
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m][n];
    }
};

//Recursive solution
class Solution {
private:
    int helper(vector<vector<int>>& grid, vector<vector<int>>& memo, int mi, int ni){
        if(mi < 0 || ni < 0) return 0;
        else if(mi == 0 && ni == 0) return 1;
        else if(memo[mi][ni] != -1) return memo[mi][ni];
        else{
            if(!grid[mi][ni])
                memo[mi][ni] = helper(grid, memo, mi-1, ni) + helper(grid, memo, mi, ni-1);
            else
                memo[mi][ni] = 0;
        }
        return memo[mi][ni];
    }
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size(), n = m ? obstacleGrid[0].size() : 0;
        if(obstacleGrid[0][0] == 1) return 0;
        vector<vector<int>> memo(m, vector<int>(n, -1));
        return helper(obstacleGrid, memo, m-1, n-1);
    }
};

