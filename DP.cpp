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

//139. Word Break
//https://leetcode.com/problems/word-break/
/* A classic DP problem. Double check later. The solution is easy to 
understand and easy to come up with. Pay attention to BFS solution instead
of DP.
 */
//Recursive solution
class Solution {
private:
    bool helper(const string& s, unordered_set<string>& dict, vector<int>& memo, int index){
        if(index >= s.size()) return true;
        if(memo[index] != -1) return memo[index];
        string tempS = "";
        for(int i = index; i < s.size(); i++){
            tempS.push_back(s[i]);
            if(dict.find(tempS) != dict.end() && helper(s, dict, memo, i+1)){
                return memo[i] = 1;
            }
        }
        return memo[index] = 0;
    }
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        if(wordDict.empty()) return false;
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        vector<int> memo(s.size() + 1, -1);
        return helper(s, dict, memo, 0);
    }
};

//Iteravte version
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        if(wordDict.empty()) return false;
        int len = s.size();
        vector<int> dp(len+1, 0);
        dp[len] = 1;
        
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        for(int i = len-1; i >=0 ; --i){
            string tempS = "";
            for(int j = i; j < len; ++j){
                //We should not use s.substr(i, j - i + 1) here, more expensive 
                tempS.push_back(s[j]);
                if(dict.find(tempS) != dict.end())
                    dp[i] = dp[j+1];
                if(dp[i])
                    break;
            }
        }
        return dp[0];
    }
};

//BFS version, very interesting!! Take a look later
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        if(wordDict.empty()) return false;
        //We use this Q to store the substring index that we are going to explore
        //Note we only explore the s.substr(0, index+1) which appears in wordDict,
        //and ignore other substrings
        queue<int> Q({0});
        unordered_set<int> visited; //store the visited nodes
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        //A general BFS approach
        while(!Q.empty()){
            int index = Q.front();
            Q.pop();
            if(visited.find(index) != visited.end()) continue;
            visited.insert(index);
            string tempS = "";
            for(int i = index; i < s.size(); i++){
                tempS.push_back(s[i]);
                if(dict.find(tempS) != dict.end()){
                    if(i + 1 == s.size()) return true;
                    Q.push(i+1);
                }
            }
        }
        return false;
    }
};

//279. Perfect Squares
//https://leetcode.com/problems/perfect-squares/
//DP solution: it's slow!
class Solution {
public:
    int numSquares(int n) {
        if(n <= 0) return 0;
        vector<int> dp(n+1, INT_MAX);
        dp[0] = 0;
        dp[1] = 1;
        for(int i = 1; i <= n; ++i){
            for(int j = 1; j*j <= i; ++j){
                dp[i] = min(dp[i - j*j] + 1, dp[i]);
            }
        }
        return dp[n];
    }
};

//using the static vector member
//We can reuse the dp table when testing multiple test case!
//This can only boost performance for multiple test case.
class Solution {
public:
    int numSquares(int n) {
        if(n <= 0) return 0;
        static vector<int> dp({0});
        int len = dp.size();
        for(int i = len; i <= n; ++i){
            dp.push_back(INT_MAX);
            for(int j = 1; j*j <= i; ++j){
                dp[i] = min(dp[i - j*j] + 1, dp[i]);
            }
        }
        return dp[n];
    }
};

//BFS solution. Very interesting!
class Solution {
public:
    int numSquares(int n) {
        if(n < 0) return 0;
        
        vector<int> possibleSquareNum;
        // record the minimum number of square numbers
        vector<int> graph(n+1, 0);
        for(int i = 1; i*i <= n; i++){
            possibleSquareNum.push_back(i * i);
            // if n == i*i, 1 will be the least number of square numbers
            graph[i*i] = 1; 
        }
        
        if(possibleSquareNum.back() == n)
            return 1;
        
        int currentLeast = 1;
        queue<int> Q;
        for(int i = possibleSquareNum.size() - 1; i >= 0; --i){
            Q.push(possibleSquareNum[i]);
        }
        
        while(!Q.empty()){
            int lenQ = Q.size();
            currentLeast++;
            
            for(int i = 0; i < lenQ; ++i){
                int tempSum = Q.front();
                Q.pop();
                for(int j = 0; j < possibleSquareNum.size(); ++j){
                    int num = tempSum + possibleSquareNum[j];
                    if(num == n)
                        return currentLeast;
                    else if(num < n && graph[num] == 0){ //graph[tempSum]== 0 indicates that we haven't explored this node
                        graph[num] = currentLeast;
                        Q.push(num);
                    }
                    else if(num > n)
                        break;
                }
            }
                
        }
        return 0;
    }
};

//322. Coin Change
//https://leetcode.com/problems/coin-change/
//Iterative version: Note how we handle initial case
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int len = coins.size();
        vector<int> dp(amount+1, -1);
        dp[0] = 0;
        int rem = amount;
        for(int i = 1; i <= amount; ++i){
            //We cannot use INT_MAX here, because in the dp[i-coins[j]]+1 will cause overflow
            int tempValue = amount+1;
            for(int j = 0; j < len; ++j){
                if(i - coins[j] >= 0)
                    tempValue = min(tempValue, dp[i-coins[j]]+1);
            }
            dp[i] = tempValue;
        }
        //If dp[amount] > amount, since tempValue is initialized with amount+1,
        //which means we did not do any calculation
        return dp[amount] > amount ? -1 : dp[amount];
    }
};

//recursive version
class Solution {
private:
    //helper() function will return the minimum number of coins we could have
    //in the given amount
    int helper(vector<int>& C, vector<int>& memo, int amount){
        if(amount < 0) return -1;
        if(amount == 0) return 0;
        if(memo[amount] != -2) return memo[amount];
        int minVal = numeric_limits<int>::max();
        for(int i = 0; i < C.size(); ++i){
            // here + 1 indicates that we need to add one more coin
            int res = helper(C, memo, amount - C[i])+1;
            //res could be -1 or positive value, should be > 0
            if(res > 0 && res < minVal)
                minVal = res; 
        }
        memo[amount] = (minVal == numeric_limits<int>::max()) ? -1 : minVal;
        return memo[amount];
    }
public:
    int coinChange(vector<int>& coins, int amount) {
        if(coins.size() == 0) return -1;
        //We cannot initilize the memo to be -1, since -1 could potentially be 
        //valid value from intermediate result.
        vector<int> memo(amount+1, -2);
        return helper(coins, memo, amount);
    }
};


//375. Guess Number Higher or Lower II
//https://leetcode.com/problems/guess-number-higher-or-lower-ii/
//Very tricky solution
//The idea is not complex, however, make it work needs some adjustment
//we always identify the range (i, j) incusively, and calculate the maximum
//local cost we can get in order to guarantee we have sufficient money for the game,
//then we minimize each possible local cost to find the optimal solution.
class Solution {
public:
    int getMoneyAmount(int n) {
        //dp[i][j] means that the minimum money that I should have 
        //to cover the potential worst strategy cost from range (i, j)
        //Inclusive
        vector<vector<int>> dp(n+1, vector<int>(n+1, 0));
        for(int j = 2; j <= n; ++j){
            for(int i = j-1; i >= 1; --i){
                int globalMin = numeric_limits<int>::max();
                for(int k = i+1; k < j; ++k){
                    int localMax = k + max(dp[i][k-1], dp[k+1][j]);
                    globalMin = min(globalMin, localMax);
                }
                dp[i][j] = (i+1 == j) ? i : globalMin;
            }
        }
        
        return dp[1][n];
    }
};


