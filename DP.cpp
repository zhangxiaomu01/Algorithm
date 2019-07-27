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

class Solution {
public:
    int climbStairs(int n) {
        //Initially, cur == dp[1], pre == dp[0]
        int cur = 1, pre = 0;
        for(int i = 1; i <= n; ++i){
            int temp = cur;
            cur = cur + pre;
            pre = temp;
        }
        return cur;
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
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = m ? obstacleGrid[0].size() : 0;
        if(m == 0 || n == 0 || obstacleGrid[0][0] == 1) return 0;
        //unsigned int to prevent overflow
        vector<vector<unsigned int>> dp(m, vector<unsigned int>(n, 0));
        dp[0][0] = 1;
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j){
                //skip the first case dp[0][0]
                if(i == 0 && j == 0) continue;
                if(obstacleGrid[i][j] != 1)
                    dp[i][j] = (i>=1 ? dp[i-1][j] : 0) + (j>=1 ? dp[i][j-1] : 0);
            }
        }
        return dp[m-1][n-1];
    }
};

//Recursive solution
class Solution {
private:
    int helper(vector<vector<int>>& G, int i, int j, vector<vector<int>>& memo){
        //Need to add G[0][0] != 1 here
        if(i == 0 && j == 0 && G[0][0] == 0) return 1;
        if(i < 0 || j < 0 || G[i][j] == 1) return 0;
        if(memo[i][j] != -1) return memo[i][j];
        memo[i][j] = helper(G, i-1, j, memo) + helper(G, i, j-1, memo);
        return memo[i][j];
    }
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = m ? obstacleGrid[0].size() : 0;
        vector<vector<int>> memo(m, vector<int>(n,-1));
        return helper(obstacleGrid, m-1, n-1, memo);
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

//Recursive version
class Solution {
private:
    //Note l should always be less than r
    //Almost the same idea as the iterative version
    int helper(vector<vector<int>>& memo, int l, int r){
        if(l >= r) return 0;
        if(memo[l][r] != 0) return memo[l][r];
        if(l+1 == r) return memo[l][r] = l;
        int localMax = 0;
        int globalMin = INT_MAX;
        for(int i = l+1; i < r; ++i){
            localMax = i + max(helper(memo, l, i-1), helper(memo, i+1, r));
            globalMin = min(globalMin, localMax);
        }
        memo[l][r] = globalMin;
        return memo[l][r];
    }
public:
    int getMoneyAmount(int n) {
        if(n <= 1) return 0;
        vector<vector<int>> memo(n+1, vector<int>(n+1, 0));
        return helper(memo, 1, n);
    }
};

/* O(n^2) solution is possible:  
https://leetcode.com/problems/guess-number-higher-or-lower-ii/discuss/84826/An-O(n2)-DP-Solution-Quite-Hard.*/


//312. Burst Balloons
//https://leetcode.com/problems/burst-balloons/
//Interesting DP problem. Record the last burst balloon is the key to solve the problem.
//Once get the idea, it's not hard. Try recursive version!
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int len = nums.size();
        //add 1 to both ends, so our valid range will be from 1 to len, inclusive
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        
        //dp[i][j] means the maximum coins we can get from index i to j, inclusive
        //Note the size should be corresponding to nums, which will be len + 2
        vector<vector<int>> dp(len+2, vector<int>(len+2, 0));
        //We build the table from the length 1 to len
        for(int cLen = 1; cLen <= len; ++cLen){
            //Define the left boundry, should start from 1 
            for(int l = 1; l <= len - cLen + 1; ++l){
                int r = l + cLen - 1; //maximum right boundry
                for(int i = l; i <= r; ++i){
                    //balloon i is the *last* balloon which is bursted. We save the maximum potential
                    //coins we can get if we choose i as the last balloon to burst in range [l, r]
                    //Note if r < l, dp[l][r] is 0
                    dp[l][r] = max(dp[l][r], nums[l-1]*nums[i]*nums[r+1] + dp[l][i-1] + dp[i+1][r]);
                }
            }
            
        }     
        //the range [1, len] in our new array
        return dp[1][len];
    }
};


//64. Minimum Path Sum
//https://leetcode.com/problems/minimum-path-sum/
/* 2D DP: MN matrix, can optimize to 1D array */
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = m ? grid[0].size() : 0;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        dp[m-1][n-1] = grid[m-1][n-1];
        for(int i = m-1; i >= 0; --i){
            for(int j = n-1; j >= 0; --j){
                if(i == m-1 && j == n-1) continue;
                if(i == m-1){
                    dp[i][j] = dp[i][j+1] + grid[i][j];
                }else if(j == n-1){
                    dp[i][j] = dp[i+1][j] + grid[i][j];
                }else{
                    dp[i][j] = min(dp[i+1][j], dp[i][j+1]) + grid[i][j];
                }
            }
        }
        return dp[0][0];
    }
};

//Optimized version
//using one dimensional array
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = m ? grid[0].size() : 0;
        vector<int> pre(n, 0);
        vector<int> cur(n, 0);
        pre[n-1] = grid[m-1][n-1];
        
        for(int i = n-2; i >= 0; --i){
            pre[i] = pre[i+1] + grid[m-1][i];
            
        }
        
        for(int i = m-2; i >= 0; --i){
            for(int j = n-1; j >= 0; --j){
                if(i == m-1 && j == n-1) continue;
                if(j == n-1){
                    cur[j] = pre[j] + grid[i][j];
                }  
                else
                    cur[j] = min(pre[j], cur[j+1]) + grid[i][j];
            }
            swap(cur, pre);
        }
        //Note in the end, we swap the pre and cur one more time
        return pre[0];
    }
};


//72. Edit Distance
//https://leetcode.com/problems/edit-distance/
/* Classical distance problem */
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        if(n == 0 || m == 0) return m ? m : n;

        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for(int i = 1; i <= m; ++i)
            dp[i][0] = i;
        for(int j = 1; j <= n; ++j)
            dp[0][j] = j;
        
        for(int i = 1; i <= m; ++i){
            for(int j = 1; j <= n; ++j){
                //Handle 4 situations:
                /*
                1. insert a character in word 1
                2. delete a character in word 2
                3. replace one character if word[i-1] != word[j-1]
                4. skip this character if word[i-1] == word[j-1]
                */
                dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]) + 1, 
                               dp[i-1][j-1] + (word1[i-1] == word2[j-1] ? 0 : 1));
            }
        }
        
        return dp[m][n];
    }
};

/* Optimized version, O(n) space */
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        if(n == 0 || m == 0) return m ? m : n;
              
        vector<int> cur(n+1, 0);
        vector<int> pre(n+1, 0);
        
        //Initialize the first row
        for(int i = 1; i <= n; ++i)
            pre[i] = i;
        
        for(int i = 1; i <= m; ++i){
            //Initialize cur[0] to be i, which means when word2 is empty,
            //we need i steps to transfer word 1 to word 2
            cur[0] = i;
            for(int j = 1; j <= n; ++j){
            //dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]) + 1, dp[i-1][j-1] + (word1[i-1] == word2[j-1] ? 0 : 1));
                cur[j] = min(min(cur[j-1], pre[j]) + 1, pre[j-1] + (word1[i-1] == word2[j-1] ? 0 : 1));
            }
            swap(cur, pre);
        }
        
        return pre[n];
    }
};


//
