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

//recursive version: hard to get it right
//I do not like the structure of current code, try it later
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

/* Slightly optimized version */
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        if(coins.empty()) return -1;
        if(amount == 0) return 0;
        
        int len = coins.size();
        //We start from the largest coin
        sort(coins.begin(), coins.end());
        
        vector<int> dp(amount+1, amount+1);
        dp[0] = 0;
        for(int j = len-1; j >= 0; --j){
            for(int i = coins[j]; i <= amount; ++i){
                //Only update those have already been checked 
                if(dp[i - coins[j]] != amount+1) 
                    dp[i] = min(dp[i-coins[j]] + 1, dp[i]);
            }
        }
        return dp[amount] == amount+1 ? -1 : dp[amount];
    }
};

/* BFS solution, too slow if coins are small and amount are large */
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        if(amount == 0) return 0;
        if(coins.empty()) return -1;
        queue<int> Q;
        vector<int> visited(amount+1, -1);
        
        sort(coins.begin(), coins.end());
        int len = coins.size();
        
        for(int i = len-1; i >= 0; --i){
            if(amount - coins[i] > 0){
                Q.push(amount - coins[i]);
                visited[amount - coins[i]] = 1;
            }else if (amount - coins[i] == 0) return 1;
        }
        //We already pushed 1
        int count = 1;
        while(!Q.empty()){
            int Qlen = Q.size();
            count++;
            for(int j = 0; j < Qlen; ++j){
                int newAmount = Q.front();
                Q.pop();
                for(int i = len-1; i >= 0; --i){
                    if(newAmount - coins[i] == 0) {
                        return count;
                    }
                    else if (newAmount - coins[i] > 0 ){
                        if(visited[newAmount - coins[i]] == -1)
                            Q.push(newAmount - coins[i]);
                        visited[newAmount - coins[i]] = min(visited[newAmount - coins[i]], count);
                    }else
                        break;
                }
            }
        }
        return -1;
    }
};

//518. Coin Change 2
//https://leetcode.com/problems/coin-change-2/
/* For such problems, we need to always start from the most
natural way to solve it. let dp[i][j] represents the ways 
if we have i types of coins in order to make up the amount j.
The dp formula is a little bit hard to get, which is 
dp[i][j] = dp[i-1][j] + dp[i][j-coins[j]].
We either give up coins[j] or select coins[j]. Note we build
the table from beginning to end, then we already consider the 
situation if amount j can include multiple coins[j]s when it
comes to j.*/
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        if(amount == 0) return 1;
        if(coins.empty()) return 0;
        int len = coins.size();
        vector<vector<int>> dp(len+1, vector<int>(amount+1, 0));
        dp[0][0] = 1;
        //dp[i][0] should always be 0 unless j == 0
        for(int j = 1; j <= amount; ++j){
            //If we have no coins, then we have no way to make up amount
            dp[0][j] = 0; 
        }
        //Note the outer loop will be coins, or it will make the
        //solution become complex. The meaning is that if we have 
        //coins[i] included, how many ways for us to get total amount
        for(int i = 1; i <= len; ++i){
            for(int j = 0; j <= amount; ++j){
                if(j == 0) {
                    dp[i][j] = 1;
                    continue;
                }
                //coins[i-1] corresponding to dp[i]
                if(j < coins[i-1]) dp[i][j] = dp[i-1][j];
                else
                    //Basically, we can either pickup coins[i-1] or not
                    //Imagine that if we pick up coins[i-1], then we will
                    //have dp[i][j-coins[i-1]] ways to make up the amount j;
                    //we also need to include the another possible way that
                    //we do not include coins[i-1]
                    dp[i][j] = dp[i][j - coins[i-1]] + dp[i-1][j]; 
            }
        }
        return dp[len][amount];
    }
};

/* Optimized version */
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        if(amount == 0) return 1;
        if(coins.empty()) return 0;
        int len = coins.size();
        //vector<vector<int>> dp(len+1, vector<int>(amount+1, 0));
        vector<int> cur(amount+1, 0);
        vector<int> pre(amount+1, 0);
        
        for(int i = 1; i <= len; ++i){
            for(int j = 0; j <= amount; ++j){
                //When j is 0, we have 1 way to pick up the coins
                if(j == 0) {
                    cur[j] = 1;
                    continue;
                }
                //coins[i-1] corresponding to cur[i]
                if(j < coins[i-1]) cur[j] = pre[j];
                else
                    cur[j] = cur[j - coins[i-1]] + pre[j];
                    //dp[i][j] = dp[i][j - coins[i-1]] + dp[i-1][j]; 
            }
            swap(cur, pre);
        }
        return pre[amount];
    }
};


/* Recursive version, very inefficient, hard to get it right - Try iterative 
version*/
class Solution {
private:
    int helper(int pos, int t, vector<int>& C, vector<vector<int>>& memo){
        if(t == 0) return 1;
        if(t < 0 || pos > C.size()) return 0;
        if(memo[pos][t] != -1) return memo[pos][t];
        
        int len = C.size();
        int res = 0;
        for(int i = pos; i <= len; ++i){
            if(t < C[i-1]) break;
            //We calculate how many coins i we can pick up for amount t
            int times = 1;
            while(times* C[i-1] <= t){
                //Calculate the next coins, so we pass i+1 here
                res += helper(i+1, t - times* C[i-1], C, memo);
                times++;
            }     
        }
        memo[pos][t] = res;
        return memo[pos][t];
    }
public:
    int change(int amount, vector<int>& coins) {
        if(amount == 0) return 1;
        if(coins.empty()) return 0;
        sort(coins.begin(), coins.end());
        int len = coins.size();
        vector<vector<int>> memo(len+1, vector<int>(amount+1, -1));
        return helper(1, amount, coins, memo);
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
    //dp[i][j] means that the minimum money that I should have 
    //to cover the potential worst strategy cost from range (i, j)
    //Inclusive
    int getMoneyAmount(int n) {
        if(n < 2) return 0;
        vector<vector<int>> dp(n+1, vector<int>(n+1, 0));
        //i is the starting index and j is the ending index
        for(int j = 1; j <= n; ++j){
            //The inner loop, we must start from right to left
            //dp[i][j] depends on dp[i][k-1] and dp[k+1][j]
            //if we start from i = 1 to j-1, there is no way
            //for us to know the information about dp[k+1][j]
            //when we explore dp[i][j]
            for(int i = j-1; i >= 1; --i){
                if(i+1 == j){
                    dp[i][j] = i;
                    continue;
                }
                int globalMin = INT_MAX;
                int localMax = 0;
                for(int k = i+1; k < j; ++k){
                    localMax = k + max(dp[i][k-1], dp[k+1][j]);
                    globalMin = min(globalMin, localMax);
                }
                
                dp[i][j] = globalMin;
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
        //We only have 1 element
        if(l >= r) return 0;
        if(memo[l][r] != 0) return memo[l][r];
        //Handle we only have two elements, we will always guess the 
        //smaller one
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
            //l will become smaller and smaller when cLen increases 
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


//97. Interleaving String
//https://leetcode.com/problems/interleaving-string/
/* Not that hard. Basically, we need to check whether the specific entry of dp can
be true or false based on whether which character form str1 or str2 maps to this 
entry in s3 */
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        int len1 = s1.size(), len2 = s2.size(), len3 = s3.size();
        if(len1 + len2 != len3) return false;
        if((len1 == 0 && s2 != s3) || (len2 == 0 && s1 != s3)) return false;
        vector<vector<short>> dp(len1+1, vector<short>(len2+1, 0));
        dp[0][0] = 1;
        
        //Check the base case that when one of the string is empty
        for(int i = 1; i <= len1; ++i){
            dp[i][0] = (dp[i-1][0] && s1[i-1] == s3[i-1]);
        }
        for(int j = 1; j <= len2; ++j){
            dp[0][j] = (dp[0][j-1] && s2[j-1] == s3[j-1]);
        }
        
        for(int i = 1; i <= len1; ++i){
            for(int j = 1; j <= len2; ++j){
                //if(i == 0 && j == 0) continue;
                dp[i][j] = (dp[i-1][j] && s1[i-1] == s3[i+j-1]) || 
                                (dp[i][j-1] && s2[j-1] == s3[i+j-1]);
            }
        }
        return dp[len1][len2];
        
    }
};


/* Optimized DP, 1D array */
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        int len1 = s1.size(), len2 = s2.size(), len3 = s3.size();
        if(len1 + len2 != len3) return false;
        if((len1 == 0 && s2 != s3) || (len2 == 0 && s1 != s3)) return false;
        
        if(len1 > len2) return isInterleave(s2, s1, s3);
        
        vector<int> cur(len2+1, 0);
        vector<int> pre(len2+1, 0);
        //vector<vector<short>> dp(len1+1, vector<short>(len2+1, 0));
        pre[0] = 1;
        
        //Check the base case that when one of the string is empty
        for(int i = 1; i <= len2; ++i){
            pre[i] = (pre[i-1] && s2[i-1] == s3[i-1]);
        }
        for(int i = 1; i <= len1; ++i){
            for(int j = 0; j <= len2; ++j){
                //dp[i][j] = (dp[i-1][j] && s1[i-1] == s3[i+j-1]) || (dp[i][j-1] && s2[j-1] == s3[i+j-1]);
                if(j == 0)
                    cur[j] = (pre[j] && s1[i-1] == s3[i + j - 1]);
                else
                    cur[j] = (pre[j] && s1[i-1] == s3[i+j-1]) || (cur[j-1] && s2[j-1] == s3[i+j-1]);
            }
            swap(cur, pre);
        }
        return pre[len2];
    }
};


//174. Dungeon Game
//https://leetcode.com/problems/dungeon-game/
/* The tricky part is we start from the Princess room, and calculate the minimum
HP we should have in order to reach room[i][j]. Once we find dp[i][j] > 0, then we
know we need at least dp[i][j] HP in order to pass room[i][j], if dp[i][j] <= 0,
then we know we have more HP than needed, we can set the minimum HP for dp[i][j],
which is 1. */
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        if(dungeon.empty()) return 1;
        int m = dungeon.size(), n = m ? dungeon[0].size() : 0;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        dp[m-1][n-1] = 1;
        for(int i = m-1; i >= 0; --i){
            for(int j = n-1; j >= 0; --j){
                if(i == m-1 && j == n-1) dp[i][j] = 1 - dungeon[i][j];
                else if(i == m-1) dp[i][j] = dp[i][j+1] - dungeon[i][j];
                else if(j == n-1) dp[i][j] = dp[i+1][j] - dungeon[i][j];
                else{
                    dp[i][j] = min(dp[i][j+1], dp[i+1][j]) - dungeon[i][j];
                }
                dp[i][j] = dp[i][j] <= 0 ? 1 : dp[i][j];
            }
        }
        return dp[0][0];
    }
};

/* Optimized version */
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        if(dungeon.empty()) return 1;
        int m = dungeon.size(), n = m ? dungeon[0].size() : 0;
        vector<int> cur(n, 0);
        vector<int> pre(n, 0);
        
        pre[n-1] = 1;

        for(int i = m-1; i >= 0; --i){
            for(int j = n-1; j >= 0; --j){
                if(i == m-1 && j == n-1) cur[j] = 1 - dungeon[i][j];
                else if(i == m-1) cur[j] = cur[j+1] - dungeon[i][j];
                else if(j == n-1) cur[j] = pre[j] - dungeon[i][j];
                else{
                    //dp[i][j] = min(dp[i][j+1], dp[i+1][j]) - dungeon[i][j];
                    cur[j] = min(cur[j+1], pre[j]) - dungeon[i][j];
                }
                //dp[i][j] = dp[i][j] <= 0 ? 1 : dp[i][j];
                cur[j] = cur[j] <= 0 ? 1 : cur[j];
            }
            swap(pre, cur);
        }
        //We have already checked pre[0] <= 0 before
        return pre[0];
    }
};


//221. Maximal Square
//https://leetcode.com/problems/maximal-square/
/* The key insight here is we use dp[i][j] record the maximum square from matrix[0][0] to matrix[i][j]. Inclusive.
Once we find a new '1' in the matrix, we know that dp[i][j] only depends on dp[i-1][j-1], dp[i-1][j] and dp[i][j-1].
Since the area is square, we can easily compute the potential minimum length for dp[i][j].*/
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if(matrix.empty()) return 0;
        int m = matrix.size(), n = m ? matrix[0].size() : 0;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        int maxLength = 0;
        for(int i = 0; i < m; ++i){
            if(matrix[i][0] == '1'){
                dp[i][0] = 1;
                maxLength = max(maxLength, dp[i][0]);
            }   
        }
        for(int j = 0; j < n; ++j){
            if(matrix[0][j] == '1'){
                dp[0][j] = 1;
                maxLength = max(maxLength, dp[0][j]);
            }   
        }
        for(int i = 1; i < m; ++i){
            for(int j = 1; j < n; ++j){
                if(matrix[i][j] == '1'){
                    /*//The following code is correct, however, we can just save the length
                    int len1 = sqrt(dp[i-1][j-1]);
                    int len2 = sqrt(dp[i-1][j]);
                    int len3 = sqrt(dp[i][j-1]);
                    int len = min(min(len1, len2), min(len1, len3)) + 1;
                    dp[i][j] = len * len;
                    maxArea = max(maxArea, dp[i][j]);
                    */
                    int len1 = dp[i-1][j-1], len2 = dp[i-1][j], len3 = dp[i][j-1];
                    int len = min(min(len1, len2), min(len1, len3)) + 1;
                    dp[i][j] = len;
                    maxLength = max(maxLength, dp[i][j]);                    
                }
            }
        }
        return maxLength * maxLength;
    }
};

class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if(matrix.empty()) return 0;
        int m = matrix.size(), n = m ? matrix[0].size() : 0;
        
        //vector<vector<int>> dp(m, vector<int>(n, 0));
        vector<int> cur(n, 0);
        vector<int> pre(n, 0);
        
        int maxLength = 0;
        
        for(int j = 0; j < n; ++j){
            if(matrix[0][j] == '1'){
                pre[j] = 1;
                maxLength = max(maxLength, pre[j]);
            }   
        }
        for(int i = 1; i < m; ++i){
            for(int j = 0; j < n; ++j){ 
                //Handle the corner case
                if(j == 0 && matrix[i][j] == '1'){
                    cur[0] = 1;
                    maxLength = max(maxLength, cur[0]);
                } 
                else if(j == 0 && matrix[i][j] == '0') cur[0] = 0;
                else if(matrix[i][j] == '1'){
                    int len1 = pre[j-1], len2 = pre[j], len3 = cur[j-1];
                    int len = min(min(len1, len2), min(len1, len3)) + 1;
                    cur[j] = len;
                    maxLength = max(maxLength, cur[j]);                    
                }
                else{
                    //Note using two 1 d array we need to reset cur[j] to 0
                    //if matrix[i][j] == '0'
                    cur[j] = 0;
                }
            }
            swap(cur, pre);
        }
        return maxLength * maxLength;
    }
};


//85. Maximal Rectangle
//https://leetcode.com/problems/maximal-rectangle/
/* The following algorithm is your first try. It's Wrong!!! Note we cannot get the 
right boundry and check the area if we only record the maximum width and height! */
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        if(matrix.empty()) return 0;
        int m = matrix.size(), n = m ? matrix[0].size() : 0;
        vector<vector<pair<int, int>>> dp(m, vector<pair<int,int>>(n, {0, 0}));
        
        int maxArea = 0;
        if(matrix[0][0] == '1') {
            dp[0][0] = make_pair(1, 1);
            maxArea = 1;
        }
        for(int i = 1; i < m; ++i){
            if(matrix[i][0] == '1'){
                //int h = max(dp[i-1][0].second, 1);
                dp[i][0].first = 1;
                dp[i][0].second += dp[i-1][0].second + 1;
                maxArea = max(maxArea, dp[i][0].second);
            }
        }
        for(int j = 1; j < n; ++j){
            //int w = max(dp[0][j-1].first, 1);
            if(matrix[0][j] == '1'){
                dp[0][j].first += dp[0][j-1].first + 1;
                dp[0][j].second = 1;
                maxArea = max(maxArea, dp[0][j].first);                
            }
        }
        //cout << maxArea <<endl;
        
        for(int i = 1; i < m; ++i){
            for(int j = 1; j < n; ++j){
                if(matrix[i][j] == '1'){
                    int localMWidth = min(dp[i-1][j-1].first, dp[i][j-1].first)+1;
                    int localMHeight = min(dp[i-1][j-1].second, dp[i-1][j].second)+1;
                    int localMaxW = max(dp[i-1][j-1].first, dp[i][j-1].first) + 1;
                    int localMaxH = max(dp[i-1][j-1].second, dp[i-1][j].second)+1;
                    cout << localMWidth << " " << localMHeight << endl;
                    maxArea = max(maxArea, localMWidth * localMHeight);
                    maxArea = max(maxArea, localMaxW * localMHeight);
                    dp[i][j] = make_pair(localMWidth, localMHeight);
                }
                else{
                    dp[i][j] = make_pair(0, 0);
                }
            }
        }
        
        return maxArea;
    }
};

/* We propose the following algorithms to solve the problems. Their ideas are similar.
However, it's critical to come up with the idea that we need to record the height
first. */
/* DP solution. We need to record the potential minimum right and maximum left
boundry, and check the valid maximum area for each entry. At first, we also
need to record the maximum height for each entry. Then we calculate the maximum 
area row by row.*/
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        if(matrix.size() == 0) return 0;
        int m = matrix.size(), n = m ? matrix[0].size() : 0;
        
        //Note left[i] represents entry i, right[j] represents entry j+1
        vector<int> height(n, 0), left(n, 0), right(n, n);
        int maxArea = 0;
        
        for(int i = 0; i < m; ++i){
            //update height and left boundry
            int curL = 0;
            for(int j = 0; j < n; ++j){
                if(matrix[i][j] == '1'){
                    height[j]++;
                    
                    left[j] = max(left[j], curL);
                }else{
                    height[j] = 0;
                    //if matrix[i][j] == '0', then curL should be next potential
                    //'1', which is j+1
                    left[j] = 0;
                    curL = j+1;
                }
            }
            
            int curR = n;
            for(int j = n-1; j >=0; --j){
                if(matrix[i][j] == '1')
                    right[j] = min(right[j], curR);
                else{
                    //when matrix[i][j] == '0', curR should be j
                    //Note right[j] is pointing to a '0' which strictly follows a '1'
                    right[j] = n;
                    curR = j;
                }
            }
            //Now for each entry i, we already have the potantial maximum left boundry and 
            //minimum right boundry and maximum height, we can compute the area one by one
            //Note we compute the area from top to bottom, the situation like below:
            /*
                0 0 0 1 1 1 0 0 0
                0 1 1 1 1 1 1 1 0
            */
            //will be handled correctly!
            for(int i = 0; i < n; ++i){
                maxArea = max(maxArea, (right[i] - left[i]) * height[i]);
            }
        }
        return maxArea;
    }
};


//198. House Robber
//https://leetcode.com/problems/house-robber/
/* Not that hard. */
class Solution {
public:
    int rob(vector<int>& nums) {
        if(nums.empty()) return 0;
        int len = nums.size();
        vector<int> dp(len+1, 0);
        dp[0] = 0;
        for(int i = 1; i <= len; ++i){
            if(i >= 2)
                dp[i] = max(dp[i-1], dp[i-2] + nums[i-1]);
            else dp[i] = nums[i-1];
        }
        return dp[len];
    }
};

//Great Explanation:
//https://leetcode.com/problems/house-robber/discuss/156523/From-good-to-great.-How-to-approach-most-of-DP-problems.
class Solution {
public:
    int rob(vector<int>& nums) {
        int len = nums.size();
        vector<int> memo(len, -1);
        return rec(len-1, nums, memo);
    }
    int rec(int i, vector<int>& nums, vector<int>&memo){
        if(i < 0)
            return 0;
        if(memo[i] >= 0){
            return memo[i];
        }
        memo[i] = max(rec(i-1, nums, memo), rec(i-2, nums, memo) + nums[i]); 
        return memo[i];
    }
};


//213. House Robber II
//https://leetcode.com/problems/house-robber-ii/
/* The general idea is similar to House Robber I. 
We break the circle and consider the two situations separately.
1. We rob the first house, and ignore the last house
2. We start from the second house, and iterate to the last one
We can merge the code together, however, it will be hard to read*/
class Solution {
public:
    int rob(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        //Handle the special case
        if(len == 1) return nums[0];
        
        vector<int> dp1(len+1, 0);
        vector<int> dp2(len+1, 0);
        dp1[0] = 0;
        dp2[0] = 0;
        //Rob the first house
        for(int i = 1; i <= len; ++i){
            if(i < 2) dp1[i] = nums[i-1];
            else
                dp1[i] = max(dp1[i-1], dp1[i-2] + nums[i-1]);
        }
        
        //Ignore the first house
        for(int i = 2; i <= len; ++i){
            if(i == 2) dp2[i] = nums[i-1];
            else 
                dp2[i] = max(dp2[i-1], dp2[i-2] + nums[i-1]);
        }
        return max(dp1[len-1], dp2[len]);
    }
};

//Two passes, remember in House Rober I:
//We have 0 - n-1 houses to robber, now, we have two cases:
//case 1: 0 - n-2; case 2: 1 - n-1
class Solution {
private:
    int doRob(const vector<int>& n, int start, int end){
        int len = n.size();
        int preNo = 0; int cur = n[start];
        for(int i = start + 2; i <= end; i++){
            int temp = max(preNo + n[i-1], cur);
            preNo = cur;
            cur = temp;
        }
        return cur;
    }
public:
    int rob(vector<int>& nums) {
        int len = nums.size();
        if(len < 2) return len ? nums[0] : 0;
        return max(doRob(nums, 0, len-1), doRob(nums, 1, len));
    }
};


//337. House Robber III
//https://leetcode.com/problems/house-robber-iii/
/* Post order traversal + DP */
class Solution {
private:
    void treeTraversal(TreeNode* node, unordered_map<TreeNode*, pair<int, int>>& dict){
        if(node == nullptr) return;
        treeTraversal(node->left, dict);
        treeTraversal(node->right, dict);
        
        if(!node->left && !node->right){
            dict[node] = make_pair(node->val, 0);
        }else{
            int chosenNode = node->val + (node->right ? dict[node->right].second : 0) + (node->left ? dict[node->left].second : 0);
            
            int chosenLeftNode = node->left ? max(dict[node->left].first, dict[node->left].second) : 0;
            int chosenRightNode = node->right ? max(dict[node->right].first, dict[node->right].second) : 0;
            dict[node] = make_pair(chosenNode, chosenLeftNode + chosenRightNode);
        }
    }
public:
    int rob(TreeNode* root) {
        //The first value in the pair indicates that select this node, what 
        //will be the maximum profit, second indicates that not select this
        //node, what will be the maximum profit
        unordered_map<TreeNode*, pair<int, int>> dict;
        treeTraversal(root, dict);
        return max(dict[root].first, dict[root].second);
        
    }
};

/* Another implementation, same idea. */
class Solution {
private:
    deque<TreeNode*>* buildQueue(TreeNode* node){
        deque<TreeNode*>* Qptr = new deque<TreeNode*>();
        stack<TreeNode*> st;
        st.push(node);
        while(!st.empty()){
            TreeNode* tempNode = st.top();
            st.pop();
            Qptr->push_front(tempNode);
            if(tempNode->right) st.push(tempNode->right);
            if(tempNode->left) st.push(tempNode->left);
        }
        return Qptr;
    }
    
public:
    int rob(TreeNode* root) {
        if(!root) return 0;
        deque<TreeNode*>* Qptr = buildQueue(root);
        //cout << Qptr->size() << endl;
        unordered_map<TreeNode*, pair<int, int>> dict;
        //The first element of pair means without robbing the house, the potential maximum profit; second means we robber the house
        dict.insert({nullptr, make_pair(0, 0)});
        for(auto it = Qptr->begin(); it != Qptr->end(); it++){
            //The max profit we can get if we rob the node
            int robNode = dict[(*it)->left].first + dict[(*it)->right].first + (*it)->val;
            //The max profit we can get if we do not rob the node
            int robnNode = max(dict[(*it)->left].first, dict[(*it)->left].second) + max(dict[(*it)->right].first, dict[(*it)->right].second);
            //Save the result to hash table
            dict[(*it)] = make_pair(robnNode, robNode);
            //cout <<"node is" << (*it)->val <<" The maximum rob it: " << robNode << " The maximum not rob it: " << robnNode <<endl; 
        }
        
        return max(dict[root].first, dict[root].second);
    }
};

/* Same idea, most beautiful idea! No need to maintain the queue or set */
class Solution {
private:
    pair<int, int> doRob(TreeNode* node){
        if(!node) return make_pair(0, 0);
        auto l = doRob(node->left);
        auto r = doRob(node->right);
        int robNodeProfit = l.first + r.first + node->val;
        int nRobNodeProfit = max(l.first, l.second) + max(r.first, r.second);
        return make_pair(nRobNodeProfit, robNodeProfit);
    }
public:
    int rob(TreeNode* root) {
        if(!root) return 0;
        auto robHouse = doRob(root);
        return max(robHouse.first, robHouse.second);
    }
};

//300. Longest Increasing Subsequence
//https://leetcode.com/problems/longest-increasing-subsequence/
/* The general idea is to use dp[i] to keep track of the 
longest increasing subsequence from 0 to i, inclusive. Then
we check j from [0, i), if A[i] > A[j], we can know now 
A[i] = max(A[i], A[j+1]). DP[j] represents nums[j-1]*/
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        if(nums.empty()) return 0;
        int len = nums.size();
        //Note as long as we have 1 element, it should
        //be at least 1
        vector<int> dp(len+1, 1);
        dp[0] = 0;
        int maxLen = 0;
        for(int i = 1; i <= len; ++i){
            for(int j = 0; j < i; ++j){
                //note dp[j+1] represents to nums[j]
                if(nums[i-1] > nums[j])
                    dp[i] = max(dp[i], dp[j+1] + 1);
                maxLen = max(maxLen, dp[i]);
            }
        }
        return maxLen;
    }
};


/* Recursive solution : Inefficient. Avoid. Convert the problem to be whether 
we select nums[i], based on whether nums[i] > nums[preHigher]. */
class Solution {
private:
    //lower represents the previous higher number, initially should be INT_MIN
    int helper(vector<int>& nums, int curPos, int preHigher, vector<vector<int>>& memo){
        if(curPos >= nums.size()) return 0;
        if(memo[curPos][preHigher] != -1) return memo[curPos][preHigher];
        
        //Chosen nums[pos]
        int chosenNumI = 0;
        if(nums[curPos] > nums[preHigher])
            chosenNumI = 1 + helper(nums, curPos+1, curPos, memo);
        
        //Not chosen nums[pos]
        int nChosenNumI = helper(nums, curPos+1, preHigher, memo);
        
        memo[curPos][preHigher] = max(chosenNumI, nChosenNumI);
        return memo[curPos][preHigher];
        
    }
public:
    int lengthOfLIS(vector<int>& nums) {
        if(nums.empty()) return 0;
        nums.insert(nums.begin(), INT_MIN);
        int len = nums.size();
        vector<vector<int>> memo(len+1, vector<int>(len+1, -1));
        return helper(nums, 1, 0, memo);
    }
};

/* Lower bound solution: O(nlogn) */
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        if(nums.empty()) return 0;
        int len = nums.size();
        vector<int> res;
        for(int i = 0; i < len; ++i){
            auto it = lower_bound(res.begin(), res.end(), nums[i]);
            if(it == res.end())
                res.push_back(nums[i]);
            else 
                *it = nums[i];
        }
        return res.size();
    }
};


//673. Number of Longest Increasing Subsequence
//https://leetcode.com/problems/number-of-longest-increasing-subsequence/
/* The general idea is to use two arrays. The idea how to calculate the maximum 
increasing sequence is exactly as the problem 300. The tricky part is how we 
count the number of ways for each length. Note we only care about the longest
potential ways, and 1 element smaller than longest potential ways.*/
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        if(nums.empty()) return 0;
        int len  = nums.size();
        //dp[i] represents the maximum potential length 
        //ending with length i
        vector<int> dp(len+1, 1);
        dp[0] = 0;
        //count should be at least one
        vector<int> count(len+1, 1);
        count[0] = 1;
        
        int maxLen = 0;
        for(int i = 1; i <= len; ++i){
            for(int j = 0; j < i; ++j){
                if(nums[i-1] > nums[j]){
                    //dp[i] = max(dp[i], dp[j+1] + 1);
                    if(dp[i] < dp[j+1] + 1){
                        dp[i] = dp[j+1] + 1;
                        count[i] = count[j+1];
                    }else if (dp[i] == dp[j+1] + 1){
                        //We need to add all the potential count
                        //with the same length. since nums[i-1] > nums[j]
                        //we only need to take dp[j+1] + 1 into account
                        count[i] += count[j+1];
                    }
                    maxLen = max(maxLen, dp[i]);
                }
            }
        }
        
        int res = 0;
        for(int i = 0; i <= len; ++i){
            if(dp[i] == maxLen)
                res +=  count[i];
        }   
        return res;
    }
};


//363. Max Sum of Rectangle No Larger Than K
//https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/
//A very good explanation:
//https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/discuss/83599/Accepted-C%2B%2B-codes-with-explanation-and-references
//https://www.youtube.com/watch?v=yCQN096CwWM
//In general, it's exhaustive search
class Solution {
public:
    int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
        if(matrix.empty()) return 0;
        int m = matrix.size(), n = m ? matrix[0].size() : 0;
        //curL and curR point to the specific column
        //Initialization curL == curR
        int curL = 0, curR = 0;
        //res should be the smallest number, we will potentially
        //have k <= 0
        int res = INT_MIN;
        
        for(curL = 0; curL < n; ++curL){
            //represents one column of elements
            vector<int> A(m, 0);
            //curR start from curL
            for(curR = curL; curR < n; ++curR){
                for(int i = 0; i < m; ++i){
                    //calculate the sum row by row
                    //Note A[i] also represents the cumulative sum
                    //from curL to curR
                    A[i] += matrix[i][curR];
                }
                
                set<int> dict;
                //represents the empty array, which has 0 elements 
                //to sum
                dict.insert(0);
                //for sum of sub array (i, j], we have curSum[j] - curSum[i]
                //Each time for the loop, we insert curSum[i] to our set,
                //then when we find the closest curSum[i] close to k, it's
                //equivalent to curSum[j] - curSum[i] >= k, then we need to 
                //find a previous sum, which is lower_bounded by curSum[j] - k
                int maxSum = INT_MIN;
                int curSum = 0;
                for(int i = 0; i < m; ++i){
                    curSum += A[i];
                    auto it = dict.lower_bound(curSum - k);
                    if(it != dict.end()) maxSum = max(maxSum, curSum - *it);
                    dict.insert(curSum);
                }
                res = max(res, maxSum);
                
                if(res == k) return res;
            }
        }
        return res;
    }
};

/* Optimize version. Exact the same idea like the previous one. Exception that we
add some acceleration code */
class Solution {
public:
    int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
        if(matrix.empty()) return 0;
        int m = matrix.size(), n = m ? matrix[0].size() : 0;
        int curL = 0, curR = 0;

        int res = INT_MIN;
        
        for(curL = 0; curL < n; ++curL){
            //represents one column of elements
            vector<int> A(m, 0);
            //curR start from curL
            for(curR = curL; curR < n; ++curR){
                for(int i = 0; i < m; ++i){
                    A[i] += matrix[i][curR];
                }
                int curSum = 0;
                
                //Acceleration
                for(int n : A){
                    curSum += n;
                    if(curSum == k || n == k) return k;
                    if(curSum < k && curSum > res) res = curSum;
                    //make sure that our curSum will always represents 
                    //value greater than or equal to 0
                    //Nee this line, cannot understand it
                    //without it, the following test case will fail
                    /*
                    [[5,-4,-3,4],[-3,-4,4,5],[5,1,5,-4]] 8
                    */
                    if(curSum < 0) curSum = 0;
                }
                if(curSum < k) continue;
                
                curSum = 0;
                set<int> dict;
                dict.insert(0);
                int maxSum = INT_MIN;
                
                for(int i = 0; i < m; ++i){
                    curSum += A[i];
                    auto it = dict.lower_bound(curSum - k);
                    if(it != dict.end()) maxSum = max(maxSum, curSum - *it);
                    if(maxSum == k) return k;
                    dict.insert(curSum);
                }
                res = max(res, maxSum);
            }
        }
        return res;
    }
};


//91. Decode Ways
//https://leetcode.com/problems/decode-ways/
/* Unoptimized version, most natural way. This problem is not hard, the only thing
is that you need to figure out all the corner cases. */
class Solution {
public:
    int numDecodings(string s) {
        int len = s.size();
        if(s[0] == '0') return 0;
        vector<int> dp(len+1, 0);
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= len; ++i){
            //dp[i] represents s[i-1], in order to get the element before s[i-1]
            //we need i-2
            if(s[i-1] == '0' && stoi(s.substr(i-2, 2)) > 26)
                return 0;
            
            if(s[i-1] == '0' && s[i-2] == '0')
                return 0;
            
            if(s[i-1] == '0' && stoi(s.substr(i-2, 2)) <= 26)
                dp[i] = dp[i-2];
            else if(s[i-2] != '0' && stoi(s.substr(i-2, 2)) <= 26){
                dp[i] = dp[i-2] + dp[i-1];
            }else
                dp[i] = dp[i-1];
        }
        return dp[len];
    }
};

/* Optimized version. O(1) space */
class Solution {
public:
    int numDecodings(string s) {
        int len = s.size();
        if(s[0] == '0') return 0;
        
        int pre = 1, cur = 1;
        for(int i = 2; i <= len; ++i){
            //cout << stoi(s.substr(i-2, 2)) << endl;
            //dp[i] represents s[i-1]
            if(s[i-1] == '0' && stoi(s.substr(i-2, 2)) > 26)
                return 0;
            
            if(s[i-1] == '0' && s[i-2] == '0')
                return 0;
            
            if(s[i-1] == '0' && stoi(s.substr(i-2, 2)) <= 26){
                swap(cur, pre);
                //dp[i] = dp[i-2];
            }
                
            else if(s[i-2] != '0' && stoi(s.substr(i-2, 2)) <= 26){
                int temp = cur;
                cur = cur + pre;
                pre = temp;
                //dp[i] = dp[i-2] + dp[i-1];
            }else
                pre = cur;
                //dp[i] = dp[i-1];

        }
        return cur;
    }
};



//44. Wildcard Matching
//https://leetcode.com/problems/wildcard-matching/
/* Original DP */
class Solution {
public:
    bool isMatch(string s, string p) {
        int lenS = s.size(), lenP = p.size();
        vector<vector<int>> dp(lenS+1, vector<int>(lenP+1, 0));
        dp[lenS][lenP] = 1;
        for(int j = lenP-1; j >= 0; --j){
            //s is empty and p[j] == '*'
            if(p[j] == '*')
                dp[lenS][j] = dp[lenS][j+1];
        }
        for(int i = lenS-1; i >= 0; --i){
            for(int j = lenP-1; j >= 0; --j){
                //if(i == lenS && j == lenP) continue;
                if(p[j] == '?' || s[i] == p[j])
                    dp[i][j] = dp[i+1][j+1];
                else if(p[j] == '*'){
                    //dp[i+1][j] - means that we skip one character of s
                    //dp[i][j+1] - '*' maps to empty
                    //At first, I used a third loop here, it's not necessary
                    //dp[i+1][j] has included the situation that 
                    //'*' maps to arbitrary characters
                    dp[i][j] = dp[i+1][j] || dp[i][j+1]; 
                    
                }else if(s[i] != p[j]){
                    dp[i][j] = 0;
                }
            }
        }
        return dp[0][0];
    }
};

/* Optimized DP solution */
class Solution {
public:
    bool isMatch(string s, string p) {
        int lenS = s.size(), lenP = p.size();
        //vector<vector<int>> dp(lenS+1, vector<int>(lenP+1, 0));
        //dp[lenS][lenP] = 1;
        vector<int> cur(lenS+1, 0);
        vector<int> pre(lenS+1, 0);
        //if(p.back() == '*')
        pre[lenS] = 1; 
        //We need to put the j for loop out side in order to
        //update cur[lenS] everytime before i loop
        for(int j = lenP-1; j >= 0; --j){
            //Update the last element, we need to do this
            cur[lenS] = pre[lenS] && p[j] == '*'; 
            for(int i = lenS-1; i >= 0; --i){
                if(p[j] == '?' || s[i] == p[j])
                    cur[i] = pre[i+1];
                else if(p[j] == '*'){
                    //this part I did not get by myself
                    cur[i] = cur[i+1] || pre[i]; 
                }else if(s[i] != p[j]){
                    cur[i] = 0;
                }
            }
            swap(cur, pre);
        }
        return pre[0];
    }
};


/* Two pointers */
/* Two pointer approach: The general idea is to maintain 
two pointers. pointer j will move forward and detect asterisk.
When we find the asterisk, we can set posStar to that position,
and move i pointer accordingly. Not easy to get it right 
in the interview!!!*/

class Solution {
public:
    bool isMatch(string s, string p) {
        int lenS = s.size(), lenP = p.size();
        int i = 0, j = 0;
        //posStar will record the last asterisk we have
        //encountered. pos_sStart will record the position
        //of the pointer i in s when we encounter an asterisk
        int posStar = -1, pos_sStart = 0;
        
        while(i < lenS){
            if(j < lenP && (s[i] == p[j] || p[j] == '?')){
                i++;
                j++;
            }else if (j < lenP && p[j] == '*'){
                //We move to the next element of p
                //Note "****" is equivalent to '*'
                posStar = j;
                j++;
                pos_sStart = i;
            }else if(posStar != -1){
                //If s[i] != s[j] some rounds after we encounter 
                //the asterisk. We start from posStar + 1 again
                //because '*' need to represent more elements
                j = posStar + 1;
                //We need to increment pos_sStart, means '*' represents
                //more elements
                pos_sStart ++;
                i = pos_sStart;
            }
            else
                return false;
        }
        
        //When i reaches the end, if j has some element other than '*'
        //return false
        for(; j < lenP; ++j){
            if(p[j] != '*')
                return false;
        }
        
        return true;
    }
};


//10. Regular Expression Matching
//https://leetcode.com/problems/regular-expression-matching/
/* Recursive solution, pretty straightforward. 
I am having trouble to understand why we match s[i+1] to p[j],
then that can represent '*' can match to the preceding 
character multiple times. */
class Solution {
    bool checkMatch(string s, string p){
        if(p.empty()) return s.empty();
        
        bool firstMatch = !s.empty() && (s[0] == p[0] || p[0] == '.');
        
        if(p.size() >= 2 && p[1] == '*')
            //if p[j] followed by '*', we could either match ith element 
            //in s to j+2 th element in p (remove p[j]); or we matched 
            //s[i] and p[j],and we now need to match s[i+1] to p[j] 
            //(duplicate p[j]) 
            return checkMatch(s, p.substr(2)) || (firstMatch && checkMatch(s.substr(1), p));
        else
            return firstMatch && checkMatch(s.substr(1), p.substr(1));
        
    }
public:
    bool isMatch(string s, string p) {
        return checkMatch(s, p);
    }
};

/* DP solution, contains tricky part, I do not really like it. */
class Solution {
public:
    bool isMatch(string s, string p) {
        int lenS = s.size(), lenP = p.size();
        bool dp[lenS+1][lenP+1];
        memset(dp, false, sizeof(dp));
        //represent the matching of two empty string
        dp[lenS][lenP] = true;
        
        for(int i = lenS; i >= 0; --i){
            for(int j = lenP; j >= 0; --j){
                //Ignore base case
                if(i == lenS && j == lenP) continue;
                //check character s[i], p[j]
                bool firstMatch = i < lenS && j < lenP && ((s[i] == p[j]) || (p[j] == '.'));
                //We cannot add i < lenS here, the dp[i+1][j] will not be able to 
                //go out of bounds because in firstMatch, if i >= lenS, firstMatch 
                //will be false
                if(j + 1 < lenP && p[j+1] == '*'){
                    //dp[i+1][j] means '*' represent preceding character
                    //once
                    dp[i][j] = dp[i][j+2] || (firstMatch && dp[i+1][j]);
                }else
                    dp[i][j] = firstMatch && dp[i+1][j+1];
            }
        }
        return dp[0][0];
        
    }
};

/* DP Solution, more intuitive for me */
class Solution {
public:
    bool isMatch(string s, string p) {
        int len_s = s.size(), len_p = p.size();
        bool dp[len_s+1][len_p+1];
        memset(dp, false, sizeof(dp));
        dp[len_s][len_p] = true;
        for(int j = len_p-1; j >= 0; --j){
            if(j + 1 < len_p && p[j+1] == '*')
                dp[len_s][j] = dp[len_s][j+2];
        }
        //We can start from len_s - 1 now for we have handled
        //the last row and last column
        for(int i = len_s-1; i >= 0; --i){
            for(int j = len_p-1; j >= 0; --j){
                bool firstMatch = (s[i] == p[j]) || (p[j] == '.');
                
                if(j+1 < len_p && p[j+1] == '*'){
                    dp[i][j] = dp[i][j+2] || (firstMatch && dp[i+1][j]);
                }else
                    dp[i][j] = firstMatch && dp[i+1][j+1];
            }
        }
        return dp[0][0];
    }
};

/* Optimized DP */
class Solution {
public:
    bool isMatch(string s, string p) {
        int len_s = s.size(), len_p = p.size();
        //bool dp[len_s+1][len_p+1];
        bool cur[2][len_p+1];
        memset(cur, false, sizeof(cur));
        cur[1][len_p] = true;
        //dp[len_s][len_p] = true;
        for(int j = len_p-1; j >= 0; --j){
            if(j + 1 < len_p && p[j+1] == '*')
                cur[1][j] = cur[1][j+2];
        }
        int count = 0;
        //We cannot swap the two loops here, since we chose
        //len_p to be the array
        for(int i = len_s-1; i >= 0; --i){
            for(int j = len_p-1; j >= 0; --j){
                bool firstMatch = (s[i] == p[j]) || (p[j] == '.');
                
                if(j+1 < len_p && p[j+1] == '*'){
                    cur[count%2][j] = cur[count%2][j+2] || (firstMatch && cur[(count+1)%2][j]);
                }else
                    cur[count%2][j] = firstMatch && cur[(count+1)%2][j+1];
                //cout << cur[count%2][j] << " ";
            }
            //cout << endl;
            //swap previous and current array
            count++;
            //Initialize cur to be false. This is important here. Since
            //it is not counting dp, we have to eliminate the influence
            //from the previous array
            for(int j = 0; j <= len_p; ++j)
                cur[count%2][j] = false;
            
        }
        return cur[(count+1)%2][0];
    }
};


