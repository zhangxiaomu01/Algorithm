#include<windows.h>
#include<algorithm>
#include<vector>
#include<array>
#include<cmath>
using namespace std;

//121. Best Time to buy and sell stock 
//https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
//Easy problem

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.empty()) return 0;
        int maxSofar = 0, maxCur = 0;
        for(int i = 1; i < prices.size(); i++){
            maxCur += prices[i] - prices[i-1];
            maxCur = max(0, maxCur);
            maxSofar = max(maxSofar, maxCur);
        }
        return maxSofar;
    }

};

//122. Best Time to buy and sell stock II
//https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
//Not that bad, just include all the potential profit into our max sum
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



//123. Best Time to buy and sell stock III
//https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
//Essentially a DP problem. The idea is to keep track of the maximum profit
//from [0, i-1] and [i, n-1]. which represents the first trasaction and second
//transaction respectively.
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        if(len <= 1) return 0;
        //Record the max profit from 0 to ith day
        vector<int> profitForward(len, 0);
        int finalProfit = 0;       

        //The first pass, we record the max potential profit if we 
        //complete the transaction 1 from day 0 to day i
        int minSofar = numeric_limits<int>::max();
        int maxProfit = 0;
        for(int i = 0; i < len; i++){
            minSofar = min(minSofar, prices[i]);
            maxProfit = max(maxProfit, prices[i] - minSofar);
            profitForward[i] = maxProfit;
        }

        //The second pass, we iterate backward, and calculate the max
        //profit from day i to n-1, so the potential max profit could
        //be sum of the two potential max profit.
        int maxSofar = numeric_limits<int>::min();
        maxProfit = 0;
        for(int i = len-1; i >=0; i--){
            maxSofar = max(maxSofar, prices[i]);
            maxProfit = max(maxProfit, maxSofar - prices[i]);
            profitForward[i] = profitForward[i] + maxProfit;
            finalProfit = max(profitForward[i], finalProfit);
        }
        return finalProfit;
    }
};

//A more simplified version which only takes O(1) space complexity...
//Same concept, but hard to come up with this idea.
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.empty()) return 0;
        int len_p = prices.size();
        int buy1 = prices[0], buy2 = prices[0];
        int sell1 = 0, sell2 = 0;
        for(int i = 1; i < len_p; i++){
            //We calculate the max profit for day i to n-1
            //Note we have sell1 here, which is clever
            buy2 = min(buy2, prices[i] - sell1);
            sell2 = max(sell2, prices[i] - buy2);

            //This is essentially the max profit for day 0 to i
            buy1 = min(buy1, prices[i]);
            sell1 = max(sell1, prices[i] - buy1);
        }
        return sell2;
    }
};



//188. Best Time to Buy and Sell Stock IV
//https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
//A hard dp problem, this is natural recursive + memorization solution
//Basically, we need to know what's the profit if we can buy and sell at most i times
//and we can buy and sell from 0 - j day, in the end, we will have our dp table
//Great explanation: from Recursion to iteration
//https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/discuss/228409/3-C%2B%2B-Simple-and-intuitive-solutions
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int len = prices.size();
        if(len <= 1) return 0;
        int maxVal = 0;      

        if(k >= len/2){
            for(int i = 1; i < len; i++){
                maxVal += max(prices[i] - prices[i-1], 0);
            }
            return maxVal;
        }      

        vector<vector<int>> memo(k+1, vector<int>(len+1, -1));        
        for(int i = 1; i <=k; i++){
            for(int j = 0; j < len; j++){
                maxVal = max(maxVal, rec(k, prices, memo, i, j));
            }
        }
        return maxVal;
    }

    int rec(int k, vector<int>& p, vector<vector<int>>& memo, int trade, int day){
        if(trade <=0 || trade > k || day <= 0 || day >= p.size())
            return 0;       

        if(memo[trade][day]!= -1)
            return memo[trade][day];

        //search the max profit for 0 to day, and we decide to sell 1 share at this day
        int preProfit = 0;
        for(int i = 0; i < day; i++){
            preProfit = max(preProfit, p[day] - p[i] + rec(k, p, memo, trade-1, i));
        }

        //We do not sell at this day, may do some optimization here
        //int profit = memo[trade][day-1] == -1 ? rec(k, p, memo, trade, day-1) : memo[trade][day-1]
        int profit = rec(k, p, memo, trade, day-1);

        //We update the memorization table
        memo[trade][day] = max(profit, preProfit);
        return memo[trade][day];
    }
};



//A better choice for iterative implementation
//This is exactly the same idea like the recursive version, this time, we also
//keep track of whether we sell a share at day q or not. And then calculate dp[i][j]:
//max profit if we can do at most i transactions through first j days
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int len = prices.size();
        if(len < 2) return 0;
        
        int maxVal = 0;
        if(k >= len/2){
            for(int i = 1; i < len; i++){
                maxVal += max(0, prices[i] - prices[i-1]);
            }
            return maxVal;
        }

        vector<vector<int>> dp(k+1, vector<int>(len, 0));
        for(int i = 1; i <= k; i++){
            for(int j = 1; j < len; j++){
                int preProfit = 0;
                for(int q = 0; q < j; q++){
                    preProfit = max(preProfit, prices[j] - prices[q] + dp[i-1][q]);
                    //The line below will not work, you need to calculate the max value
                    //of sell a share at day j first
                    //dp[i][j] = max(prices[j] - prices[q] + dp[i-1][q-1], dp[i][j-1]);
                }
                dp[i][j] = max(preProfit, dp[i][j-1]);
            }
        }
        return dp[k][len-1];
    }
};



//Optimized solution, get rid of the third loop
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int len = prices.size();
        if(len < 2) return 0;
        
        int maxVal = 0;
        if(k >= len/2){
            for(int i = 1; i < len; i++){
                maxVal += max(0, prices[i] - prices[i-1]);
            }
            return maxVal;
        }
        vector<vector<int>> dp(k+1, vector<int>(len, 0));
        for(int i = 1; i <= k; i++){
            //instead of keep traking of the max profit, we keep
            //track of the minimum value, the initial min should be prices[0]
            //Considering the following dp equation, we start j with 1
            int minV = prices[0];
            for(int j = 1; j < len; j++){
                //This is exactly the same idea like best buy and sell stock III, we keep 
                //track of what's the maximum potential profit before some day, and repeatly
                //calculate the maximum profit.
                minV = min(minV, prices[j] - dp[i-1][j-1]);
                dp[i][j] = max(prices[j] - minV, dp[i][j-1]);
            }
        }
        return dp[k][len-1];
    }
};



//Most optimized solution, reduce the space complexity to linear.
//Hard to get and implement
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int len = prices.size();
        if(len < 2) return 0;
        
        int maxVal = 0;
        if(k >= len/2){
            for(int i = 1; i < len; i++){
                maxVal += max(0, prices[i] - prices[i-1]);
            }
            return maxVal;
        }

        vector<int> dp(k+1, 0);
        vector<int> minV(k+1, prices[0]);
        for(int i = 1; i < len; i++){
            for(int j = 1; j <= k; j++){
                //Tricky here, note we have minv[] to keep track of previous value
                minV[j] = min(minV[j], prices[i] - dp[j-1]);
                dp[j] = max(prices[i] - minV[j], dp[j]);
            }
        }
        return dp[k];
    }
};