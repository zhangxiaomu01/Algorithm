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
            //Note prices[i] - sell1 denotes the max profit we have
            //although its value is negative
            buy2 = min(buy2, prices[i] - sell1);
            sell2 = max(sell2, prices[i] - buy2);

            //This is essentially the max profit for day 0 to i
            buy1 = min(buy1, prices[i]);
            sell1 = max(sell1, prices[i] - buy1);
        }
        return sell2;
    }
};