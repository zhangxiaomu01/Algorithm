#include<windows.h>
#include<algorithm>
#include<vector>
#include<array>
#include<cmath>
#include<random>
#include<sstream>
#include<unordered_map>
#include<numeric>
#include<iterator>

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
//Hard to get and implement. Read carefully
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


/**************************************************************
The following problems are related to random number generation. 
Some also related to generate specific distribution.
**************************************************************/

//Sample Offline data
/*
Implement an algorith that takes as input an array of distinct elements and size, and returns a subset of the given size of the array elements. All subsets should be equally likely. Return the result in input array itself.
*/
void randomSampling(int k, std::vector<int>& A)
{
	int len = A.size();
	if (k <= 0 || len <= k) return;
	std::default_random_engine seed((std::random_device())());
	for (int i = 0; i < k; i++) {
		std::uniform_int_distribution<int> distribution(i, static_cast<int>(len) - 1);
		int num = distribution(seed);
		std::swap(A[i], A[num]);
	}
	for (int i = 0; i < k; i++)
		std::cout << A[i] << " ";

	std::cout << std::endl;
}

//Sample Online data
/*
Design a program that takes as input of a size k, and reads packets, continuously maintaining a uniform random subset of size k of the read packets.
*/
std::vector<int> randomSamplingStream(std::istringstream *s, int k)
{
	int x;
	std::vector<int> res;
	for (int i = 0; i < k && *s >> x; i++) {
		res.push_back(x);
	}
	std::default_random_engine seed((std::random_device())());
	int num_sofar = k;
	while (*s >> x) {
		num_sofar++;
		const int index = std::uniform_int_distribution<int>{ 0, num_sofar - 1}(seed);
		if (index < res.size())
			res[index] = x;
	}
	return res;
}

//Compute a random subset
/*
Write a program that takes as input a positive integer n and a size k <= n, and returns a size k subset of [0, n-1]. The subset should be represented as an array, all subsets should be equally likely and, in addition, all permutations of elements of the array should be equally likely.
*/
//n is the total range, k is the subset of the range
std::vector<int> randomSubset(int n, int k)
{
	std::vector<int> res;
	std::unordered_map<int, int> map;
	std::default_random_engine seed((std::random_device())());
	for (int i = 0; i < k; i++) {
		//int index = std::uniform_int_distribution<int>{i, n-1}(seed);
		std::uniform_int_distribution<int> temp(i, n - 1);
		int index = temp(seed);
		auto p1 = map.find(i);
		auto p2 = map.find(index);
		if (p1 == map.end() && p2 == map.end()) {
			map[i] = index;
			map[index] = i;
		}
		else if (p1 != map.end() && p2 == map.end()) {
			map[index] = p1->second;
			p1->second = index;
		}
		else if (p1 == map.end() && p2 != map.end()) {
			map[i] = p2->second;
			p2->second = i;
		}
		else {
			int temp = p1->second;
			p1->second = p2->second;
			p2->second = temp;
		}
	}

	for (int i = 0; i < k; i++) {
		res.emplace_back(map[i]);
		std::cout << res[i] << " ";
	}
	std::cout << '\n';
	return res;
}


//Generate nonuniform random numbers
/*
Given a random number generator which produces values in [0,1] uniformly (floating number), how could you generate one of the n numers according to the specific probability? say [3, 5, 7, 11], we have coresponding probability [9/18, 6/18, 2/18, 1/18].
*/
int nonuniformRandomNumberGenerator(std::vector<int> nums, std::vector<double> p)
{
	if (p.size() != nums.size()) return -1;
	
	std::vector<double> prefixSumP;
	prefixSumP.emplace_back(0.0);
	std::partial_sum(p.cbegin(), p.cend(), std::back_inserter(prefixSumP));

	std::default_random_engine seed((std::random_device())());
	const double uniform_0_1 = std::generate_canonical<double, std::numeric_limits<double>::digits>(seed);

	const int index = std::distance(prefixSumP.cbegin(), std::upper_bound(prefixSumP.cbegin(), prefixSumP.cend(), uniform_0_1));

	return nums[index];
}

//217. Contains Duplicate
//https://leetcode.com/problems/contains-duplicate/
/* Sorting plus linear scan! */
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        if(nums.size() <= 1) return false; 
        sort(nums.begin(), nums.end());
        int len = nums.size();
        for(int i = 0; i < len - 1; ++i){
            if(nums[i] == nums[i+1])
                return true;
        }
        return false;
    }
};



