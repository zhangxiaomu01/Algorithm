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
/* Unordered set, keep track of explored elements */
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_set<int> dict;
        for(int i = 0; i < nums.size(); ++i){
            if(dict.count(nums[i])!=0)
                return true;
            dict.insert(nums[i]);
        }
        return false;
    }
};

/* We have one bit mask solution. Allocate an array of integers, 
and each bit in the integer represents one number from [0, INT_MAX].
Whenever we want to examine the number, we calculate the corresponding
entry by nums[i]/32, then for this entry, we have nums[i]%32 to get the
actual bit. Too fancy to implment! Ignore here*/


//219. Contains Duplicate II
//https://leetcode.com/problems/contains-duplicate-ii/
//The problem definition is a little bit vague! Note we are trying to find one 
//of the potential pairs that meets the demand. If there exists one, we should
//return true!
class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        unordered_map<int, int> dict;
        int len = nums.size();
        for(int i = 0; i < len; ++i){
            if(dict.find(nums[i]) != dict.end() && i - dict[nums[i]] <= k){
                return true;
            }
            dict[nums[i]] = i;
        }
        return false;
    }
};


//220. Contains Duplicate III
//https://leetcode.com/problems/contains-duplicate-iii/
//Unordered set using sliding window! insert() method will return a pair.
//The second argument will be a boolean which indicates whether we have a
//repetitive element or not
class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        int w = 0;
        int len = nums.size();
        unordered_set<int> uset;
        for(int i = 0; i < len; ++i){
            if(i - w > k){
                //Note we need to increase w when we erase the element
                uset.erase(nums[w++]);
            }
            auto r = uset.insert(nums[i]);
            if(!r.second) return true;
        }
        return false;
    }
};

//The general idea is that we maintain a set of size k, then we slide the set
//from left to right. Since we have a set, we can easily find the lowerbound 
//of the set. 
class Solution {
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        int len = nums.size();
        if(len == 0) return false;
        int w = 0;
        //Using a set instead of an unordered_set, because we need to get
        //the range of the sliding window
        set<long> dict;
        for(int i = 0; i < len; ++i){
            if(i - w > k){
                dict.erase(nums[w++]);
            }
            //If the new value - any value from the window is less than or equal to t, then
            // |x - nums[i]| <= t  ==> -t <= x - nums[i] <= t;
            // x-nums[i] >= -t ==> x >= nums[i]-t, Calculate the lower bound from the current window
            auto it = dict.lower_bound(static_cast<long>(nums[i]) - t);
            //Since we already know the potential minimum valid x from the window, then if we want to
            //calclucate the upper side. (it is possible that x is at the end of the window, which is 
            //dict.end())
            // x - nums[i] <= t ==> |x - nums[i]| <= t
            if(it != dict.end() && *it - nums[i] <= t){
                return true;
            }
            dict.insert(nums[i]);
        }
        return false;
    }
};


//56. Merge Intervals
//https://leetcode.com/problems/merge-intervals/
//The key insight is to sort the array based on the first element.
//Then we maintain a vector and if we find that res.back()[1] < intervals[i][0],
//we push back the intervals[i][0]
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> res;
        int len = intervals.size();
        if(len == 0) return res;
        sort(intervals.begin(), intervals.end(), [](vector<int>& a, vector<int>& b){ return a[0] < b[0]; });
        res.push_back(intervals[0]);
        for(int i = 1; i < len; ++i){
            if(res.back()[1] < intervals[i][0]) 
                res.push_back(intervals[i]);
            else{
                res.back()[1] = max(res.back()[1], intervals[i][1]);
            }
        }
        return res;
    }
};

//27. Remove Element
//https://leetcode.com/problems/remove-element/
//Not hard
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int len_l = nums.size();
        int k = 0;
        for(int i = 0; i<len_l; i++){
            if(nums[i] != val){
                nums[k] = nums[i];
                k++;
            }      
        }
        return k;
    }
};
//Different Implementation
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int len = nums.size();
        for(int i = 0; i < len; ++i){
            while(nums[len-1] == val){
                len--;
                
                if(len == 0)
                    return 0;
                if(len == i){
                    return len;
                }  
            }
            if(nums[i] == val){
                swap(nums[i], nums[len-1]);
            }
        }
        return len;
    }
};


//26. Remove Duplicates from Sorted Array
//https://leetcode.com/problems/remove-duplicates-from-sorted-array/
/* Not hard */
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int len = nums.size();
        int k = 0;
        for(int i = 0; i < len; ++i){
            while(i < len-1 && nums[i] == nums[i+1])
                i++;
            nums[k] = nums[i];
            k++;
        }
        return k;
    }
};


//80. Remove Duplicates from Sorted Array II
//https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
/* Intresting idea */
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int i = 0;
        for(int n:nums){
            if(i < 2 || n > nums[i-2])
                nums[i++] = n;
        }
        return i;
    }
};

//189. Rotate Array
//https://leetcode.com/problems/rotate-array/
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int len = nums.size();
        reverse(nums.begin(), nums.end());
        //Note the reverse is [i,j)
        int m = k%len;
        reverse(nums.begin(), nums.begin()+m);
        reverse(nums.begin()+m, nums.end());
    }
};

//41. First Missing Positive
//https://leetcode.com/problems/first-missing-positive/
/* A very tricky problem. The general idea is that we swap the nums[i] to the 
corresponding index nums[i]-1 (if nums[i] is greater than 0 and less than the array length). 
For example, 1 will finally be swapped to index 0, so on and so forth.
Since each element will only be potentially swapped once, and we need to iterate the array
from left to right, which means that we will visit each element at most twice.
Note:negative numbers will be handled correctly because potentially they will be swapped to 
some index that cannot have the correspoding nums[i] in it.
*/
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int len = nums.size();
        for(int i = 0; i < len; ++i){
            //We swap nums[i] to the potential index, we need to have nums[nums[i]-1] != nums[i]
            //to prevent a loop
            while(nums[i] > 0 && nums[i] <= len && nums[nums[i]-1] != nums[i])
                swap(nums[nums[i]-1], nums[i]);
        }
        
        //Once we swapped the potential element to right index, we know if nums[i] == i+1, then
        //we know the i+1 exists here, and we move forward, until we find the first nums[i], that
        //nums[i] != i+1, then i+1 must be the first positive gap
        for(int i = 0; i < len; ++i){
            if(nums[i] != i+1)
                return i+1;
        }
        //if we do not terminate earlier, we know the positive number should be len+1
        return len+1;
    }
};


//299. Bulls and Cows
//https://leetcode.com/problems/bulls-and-cows/
//Hash implementation, too costy
class Solution {
public:
    string getHint(string secret, string guess) {
        unordered_map<char, int> umap;
        int len = secret.size();
        string res = "";
        int countA = 0, countB = 0;
        for(int i = 0; i < len; ++i){
            if(secret[i] == guess[i]){
                countA++;
            }
            else{
                if(umap.find(secret[i]) == umap.end()){
                    umap[secret[i]] = 1;
                }else{
                    ++umap[secret[i]];
                }
            } 
        }
        for(int i = 0; i < len; ++i){
            if(secret[i] != guess[i]){
                if(umap.find(guess[i]) != umap.end()){
                    umap[guess[i]]--;
                    countB++;
                    if(umap[guess[i]] == 0)
                        umap.erase(guess[i]);
                }
            }
        }
        res = to_string(countA) + 'A' + to_string(countB) + 'B';
        return res;
    }
};

//Optimized version
class Solution {
public:
    string getHint(string secret, string guess) {
        int len = secret.size();
        int countA = 0, countB = 0;
        vector<int> dict(10, 0); // store '0' - '9'
        for(int i = 0; i < len; ++i){
            if(secret[i] == guess[i])
                countA++;
            else{
                //Note if(i++), then i = i+1 will only happens at the end of the if statement
                //even if we have if((i++)>1), we still have the check if(i > 1), then at the 
                //end, we have i++. Tricky here.
                if(dict[secret[i] - '0'] ++ < 0){
                    countB++;
                }
                    
                if(dict[guess[i] - '0'] -- > 0)
                    countB++;
            }
        }
        return to_string(countA) + 'A' + to_string(countB) + 'B';;
    }
};


//134. Gas Station
//https://leetcode.com/problems/gas-station/
/* An interesting problem, the key insight is that if the total gas - total cost is negative,
we will never have a solution. At the same time, if we are able to drive from i to i+1, then 
our local tank cannot be negative. If the last station fails, we have to try the next station (i+1)*/
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int remaining = 0, total = 0, tank = 0;
        int start = 0;
        int len = gas.size();
        for(int i = 0; i < len; i++){
            remaining = gas[i] - cost[i];
            tank += remaining;
            total += remaining;
            if(tank < 0){
                start = i+1;
                tank = 0;
            }
        }
        return total < 0 ? -1:start;
    }
};

//274. H-Index
//https://leetcode.com/problems/h-index/
/* The general idea is we first allocate an array to record how many papers have
ciatation i. i represents the entry for this array. Then we iterate through the 
array in descending order and calculate the total number of papers which has 
citation above i, whenever we find that papers >= i, this must be the maximum boundary 
of h-index.*/
class Solution {
public:
    int hIndex(vector<int>& citations) {
        int len = citations.size();
        //dict records how many papers have citation i
        //i is the index of the entry
        vector<int> dict(len+1, 0);
        for(int i = 0; i < len; ++i){
            if(citations[i] >=len)
                dict[len]++;
            else
                dict[citations[i]]++;
        }
        int papers = 0;
        for(int i = len; i >= 0; --i){
            papers += dict[i];
            if(papers >= i)
                return i;
        }
        return 1;
    }
};


