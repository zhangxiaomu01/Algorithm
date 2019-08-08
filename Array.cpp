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
of h-index. O(n)*/
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

//275. H-Index II
//https://leetcode.com/problems/h-index-ii/
/*The key insight here is to find the minimum index that satisfy 
citations[index] >= citations.size() - index. Then our result will
be citations.size() - index. The general binary search does not work
here, pay attention to how to handle corner case.*/
class Solution {
public:
    int hIndex(vector<int>& citations) {
        int len = citations.size();
        int l = 0, count = citations.size();
        int step = 0;
        while(count > 0){
            step = count / 2;
            //We initialize mid every time, so it will be either larger than
            //the previous mid or smaller than previous mid depends on l
            int mid = l + step;
            if(citations[mid] < len - mid){
                //note l will be in the second half if we first go here
                l = mid + 1;
                count = count - (step + 1);
            }
            else
                count = step;
        }
        return len - l;
    }
};

//55. Jump Game
//https://leetcode.com/problems/jump-game/
/* A general approach is to use dynamic programming, and keep track of each entry
i with dp[i] to avoid repetitive calculation. However, using greedy algorithm and 
always keep track of the furthest potential position we can get for each entry, and
we can terminate eralier if we can not proceed further. If the furthest position we 
can reach is beyond the length, then we know we can get there. */
//Greedy
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int furthestPos = 0;
        int len = nums.size();
        for(int i = 0; i < len && i <= furthestPos; ++i){
            furthestPos = max(furthestPos, i + nums[i]);
            if(furthestPos >= len-1)
                return true;
        }
        return false;
    }
};

//DP: not efficient O(n^2)
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int len = nums.size();
        vector<int> dp(len+1, 0);
        dp[len] = 1;
        //we need to handle the situation nums.back() == 0
        dp[len - 1] = 1;
        //We start from i = len - 2
        for(int i = len-2; i >= 0; --i){
            int localMax = nums[i];
            for(int j = 1; j <= localMax && j < len; ++j){
                dp[i] = dp[i] || dp[i+j];
            }
        }
        return dp[0];
    }
};


//45. Jump Game II
//https://leetcode.com/problems/jump-game-ii/
/* In general, we can still use the techniques from Jump Game. 
Still do greedy plus dp. We can use greedy to check whether 
this is a valid sequence, and use dp to get the minimum steps.
However, for this question, we can do better, by optimizing the 
previous dp solution, we can get the minimum jumps in linear time.
*/
class Solution {
public:
    int jump(vector<int>& nums) {
        int len = nums.size();
        if(len == 1) return 0;
        int localFurthestIndex = 0, steps = 0;
        int rightBoundry = 0;
        for(int i = 0; i < len; ++i){
            localFurthestIndex = max(localFurthestIndex, nums[i] + i);
            //We know we reach the end
            if(localFurthestIndex >= len-1) return steps+1;
            //No way to reach the end
            if(i > localFurthestIndex) return 0;
            //Note we do the jump only when we need to. Which means when
            //we reach an index that reaches the right boundry.
            if(i >= rightBoundry){
                rightBoundry = localFurthestIndex;
                steps++;
            }
        }
        return steps;
    }
}; 


//11. Container With Most Water
//https://leetcode.com/problems/container-with-most-water/
/* O(n^2) solution is trivial. Two pointers work perfectly. */
class Solution {
public:
    int maxArea(vector<int>& height) {
        int len = height.size();
        int l = 0, r = len - 1;
        int maxArea = 0;
        while(l < r){
            if(height[l] <= height[r]){
                maxArea = max(maxArea, height[l] * (r - l));
                l++;
            }else{
                maxArea = max(maxArea, height[r] * (r - l));
                r--;
            }
        }
        return maxArea;
    }
};


//403. Frog Jump
//https://leetcode.com/problems/frog-jump/
/* Interesting problem */
/* Similar to DP. The general idea is that we store the potential previous jumps to
stone i. Then when we update the previous steps in stone (i + step - 1) with step -1, stone 
(i + step) with step and stone (i + step + 1) with step +1.
In the end, we only need to check whether umap[stones[len-1]] is empty. If it's empty, then we know
we cannot reach the end. It's not efficient!*/
class Solution {
public:
    bool canCross(vector<int>& stones) {
        int len = stones.size();
        unordered_map<int, unordered_set<int>> uDict;
        //Note for initial stone 0, we only need 0 steps
        uDict.insert({0, {0}});
        int step = 0;
        for(int i = 0; i < len; ++i){
            for(int step : uDict[stones[i]]){
                if(step - 1 > 0) uDict[stones[i] + step - 1].insert(step - 1);
                uDict[stones[i] + step].insert(step);
                uDict[stones[i] + step + 1].insert(step + 1);
            }
        }
        
        return !uDict[stones[len-1]].empty();
    }
};

/* Another DP, efficient */
//Another DP approach, very efficient
class Solution {
private:
    unordered_map<int, bool> dict;
    //pos is the index of the previous stone
    //gap is the maximum potential step from previous two stones
    bool Cross(vector<int>& stones, int pos, int gap){
        //We want to calculate a unique key for a pair of pos and gap
        //The reason why we left shift gap 11 bits is because the pos is between 2 and 1100
        //which means ths range is [2, 2^11], we can safely move step 11 bits left and 
        //leave enough room for pos. Another observation is that gap will not be too large,
        //The maximum valid gap should be 1100 . 
        //Any value larger than 1100 will be discarded.
        //Since each int will have 32 bits, it will be enough for store both the values.
        int key = (gap << 11) | pos;
        
        if(dict.find(key) != dict.end()) return dict[key];
        
        for(int i = pos+1; i < stones.size(); ++i){
            int localGap = stones[i] - stones[pos];
            //our gap is too narrow, that we cannot land on stones i
            if(localGap < gap - 1) continue;
            //If the gap is too large, we cannot move forward any more
            if(localGap > gap + 1) return dict[key] = false;
            //Since we can sucessfully land on i, then check the whether we can land on the 
            //last stone
            if(Cross(stones, i, localGap)) return dict[key] = true;
        }
        
        return pos == stones.size()-1 ? dict[key] = true : dict[key] = false;
    }
public:
    bool canCross(vector<int>& stones) {
        int len = stones.size();
        if(len <= 1) return true;
        return Cross(stones, 0, 0);
    }
};


//53. Maximum Subarray
//https://leetcode.com/problems/maximum-subarray/
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        int sum = nums[0], ans = nums[0];
        for(int i = 1; i < len; ++i){
            //This is the tricky part! if sum is below 0, we can start
            //with the new index
            sum = max(sum + nums[i], nums[i]);
            ans = max(ans, sum);
        }
        return ans;
    }
};


//309. Best Time to Buy and Sell Stock with Cooldown
//https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
//The best explanation: 
//https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/240097/Come-on-in-you-will-not-regret-most-general-java-code-like-all-other-DP-solutions
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        if(len <= 1) return 0;
        //dp[i] means the maximum profit in day i
        //dp[0] means the first day
        vector<int> dp(len+1, 0);
        dp[1] = max(prices[1] - prices[0], 0);
        for(int i = 2; i < len; ++i){
            //represents the rest day (no buy or sell)
            dp[i] = max(dp[i], dp[i-1]);
            for(int j = 0; j < i; ++j){
                int pre = j >= 2 ? dp[j-2] : 0;
                //find the maximum of pre + prices[i] - prices[j]
                dp[i] = max(dp[i], pre + prices[i] - prices[j]);
            }
        }
        return dp[len-1];
    }
};

//optimized version, note how we get rid of the inner loop
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        if(len <= 1) return 0;
        //dp[i] means the maximum profit in day i
        //dp[0] means the first day
        vector<int> dp(len+1, 0);
        dp[1] = max(prices[1] - prices[0], 0);
        int maxDiff = INT_MIN; //dp[j-2] - prices[j]
        maxDiff = max(maxDiff, -prices[0]);
        maxDiff = max(maxDiff, -prices[1]);
        for(int i = 2; i < len; ++i){
            maxDiff = max(maxDiff, dp[i-2] - prices[i]);
            //Equivalent to dp[i] = max(dp[i-1], dp[j-2] + prices[i] - prices[j]);
            dp[i] = max(dp[i-1], maxDiff + prices[i]);
        }
        return dp[len-1];
    }
};


//4. Median of Two Sorted Arrays
//https://leetcode.com/problems/median-of-two-sorted-arrays/
//The idea is based on the algorithm of getting the kth smallest element for two sorted list
//A detailed explanation can be found: 
//https://windliang.cc/2018/07/18/leetCode-4-Median-of-Two-Sorted-Arrays/
//A very very tricky problem! Hate it!
class Solution {
private:
    //note kth elment represents nums[k-1]
    int getKth(vector<int>& n1, int s1, vector<int>& n2, int s2, int k){
        int len1 = n1.size() - s1, len2 = n2.size() - s2;
        //When k is greater than the length of the two arrays, the array will always be n1
        if(len1 > len2) return getKth(n2, s2, n1, s1, k);
        //If n1 reaches the end, n1 will always be shorter
        if(len1 == 0) return n2[s2 + k - 1];
        //base case, that we reaches the kth element
        if(k == 1) return min(n1[s1], n2[s2]);
        
        //Potential kth element
        int i = s1 + min(len1, k/2) - 1;
        int j = s2 + min(len2, k/2) - 1;

        //Discard 
        if(n1[i] > n2[j])
            return getKth(n1, s1, n2, j+1, k - min(len2, k/2));
        else
            return getKth(n1, i+1, n2, s2, k - min(len1, k/2));
        
    }
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        //We eliminate the difference between odd and even length array by the following code
        //For odd length array, the left and right will get the same value
        int left = (m + n + 1) / 2, right = (m + n + 2) / 2;
        //Check corner case
        if(m == 0) return (nums2[left-1] + nums2[right-1])/2.0;
        else if(n == 0) return (nums1[left-1] + nums1[right-1])/2.0;
        
        return (getKth(nums1, 0, nums2, 0, left) + getKth(nums1, 0, nums2, 0, right))/2.0;
    }
};


//42. Trapping Rain Water
//https://leetcode.com/problems/trapping-rain-water/
//The general idea is to use a linear scan. For each entry,
//we compare it with the maximum height from both sides. If
//nums[i] <= maxHeight, we know it can potentially hold 
//maxHeight - nums[i] water for that entry. We scan the array
//and add these potential water together. Note how we handle 
//when we can really add the water to final result/
//two pointers - interesting problem
class Solution {
public:
    int trap(vector<int>& nums) {
        int l = 0, r = nums.size()-1;
        int maxL = 0, maxR = 0;
        int res = 0;
        while(l < r){
            if(nums[l] <= nums[r]){
                if(nums[l] >= maxL){
                    //We update maxL here and move forward
                    maxL = nums[l];
                }else{
                    res+= maxL - nums[l];
                }
                l++;
            }else{
                if(nums[r] >= maxR){
                    maxR = nums[r];
                }else{
                    res += maxR - nums[r];
                }
                r--;
            }
        }
        return res;
    }
};

//724. Find Pivot Index
//https://leetcode.com/problems/find-pivot-index/
/* Prefix sum, nothing fancy */
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        int sum = 0, leftSum = 0;
        for(int n : nums) sum += n;
        for(int i = 0; i < nums.size(); ++i){
            if(leftSum == sum - nums[i] - leftSum) return i;
            leftSum += nums[i];
        }
        return -1;
    }
};

//15. 3Sum
//https://leetcode.com/problems/3sum/
//Sorting and optimization is critical here
//Another tricky part is how we avoid duplicate results
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        int len = nums.size();
        if(len <= 2) return res;
        sort(nums.begin(), nums.end());
        for(int i = 0; i < len; ++i){
            int a = nums[i];
            if(a > 0) break;
            //a == nums[i+1] won't work here, we will
            //miss the case like [-1, -1, 2]
            if(i - 1 >= 0 && a == nums[i-1]) continue;
            int k = len - 1;
            //Check the rest of the pair
            for(int j = i + 1; j < k;){
                int b = nums[j], c = nums[k];
                int localSum = a + b + c;
                if(localSum == 0){
                    res.push_back(vector<int>{a, b, c});
                    while(j < k && b == nums[++j]);
                    while(k > j && c == nums[--k]);
                }
                else if (localSum > 0)
                    --k;
                else
                    ++j;
            }
        }
        return res;
    }
};


//18. 4Sum
//https://leetcode.com/problems/4sum/
/* Very tricky problem. Got stuck for the corner cases for quite a long time!
Double check tomorrow. Pay attention to the generalized k sum approach.*/
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        int len = nums.size();
        vector<vector<int>> res;
        if(len <= 3) return res;
        sort(nums.begin(), nums.end());
        for(int i = 0; i < len - 3;){
            int a = nums[i];
            for(int j = i + 1; j < len - 2;){
                int b = nums[j];
                int q = len - 1;
                for(int p = j+1; p < q;){
                    int c = nums[p], d = nums[q];
                    int localSum = a + b + c + d;
                    if(localSum == target){
                        res.push_back({a, b, c, d});
                        p++;
                        q--;
                        while(p < q && nums[p] == c) p++;
                        while(p < q && nums[q] == d) q--;
                    }
                    else if(localSum > target)
                        q--;
                    else
                        p++;
                }
                ++j;
                //Note current we are at j
                while(j < len - 2 && nums[j-1] == nums[j]) ++j;
            }
            ++i;
            while(i < len-3 && nums[i-1] == nums[i]) ++i;
        }
        return res;
    }
};

//Generalized ksum solution, tricky to implement. 
//Too many corner cases.
class Solution {
    void kSum(vector<int>& nums, vector<vector<int>>& res, vector<int>& tempRes, int t, int pos, int k){
        if(k == 2){
            int p = pos, q = nums.size() - 1;
            while(p < q){
                int localSum = nums[p] + nums[q];
                if(localSum > t){
                    q--;
                }else if(localSum < t)
                    p++;
                else{
                    tempRes.push_back(nums[p++]);
                    tempRes.push_back(nums[q--]);
                    res.push_back(tempRes);
                    tempRes.pop_back();
                    tempRes.pop_back();
                    while(p < q && nums[p] == nums[p-1])p++;
                    while(p < q && nums[q] == nums[q+1])q--;
                }
            }
        }
        else{
            //Note j <= nums.size() - k, instead of <
            for(int j = pos; j <= nums.size() - k; ){
                int sum = 0;
                //prone the result, if sum of nums[j]....nums[j+k-1] is smaller than
                //t, we can terminate here.
                for(int i = 0; i < k; ++i) sum += nums[j+i];
                if(sum > t) break;
                //reset sum
                sum = nums[j];
                //prone the result, if sum of nums[j] + nums[len-k+1] + ..nums[len-1] 
                //is smaller than t, we need to move forward j
                for(int i = nums.size() - 1; i >= nums.size() - k + 1; --i) sum += nums[i];
                if(sum < t) {
                    j++;//Critical, or we potentially go to infinity loop
                    continue;
                }
                
                tempRes.push_back(nums[j]);
                kSum(nums, res, tempRes, t - nums[j], j+1, k-1);
                tempRes.pop_back();
                ++j; //an alternative way is to define ++j in the for() loop body
                //then remove j++ in the if(sum < t), and change nums[j] == nums[j+1]
                //and remove j-1 >= 0 (or we will have duplicates)
                while(j - 1 >= 0 && j <= nums.size() - k && nums[j] == nums[j-1]) j++;
            }
            
        }
    }
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        int len = nums.size();
        vector<vector<int>> res;
        vector<int> tempRes;
        int k = 4;
        if(len <= k-1) return res;
        sort(nums.begin(), nums.end());
        
        kSum(nums, res, tempRes, target, 0, k);
        return res;
    }
};



//412. Fizz Buzz
//https://leetcode.com/problems/fizz-buzz/
/* Do not really understand why this question exists... */
class Solution {
public:
    vector<string> fizzBuzz(int n) {
        string fizz = "Fizz", buzz = "Buzz", fizzbuzz = "FizzBuzz";
        vector<string> res;
        for(int i = 1; i <= n; ++i){
            if(i%3 == 0 && i%5 == 0)
                res.push_back(fizzbuzz);
            else if(i%5 == 0)
                res.push_back(buzz);
            else if(i%3 == 0)
                res.push_back(fizz);
            else
                res.push_back(to_string(i));
        }
        return res;
    }
};

//242. Valid Anagram
//https://leetcode.com/problems/valid-anagram/
class Solution {
public:
    bool isAnagram(string s, string t) {
        int res = 0;
        if(s.size() != t.size()) return false;
        vector<int> dict(256, 0);
        for(int i = 0; i < s.size(); ++i){
            dict[s[i] - '0'] ++;
            dict[t[i] - '0'] --;
        }
        for(int i = 0; i < 256; ++i){
            if(dict[i] != 0)
                return false;
        }
        return true;
    }
};


//238. Product of Array Except Self
//https://leetcode.com/problems/product-of-array-except-self/
//Using two extra array... No division included
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> res(nums.size(), 1);
        int len = nums.size();
        vector<int> fA(len+1, 1), bA(len+1, 1);
        int i = 0, j = len - 1;
        while(i <= len - 1){
            fA[i+1] = fA[i] * nums[i];
            bA[j] = bA[j+1] * nums[j];
            i++;
            j--;
        }
        for(int k = 1; k <= len; ++k){
            res[k-1] = fA[k-1] * bA[k];
        }
        return res;
    }
};

//O(1) space solution
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int len = nums.size();
        //Using the array to store left product, build the 
        //right product on the fly!
        vector<int> res(len, 1);
        int rProduct = 1;
        for(int i = 1; i < len; ++i){
            res[i] = res[i-1] * nums[i-1];
        }
        
        for(int i = len-1; i >= 0; --i){
            res[i] = res[i] * rProduct;
            //update rProduct on the fly
            rProduct *= nums[i];
        }
        return res;
    }
};


//84. Largest Rectangle in Histogram
//https://leetcode.com/problems/largest-rectangle-in-histogram/
//Very tricky problem, the corner case is not easy to catch.
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        if(heights.empty()) return 0;
        int len = heights.size();
        
        //In order to handle the case like [4, 3]
        //We need to make sure that the last element gets properly handled
        heights.push_back(0);
        //heights.insert(heights.begin(), numeric_limits<int>::max());
        stack<int> hSt;
        
        int maxArea = 0;
        // set the left boundry to be infinite
        for(int i = 0; i <= len;){
            if(hSt.empty() || heights[i] >= heights[hSt.top()]){
                //Only update i here
                hSt.push(i++);
            }else{
                while(!hSt.empty() && heights[i] < heights[hSt.top()]){
                    int index = hSt.top();
                    hSt.pop();
                    //Note how we can update this width. When heights[i] < heights[hSt.top()]
                    //We first pop up hSt.top(). And calculate the area based on the former element's index
                    //In the case of [2,1,5,6,2,3], imagine that 
                    //we are at the end of the array,we first pop 3 out of stack, the index will be 5, then we can
                    //do heights[index] * (6 (we push 0 in the end) - 1 - 4 (which is the 5th element)). 
                    //The last thing is if the stack is empty, we know we can include all the elements from
                    //index 0 to index i-1, which means current valid length should be i. then we need to return 
                    //-1 here to counter the effect of i-1. Very tricky statement
                    maxArea = max(maxArea, (i - 1 - (hSt.empty()? -1 : hSt.top())) * heights[index]);
                }
            }
        }
        return maxArea;
    }
};

/* Second Solution */
//This solution is based on the solution of problem 84. The general idea is that we will
//first calculate the height of each entry row by row. Then for each row, we call the 
//algorthm to calculate the largest triangle in our defined histogram. We maintain a 
//maxArea variable to keep track of the maximum area we could potentially have.
class Solution {
private:
    int calMaxArea(vector<int>& V){
        stack<int> hSt;
        int len = V.size() - 1;
        int maxArea = 0;
        for(int i = 0; i <= len;){
            if(hSt.empty() || V[hSt.top()] <= V[i])
                hSt.push(i++);
            else{
                while(!hSt.empty() && V[hSt.top()] > V[i]){
                    int index = hSt.top();
                    hSt.pop();
                    maxArea = max(maxArea, (i - 1 - (hSt.empty() ? -1 : hSt.top())) * V[index]);
                }
            }
        }
        return maxArea;
    }
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        if(matrix.empty()) return 0;
        int m = matrix.size(), n = m ? matrix[0].size() : 0;
        vector<int> height(n+1, 0);
        int maxArea = 0;
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j){
                if(matrix[i][j] == '1'){
                    height[j]++;
                }            
                else{
                    height[j] = 0;
                }
            }
            int localMax = calMaxArea(height);

            maxArea = max(maxArea, localMax);
        }
        return maxArea;
    }
};


//334. Increasing Triplet Subsequence
//https://leetcode.com/problems/increasing-triplet-subsequence/
class Solution {
public:
    bool increasingTriplet(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return false;
        vector<int> dict;
        for(int i = 0; i < len; ++i){
            if(i == 0 || nums[i] > dict.back()){
                dict.push_back(nums[i]);
                if(dict.size() == 3) return true;
            }
            else if(nums[i] <= dict.back()){
                for(int j = 0; j < dict.size(); ++j){
                    if(dict[j] >= nums[i]) {
                        dict[j] = nums[i];
                        //Once we update the value, we need to 
                        //terminate the loop
                        break;
                    }
                }
            }   
        }
        return false;
    }
};

//128. Longest Consecutive Sequence
//https://leetcode.com/problems/longest-consecutive-sequence/
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> dict(nums.begin(), nums.end());
        int maxLen = 0;
        for(int n : nums){
            //We always search from the smallest consecutive element
            if(dict.find(n-1) == dict.end()){
                int curLen = 1;
                int temp = n;
                //We need to erase element in order to prevent repetitive
                //check
                while(dict.count(temp+1) != 0){
                    dict.erase(temp);
                    temp++;
                    curLen++;
                }
                maxLen = max(maxLen, curLen);
            }
        }
        return maxLen;
    }
};


//287. Find the Duplicate Number
//https://leetcode.com/problems/find-the-duplicate-number/
/* This solution sort the array, we try to put each number i to the corresponding
entry i-1. In the end, if we find some element who is not equal to i+1 and 
nums[nums[i]-1] entry (the correct entry for this element) has the element equal 
to nums[i]. Then we know this must be the extra duplicate element! */
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int len = nums.size();
        for(int i = 0; i < len; ++i){
            //If nums[nums[i]] is already 
            if(i+1 != nums[i]){
                while(i + 1 != nums[i]  && nums[nums[i]-1] != nums[i]) {
                    swap(nums[nums[i]-1], nums[i]);
                }
                //if num i+1 is in index i, then skip it
                if(nums[nums[i]-1] != i+1 && nums[nums[i]-1] == nums[i]){
                    return nums[i]; 
                }                   
            }
        }
        return -1;
    }
};

//Cycle detection! Similar to problem 142
//Since the length of the array is n+1, our pointer will never
//exceed the boundry!
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int fastPtr = 0, slowPtr = 0;
        do{
            fastPtr = nums[nums[fastPtr]];
            slowPtr = nums[slowPtr];
        }while(fastPtr != slowPtr);
        
        //Once the pointer meet with each other, we need to
        //iterate through the begining of the array, and find
        //the duplicate element!
        int ptr = 0;
        while(ptr != fastPtr){
            ptr = nums[ptr];
            fastPtr = nums[fastPtr];
        }
        return ptr;
    }
};


//239. Sliding Window Maximum
//https://leetcode.com/problems/sliding-window-maximum/
/* multiset */
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        multiset<int> mSet;
        int len = nums.size();
        vector<int> res;
        
        for(int i = 0; i < len; ++i){
            mSet.insert(nums[i]);
            if(i >= k-1){
                res.push_back(*mSet.rbegin());
                //Remove the left element
                //We cannot use mSet.erase(nums[i-k+1])
                //It will remove all the duplicate elements
                mSet.erase(mSet.find(nums[i-k+1]));
            }
               
        }
        return res;
    }
};

/* priority queue */
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int len = nums.size();
        if(len == 0) return vector<int>();
        vector<int> res;
        //must use pair, in order to sort the pq based on value
        priority_queue<pair<int,int>> pq;
        for(int i = 0; i < len; ++i){
            while(!pq.empty() && pq.top().second <= i - k)
                pq.pop();
            pq.push(make_pair(nums[i], i));
            //imagine k == 3, then when we visit the second element
            //we need to push it to result
            if(i >= k-1){
                res.push_back(pq.top().first);
            }
        }
        return res;
    }
};

/* O(n) */
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        if(nums.empty()) return res;
        int len = nums.size();
        //in this dq, we only save the largest element's index 
        //within the given range. We pop out all the other elements
        deque<int> dq;
        for(int i = 0; i < len; ++i){
            //if the front elements are out of range, delete it
            while(!dq.empty() && dq.front() <= i - k)
                dq.pop_front();
            //Make sure we only save the largest element's index
            while(!dq.empty() && nums[dq.back()] < nums[i])
                dq.pop_back();
            
            dq.push_back(i);
            
            if(i >= k-1)
                res.push_back(nums[dq.front()]);
        }
        return res;
    }
};


//53. Maximum Subarray
//https://leetcode.com/problems/maximum-subarray/
/* Kadane's algorithm */
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if(nums.empty()) return 0;
        int len = nums.size();
        int localMax = 0, res = INT_MIN;
        for(int i = 0; i < len; ++i){
            localMax = max(nums[i], localMax + nums[i]);
            res = max(res, localMax);
        }
        return res;
    }
};


//209. Minimum Size Subarray Sum
//https://leetcode.com/problems/minimum-size-subarray-sum/
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        int count = 0;
        int minC = INT_MAX;
        int sum = 0;
        int j = 0;
        for(int i = 0; i < len; ++i){
            sum += nums[i];
            count++;
            while(sum >= s){
                minC = min(minC, count);
                sum -= nums[j++];
                count --;
            }
        }
        return minC == INT_MAX ? 0 : minC;
    }
};


//228. Summary Ranges
//https://leetcode.com/problems/summary-ranges/
class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums) {
        int len = nums.size();
        vector<string> res;
        if(len == 0) return res;
        if(len == 1) {
            res.push_back(to_string(nums[0]));
            return res;
        }
        
        int j = 0;
        int count = 0;
        string s = to_string(nums[0]);
        for(int i = 1; i < len; ++i){
            if(nums[j] + 1 == nums[i]){
                count++;
            }
            else{
                if(count > 0){
                    s.push_back('-');
                    s.push_back('>');
                    s.append(to_string(nums[j]));
                }
                count = 0;
                res.push_back(s);
                s = to_string(nums[i]);
                if(i == len-1){
                    res.push_back(s);
                    break;
                }
            }
            j++; 
        }
        if(count > 0){
            s.push_back('-');
            s.push_back('>');
            //we have advanced one more time
            s.append(to_string(nums[j]));
            res.push_back(s);
        }
        
        return res;
    }
};


//283. Move Zeroes
//https://leetcode.com/problems/move-zeroes/
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int len = nums.size();
        int j = 0;
        for(int i = 0; i < len && j < len;){
            if(nums[i] != 0){
                j++;
                i++;
            }else{
                j++;
                while(j < len && nums[j] == 0)
                    j++;
                if(j == len) return;
                swap(nums[i], nums[j]);
                i++;
            }
        }
    }
};


//373. Find K Pairs with Smallest Sums
//https://leetcode.com/problems/find-k-pairs-with-smallest-sums/
/* Priority queue solution: O(MNlogK)*/
class myComp{
public:
    bool operator()(const vector<int>& A, const vector<int>& B) const
    {
        return A[0] + A[1] < B[0] + B[1];
    }
};

class Solution {
public:
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<vector<int>> res;
        int len1 = nums1.size(), len2 = nums2.size();
        if(len1 == 0 || len2 == 0 || k == 0) return res;
        priority_queue<vector<int>, vector<vector<int>>, myComp> maxPQ;
        //small optimization, if len >= k, then k+1th element will not in the final result
        //maxPQ will always has size of no more than k
        for(int i = 0; i < min(len1, k); ++i){
            for(int j = 0; j < min(len2, k); ++j){
                if(maxPQ.size() < k)
                    maxPQ.push(vector<int>({nums1[i], nums2[j]}));
                else if (nums1[i] + nums2[j] < maxPQ.top()[0] + maxPQ.top()[1]){
                    maxPQ.push(vector<int>({nums1[i], nums2[j]}));
                    maxPQ.pop();
                }
            }
        }
        
        while(!maxPQ.empty()){
            auto& v = maxPQ.top();
            res.push_back(v);
            maxPQ.pop();
        }
        return res;
    }
};


/* Optimized priority queue solution. Tricky!
Note for the first pass, we push the {nums1[i], nums2[0], 0} to our min queue.
Then in the second pass, we pop the smallest element {nums1[i], nums2[j], j} and push it to result.
We also update the next smallest element from the two lists by pushing {nums1[i], nums2[j+1], j+1}
to the min queue (j+1 < min(len2, k)). This does not mean this is the smallest element in the queue,
but this must be the next smallest element after {nums1[i], nums2[j], j}
*/
class myComp{
public:
    bool operator()(const vector<int>& A, const vector<int>& B) const
    {
        return A[0] + A[1] > B[0] + B[1];
    }
};

class Solution {
public:
    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<vector<int>> res;
        int len1 = nums1.size(), len2 = nums2.size();
        if(len1 == 0 || len2 == 0 || k == 0) return res;
        priority_queue<vector<int>, vector<vector<int>>, myComp> minPQ;

        for(int i = 0; i < min(len1, k); ++i){
            //the last element is the index of element from nums2
            minPQ.push(vector<int>({nums1[i], nums2[0], 0}));
        }
        
        while(res.size() < k && !minPQ.empty()){
            auto& v = minPQ.top();
            res.push_back({v[0], v[1]});
            //Be careful with corner case!
            if(v[2] < min(len2, k)-1){
                int index = v[2] + 1;
                minPQ.push(vector<int>({v[0], nums2[index], index}));
            }
            minPQ.pop();   
        }
        
        return res;
    }
};


//164. Maximum Gap
//https://leetcode.com/problems/maximum-gap/
/* sorting is trivial */
class Solution {
public:
    int maximumGap(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int gap = 0;
        for(int i = 1; i < nums.size(); ++i){
            gap = max(gap,nums[i] - nums[i-1]);
        }
        return gap;
    }
};


/* Bucket sort solution. Interesting */
//Good explanation:
//https://leetcode.com/problems/maximum-gap/discuss/50694/12ms-C%2B%2B-Suggested-Solution
class Solution {
public:
    int maximumGap(vector<int>& nums) {
        int len = nums.size();
        if(len <= 1) return 0;
        auto minMax = minmax_element(nums.begin(), nums.end());
        //std::pair<ForwardIt,ForwardIt> 
        int lo = *minMax.first, hi = *minMax.second;
        //We need to gurantee that gap should be at least 1
        int gap = max((hi - lo) / (len - 1), 1);
        //m represents the num of buckets
        int m = (hi - lo) / gap + 1;
        
        //record the min and max element for each bucket
        //The max potential gap is from difference between the 
        //min element from i+1 bucket - max element from i bucket
        vector<int> bucketMin(m, INT_MAX);
        vector<int> bucketMax(m, INT_MIN);
        
        for(int i = 0; i < len; ++i){
            //map each element to corresponding bucket
            int k = (nums[i] - lo) / gap;
            bucketMax[k] = (bucketMax[k] < nums[i]) ? nums[i] : bucketMax[k];
            bucketMin[k] = (bucketMin[k] > nums[i]) ? nums[i] : bucketMin[k];
        }
        
        int maxGap = 0;
        int i = 0, j = 0;
        while(j < m){
            while(j < m && bucketMax[j] == INT_MIN && bucketMin[j] == INT_MAX)
                j++;
            if(j == m) break;
            
            int localGap = bucketMax[j] - bucketMin[j];
            maxGap = max(maxGap, localGap);
            maxGap = max(maxGap, bucketMin[j] - bucketMax[i]);
            //Move the i points to current j
            i = j;
            j++;
        }
        return maxGap;
    }
};


//268. Missing Number
//https://leetcode.com/problems/missing-number/
/* Using sorting or unordered_set is trivial, ignore here.
Unordered_set solution: put all elements in the unordered set,
query each entry element n whether exists n-1 or n+1, if not
exists, we find it. */
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int res = 0;
        //we have n element in total and all distinctive
        //from 0, 1, 2, ... , n. Then the total sum of 
        //will be (n+1)*n /2, then we can substract each 
        //element from the array, and get the final result.
        for(int i = 0; i < nums.size(); ++i){
            //we substract nums[i] here to avoid overflow
            res += (i+1 - nums[i]);
        }
        return res;
    }
};


class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int res = 0;
        for(int i = 0; i < nums.size(); ++i){
            //We xor each number with the corresponding index,
            //in the end, since we have n element, we can eliminate
            //thouse we have the number and corresponding index one
            res ^= (i+1) ^ nums[i];
        }
        return res;
    }
};


//88. Merge Sorted Array
//https://leetcode.com/problems/merge-sorted-array/
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        while(n)
            nums1[m+n-1] = (m > 0 && nums1[m-1] > nums2[n-1])? nums1[--m] : nums2[--n];
    }
};


//75. Sort Colors
//https://leetcode.com/problems/sort-colors/
/* 3 pointers solution */
class Solution {
public:
    void sortColors(vector<int>& nums) {
        //j represents the end of the red (0)
        //k represents the start of the begin (2)
        int j = 0, k = nums.size() - 1;
        for(int i = 0; i <= k; ++i){
            //Increment j when we swap 0 to the start
            //of the array
            if(nums[i] == 0)
                swap(nums[i], nums[j++]);
            //We need to decrease i because we need to 
            //double check kth element
            else if (nums[i] == 2)
                swap(nums[i--], nums[k--]);
        }
    }
};

//324. Wiggle Sort II
//https://leetcode.com/problems/wiggle-sort-ii/
/* Sorting and rearrange elements. The key insight is to put the smallest 
elements to even index, while put the other half to odd index */
class Solution {
public:
    void wiggleSort(vector<int>& nums) {
        vector<int> sorted(nums);
        sort(sorted.begin(), sorted.end());
        for(int i = nums.size()-1, j = 0, k = i/2 + 1; i >= 0; --i){
            nums[i] = sorted[(i&1) ? k++ : j++];
        }
    }
};

/* This sorting implementation is too tricky, almost useless */
class Solution {
public:
    void wiggleSort(vector<int>& nums) {
        vector<int> sorted(nums);
        //sort the array from large to small number
        sort(sorted.rbegin(), sorted.rend());
        int len = nums.size();
        // if len is even, change it to odd
        int mod = len | 1; 
        for(int i = 0; i < len; ++i){
            nums[(2 * i + 1) % mod] = sorted[i];
        }
    }
};

/* O(n), O(1) solution */
/* O(n) time, O(1) space solution.
Impossible to get, I cannot even prove it is right.
I cannot really understand it.
https://leetcode.com/problems/wiggle-sort-ii/discuss/77681/O(n)-time-O(1)-space-solution-with-detail-explanations
*/
class Solution {
public:
    void wiggleSort(vector<int>& nums) {
        int len = nums.size();
        
        //find the median of the array - O(n)
        auto midIt = nums.begin() + len/2;
        nth_element(nums.begin(), midIt, nums.end());
        int median = *midIt;
        
        //Remap the original index to target index
        auto m = [len](int index){ return (index*2 + 1) % (len | 1);};
        
        //Three partition method for array.We will put the 
        //elements larger than median at the even index start from
        //the front of the array. Those smaller than median at the
        //odd index start from the end of the array. Elements equal 
        //to median put in the remaining slots.
        int i = 0, mid = 0, j = len - 1;
        while(mid <= j){
            if(nums[m(mid)] > median){
                swap(nums[m(i)], nums[m(mid)]);
                i++;
                mid++;
            }else if(nums[m(mid)] < median){
                swap(nums[m(j)], nums[m(mid)]);
                j--;
            }else
                mid++;
        }
    }
};


//376. Wiggle Subsequence
//https://leetcode.com/problems/wiggle-subsequence/
/* Actually, we can start from Brute force algorithm. We keep track of whether
current nums[i] forms an ascending order sequence or descending order sequence.
Then we recursively check the maximum length from the rest of sequence. The time
complexity is O(n!) */


/* DP solution. We maintain two sequences up[i], down[i]. When we reach the index i,
if(nums[i] > nums[j]) then up[i] should be max(up[i], down[j] + 1). The code is 
pretty straightforward. O(n^2)*/
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int len = nums.size();
        if(len < 2) return len;
        vector<int> up(len, 0), down(len, 0);
        //The base case: when we have only 1 element
        up[0] = down[0] = 1;
        for(int i = 1; i < len; ++i){
            for(int j = 0; j < i; ++j){
                if(nums[i] > nums[j]){
                    //Since the last index i is in ascending order,
                    //we need to find the previous descending sequence
                    //and update up[i]
                    up[i] = max(up[i], down[j] + 1);
                }else if(nums[i] < nums[j]){
                    //The same idea here. 
                    down[i] = max(down[i], up[j] + 1);
                }else{
                    //since nums[i] == nums[j], feel free to move to 
                    //skip this index
                    up[i] = max(up[i], up[j]);
                    down[i] = max(down[i], down[j]);
                }
            }
        }
        return max(up[len-1], down[len-1]);
    }
};

/* Optimized DP. Get rid of the inner loop. The reason */
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int len = nums.size();
        if(len < 2) return len;
        vector<int> up(len, 0), down(len, 0);
        //The base case: when we have only 1 element
        up[0] = down[0] = 1;
        for(int i = 1; i < len; ++i){
            if(nums[i] > nums[i-1]){
                up[i] = max(up[i], down[i-1] + 1);
                //update down[i] here to be down[i-1]
                down[i] = down[i-1];
            }else if(nums[i] < nums[i-1]){
                //The same idea here. 
                down[i] = max(down[i], up[i-1] + 1);
                up[i] = up[i-1];
            }else{
                //since nums[i] == nums[j], feel free to move to 
                //skip this index
                up[i] = up[i-1];
                down[i] = down[i-1];
            }
        }
        return max(up[len-1], down[len-1]);
    }
};


/* Optimized DP. Note that when we update up[i] or down[i], we only need down[i-1]
or up[i-1]. So we do not need an array for this problem, we can just maintain two
variables. */






