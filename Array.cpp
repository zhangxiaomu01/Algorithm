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
Write a program that takes as input a positive integer n and a size k <= n, and 
returns a size k subset of [0, n-1]. The subset should be represented as an array,
all subsets should be equally likely and, in addition, all permutations of 
elements of the array should be equally likely.
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
Given a random number generator which produces values in [0,1] uniformly 
(floating number), how could you generate one of the n numers according 
to the specific probability? say [3, 5, 7, 11], we have coresponding probability 
[9/18, 6/18, 2/18, 1/18].
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

//Divid and conquer approach!
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        return maxSubArray(nums, 0, nums.size() - 1);
    }
private:
    int maxSubArray(vector<int>& nums, int l, int r) {
        if (l > r) {
            return INT_MIN;
        }
        int m = l + (r - l) / 2, ml = 0, mr = 0;
        //max subarray sum appears to be in left half
        int lmax = maxSubArray(nums, l, m - 1);
        //max subarray sum appears to be in right half
        int rmax = maxSubArray(nums, m + 1, r);
        
        //max subarray sum appears to be in both halves
        for (int i = m - 1, sum = 0; i >= l; i--) {
            sum += nums[i];
            ml = max(sum, ml);
        }
        for (int i = m + 1, sum = 0; i <= r; i++) {
            sum += nums[i];
            mr = max(sum, mr);
        }
        //Combine the merge together
        return max(max(lmax, rmax), ml + mr + nums[m]);
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


//714. Best Time to Buy and Sell Stock with Transaction Fee
//https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
/* DP problem: We maintain two vectors. cash[i] means that at the end of day[i],
we do not hold a share of stock in hand. hold[i] means that at the end of day[i],
we hold a share of stock in hand. 
Then for cash[i], we have:
1. do not buy or sell, cash[i] = cash[i-1];
2. sell a stock at day i, cash[i] = hold[i-1] + prices[i] - fee
we will get the maximum of the two.
For hold[i], we have:
1. do not sell or buy, hold[i] = hold[i-1]
2. buy one share in day[i], hold[i] = cash[i-1] - prices[i]
3. sell one share in day[i] and buy it again. (does not make sense): 
hold[i] = hold[i-1] + prices[i] - fee - prices[i]
we will get the maximum of the three*/
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int len = prices.size();
        if(len == 0) return 0;
        vector<int> cash(len, 0), hold(len, 0);
        cash[0] = 0;
        hold[0] = -prices[0];
        for(int i = 1; i < len; ++i){
            cash[i] = max(cash[i-1], hold[i-1] + prices[i] - fee);
            hold[i] = max(hold[i-1], cash[i-1] - prices[i]);
        }
        return cash[len-1];
    }
};

/* Optimized DP*/
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int len = prices.size();
        if(len == 0) return 0;
        int cash = 0, hold = -prices[0];
        for(int i = 1; i < len; ++i){
            int prevCash = max(cash, hold + prices[i] - fee);
            hold = max(hold, cash - prices[i]);
            cash = prevCash;
        }
        return cash;
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

class Solution {
public:
    int trap(vector<int>& height) {
        if(height.size() <= 1) return 0;
        int len = height.size();
        int left[len] = {0}, right[len] = {0};
        int maxLeft = 0, maxRight = 0;
        for(int i = 0; i < len; ++i){
            maxLeft = max(maxLeft, height[i]);
            left[i] = maxLeft;
            maxRight = max(maxRight, height[len-1-i]);
            right[len-1 -i] = maxRight; 
        }
        int res = 0;
        for(int i = 1; i < len-1; ++i){
            res += max(0, min(left[i], right[i]) - height[i]);
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


//480. Sliding Window Median
//https://leetcode.com/problems/sliding-window-median/
//The naive approach is to sort the window each iteration! and save the result
//to our result array.
//Solution with two multiset. We need to balance the two sets on the fly.
//In general, the solution is hard to get during the interview!
class Solution {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        //lo stores the lower half elements, hi stores the higher half 
        //elements
        multiset<int> lo, hi;
        int len = nums.size();
        vector<double> res;
        
        //Note we will always make lo and hi balanced. If k is odd, then
        //lo will have 1 more element compared with hi. If k is even, then
        //lo will have the same number of elements as hi
        for(int i = 0; i < len; ++i){
            //When we have sufficient 
            if(i >= k){
                if(nums[i-k] <= *lo.rbegin())
                    lo.erase(lo.find(nums[i-k]));
                else
                    hi.erase(hi.find(nums[i-k]));
            }
            
            lo.insert(nums[i]);
            
            //Maitain the balance of two multiset
            hi.insert(*lo.rbegin());
            lo.erase(--lo.end());
            //It could potentially lo.size() is smaller, because we may 
            //extract the smaller elements from the list.
            if(lo.size() < hi.size()){
                lo.insert(*hi.begin());
                hi.erase(hi.begin());
            }
            
            if(i >= k-1){
                if(k%2 == 1)
                    res.push_back(double(*lo.rbegin()));
                else
                    res.push_back((double(*lo.rbegin()) + double(*hi.begin()))/2.0);
            }
            
        }
        return res;
    }
};



//295. Find Median from Data Stream
//https://leetcode.com/problems/find-median-from-data-stream/
/* Insertion sort Solution (O(n)) */
class MedianFinder {
private:
    vector<int> Con;
public:
    /** initialize your data structure here. */
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        auto it = lower_bound(Con.begin(), Con.end(), num);
        Con.insert(it, num);
    }
    
    double findMedian() {
        int len = Con.size();
        if(len == 0) return 0;
        else return len % 1 ? Con[len/2] : (Con[(len-1)/2] + Con[len/2]) * 0.5;
    }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */

/* Priority queue solution O(log n) */
class MedianFinder {
private:
    //The top element will be the largest of the queue
    priority_queue<int> maxQ;
    //The top element will be the smallest. We save larger half 
    //numbers in this queue, so the top will be close to median
    priority_queue<int, vector<int>, greater<int>> minQ;
public:
    /** initialize your data structure here. */
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        maxQ.push(num);
        //Balancing value, this step is critical!
        minQ.push(maxQ.top());
        maxQ.pop();
        //We always make maxQ have equal or more number of elements than minQ
        if(maxQ.size() < minQ.size()){
            maxQ.push(minQ.top());
            minQ.pop();
        }
        
    }
    
    double findMedian() {
        int len = maxQ.size() + minQ.size();
        if(len == 0) return 0;
        if(len & 1)
            return maxQ.top();
        else 
            return (minQ.top() + maxQ.top()) * 0.5;
    }
};

//Multiset solution. Easy to get the idea. 
//Very tricky when we handle the repetitive elements in the array.
//Pay attention to when length is even, how to handle the case.
//Still not very efficient
class MedianFinder {
private:
    multiset<int> mSet;
    multiset<int>::iterator loIt, hiIt;
public:
    /** initialize your data structure here. */
    MedianFinder() : loIt(mSet.end()), hiIt(mSet.end()) {
        
    }
    
    void addNum(int num) {
        //Get the length before inserting element
        int len = mSet.size();
        //When len is odd, after insert one element, the len will
        //be even.
        mSet.insert(num);
        
        if(len == 0){
            loIt = mSet.begin();
            hiIt = mSet.begin();
            return;
        }
        
        if(len & 1){
            if(num < *loIt)
                loIt--;
            else
                hiIt++;
        }else{
            //Note C++ will insert the new repetitive element in the 
            //end of all repetitive elements
            if(num > *loIt && num < *hiIt){
                loIt++;
                hiIt--;
            }
            else if(num >= *hiIt)
                loIt ++;
            else // num <= *loIt < *hiIt
                loIt = --hiIt; //insertion at the end of equal range spoils loIt
            //so we need loIt = --hiIt, instead of just hiIt--
                
        }
    }
    double findMedian() {
        if(loIt == mSet.end() && hiIt == mSet.end())
            return -1;
        return (*loIt + *hiIt) / 2.0;
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
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int len = nums.size();
        if(len < 2) return len;
        //vector<int> up(len, 0), down(len, 0);
        //The base case: when we have only 1 element
        int preUp = 1, preDown = 1;
        int curUp = 0, curDown = 0;
        //up[0] = down[0] = 1;
        for(int i = 1; i < len; ++i){
            if(nums[i] > nums[i-1]){
                //up[i] = max(up[i], down[i-1] + 1);
                //down[i] = down[i-1];
                curUp = preDown + 1;
                curDown = preDown;
            }else if(nums[i] < nums[i-1]){
                //down[i] = max(down[i], up[i-1] + 1);
                //up[i] = up[i-1];
                curDown = preUp + 1;
                curUp = preUp;
            }else{
                //up[i] = up[i-1];
                //down[i] = down[i-1];
                curUp = preUp;
                curDown = preDown;
            }
            swap(curUp, preUp);
            swap(curDown, preDown);
            curUp = curDown = 0;
        }
        return max(preUp, preDown);
    }
};

/* Greedy. */
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int len = nums.size();
        if(len < 2) return len;
        //We maintain a preDiff variable to check whether
        //we are at a descending list or ascending list.
        int preDiff = 0;
        //at least one
        int maxCount = 1;
        //We only update maxCount when preDiff changes the sign
        for(int i = 1; i < len; ++i){
            if(preDiff == 0 && (nums[i] - nums[i-1] != 0)){
                preDiff = nums[i] - nums[i-1];
                maxCount += 1;
            }else if(preDiff > 0 && nums[i] - nums[i-1] < 0){
                maxCount += 1;
                preDiff = nums[i] - nums[i-1];
            }else if(preDiff < 0 && nums[i] - nums[i-1] > 0){
                maxCount += 1;
                preDiff = nums[i] - nums[i-1];
            }
        }
        return maxCount;
    }
};


//327. Count of Range Sum
//https://leetcode.com/problems/count-of-range-sum/
/* Binary search solution! Hard to get it right! Take a close look at later! */
class Solution {
private:
    int calRangeSum(vector<long>& preSum, int lo, int up, int l, int r){
        //since r is exclusive, when r - l < 1, we only have 0 element
        if(r - l <= 1) return 0;
        int mid = l + (r-l)/2;
        int count = 0;
        count = calRangeSum(preSum, lo, up, l, mid) + calRangeSum(preSum, lo, up, mid, r);
        for(int i = l; i < mid; ++i){
            auto loIt = lower_bound(preSum.begin() + mid, preSum.begin() + r, preSum[i] + lo);
            auto upIt = upper_bound(preSum.begin() + mid, preSum.begin() + r, preSum[i] + up);
            count += upIt - loIt;
            //cout << count << endl;
        }
        //merge the two sub arrays in place, sort in progress
        inplace_merge(preSum.begin()+l, preSum.begin() + mid, preSum.begin() + r);
        return count;
    }
public:
    int countRangeSum(vector<int>& nums, int lower, int upper) {
        int len = nums.size();
        if(len == 0) return 0;
        vector<long> sum(len + 1, 0);
        for(int i = 0; i < len; ++i){
            sum[i+1] += sum[i] + nums[i];
        }
        //Note the r value is exclusive
        return calRangeSum(sum, lower, upper, 0, len+1);
        //return 0;
    }
};


//306. Additive Number
//https://leetcode.com/problems/additive-number/
/* Tricky problem. The idea is simple. Handle the test case related to '0'
is not easy! Another thing is we need to handle the sum correctly in order
to avoid overflow. */
class Solution {
    bool checkString(string& s, int pos, long preSum, bool& toEnd){
        unsigned long sum = 0;
        int len = s.size();
        for(int i = pos; i < len; ++i){
            if(i > pos && s[pos] == '0')
                break;
            sum = sum * 10 + (s[i] - '0');
            if(i == len-1) toEnd = true;
            if(sum > preSum) return false;
            if(sum == preSum) return true;
        }
        return false;
    }
    bool helper(string& s, int pos, long preNum){
        int len = s.size();
        //Cannot put the following statement here. Or we will always return true
        //if(pos == len) return true;
        unsigned long tempSum = 0;
        for(int i = pos; i < len; ++i){
            bool reachEnd = false;
            tempSum = tempSum * 10 + (s[i] - '0');
            if(i > pos && s[pos] == '0')
                break;
            bool isMatch = checkString(s, i+1, tempSum + preNum, reachEnd);
            //cout << tempSum + preNum << " " << isMatch << endl;
            if(isMatch && reachEnd)
                return true;
            else if(isMatch && helper(s, i+1, tempSum))
                return true;

        }
        return false;
    }
public:
    bool isAdditiveNumber(string num) {
        int len = num.size();
        //We need to use unsigned long to handle overflow! not good.
        //We can calculate the sum in terms of string
        unsigned long sum = 0;
        for(int i = 0; i < num.size(); ++i){
            sum = sum * 10 + (num[i] - '0');
            //cout << sum << " " << endl;
            if(helper(num, i+1, sum))
                return true;
            //Handle the case '0235'. Should be false
            if(num[0] == '0')
                return false;
        }
        return false;
    }
};


/* Optimized version: beautiful! */
class Solution {
private:
    /* The addString function calculates the result digit by digit, so it will have no overflow.
    Powerful techniques, please use it in the future*/
    string addString(string& s1, string& s2){
        int sum = 0;
        int carry = 0;
        string res;
        int i = s1.size() - 1, j = s2.size() - 1;
        //save the result to res string
        while(i >= 0 || j >= 0){
            sum = carry + (i >= 0 ? (s1[i--] - '0') : 0) + (j >= 0 ? (s2[j--] - '0') : 0);
            carry = sum / 10;
            sum = sum % 10;
            res.push_back(sum + '0');
        }
        if(carry) res.push_back(carry + '0');
        reverse(res.begin(), res.end());
        return res;
    }
    bool checkAdditive(string s1, string s2, string sum){
        int len1 = s1.size(), len2 = s2.size();
        //Handle the case when leading character is 0
        if((len1 > 1 && s1[0] == '0') || (len2 > 1 && s2[0] == '0')) return false;
        string localSum = addString(s1, s2);
        if(localSum == sum) return true;
        else if(localSum.size() >= sum.size() || localSum.compare(sum.substr(0, localSum.size()))!= 0)
            return false;
        else //check from s2 recursively. Note we have to start from localSum.size() for the third parameter
            return checkAdditive(s2, localSum, sum.substr(localSum.size()));
    }
public:
    bool isAdditiveNumber(string num) {
        int len = num.size();
        //the length of the first number cannot exceed
        //len / 2
        for(int i = 1; i <= len/2; ++i){
            //The length of the second number can not exceed (len-i)/2
            for(int j = 1; j <= (len - i) / 2; ++j){
                //num.substr(i+j), potential sum from position i+j to the end
                if(checkAdditive(num.substr(0, i), num.substr(i, j), num.substr(i+j))) 
                    return true;
            }
        }
        return false;
    }
};


//321. Create Maximum Number
//https://leetcode.com/problems/create-maximum-number/
/* A very tricky problem! Took me some time to finish it. This implementation is 
elegant. Take a look at it later.
The general idea for this problem is we pick k elements total from both arrays. So
we can first check if we pick all the min(len1, k) elements from array 1, and check
what will be the maximum value from array 1. Then we pick up k - min(len1, k) 
elements from array 2 and check the maximum value from array 2. Then we merge the
two results together we will get a tempRes, which will be the maximum value for
this round. We then iterate through all the potential pickup from array 1 and 2,
and calculate the maximum value in the end.
 */
class Solution {
private:
    vector<int> mergeVector(vector<int>&& nums1, vector<int>&& nums2){
        int k = nums1.size() + nums2.size();
        vector<int> res(k, 0);
        for(int i = 0, j = 0, r = 0; r < k; ++r){
            if(compareGreaterRes(nums1, i, nums2, j)) res[r] = nums1[i++];
            else res[r] = nums2[j++];
        }
        return res; 
    }
    vector<int> calMaxVector(vector<int>& nums, int k){
        if(k == 0) return vector<int>();
        vector<int> vec;
        int len = nums.size();
        vec.push_back(nums[0]);
        //need to guarantee that vec.size() <= k
        for(int i = 1; i < len; ++i){
            while(!vec.empty() && vec.size() + len - i > k && nums[i] > vec.back()){
                vec.pop_back();
            }
            if(vec.size() < k)
                vec.push_back(nums[i]);
            else if(vec.size() == k && vec.back() < nums[i])
                vec.back() = nums[i];
        }
        return vec;
    }
    /* This compareGreaterRes function is tricky. Note it does not return true if nums is 
    literally greater than res. It will return true if it finds the first character from
    nums[i] which is greater than res[i]. For example:
    nums = [6,7], res = [6,0,7] It will return true, and we always get the greater numbers
    out first.*/
    bool compareGreaterRes(vector<int>& nums, int pos1, vector<int>& res, int pos2){
        int len = nums.size(), lenR = res.size();
        while(pos1 < len && pos2 < lenR && nums[pos1] == res[pos2]){
            pos1++;
            pos2++;
        }
        //either pos2 reaches the end of the string
        return pos2 == lenR || (pos1 < len && nums[pos1] > res[pos2]);
    }
    
public:
    vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<int> res(k, 0);
        int len1 = nums1.size(), len2 = nums2.size();
        //In the first round, we start from adding potential maximum elements from nums2 to res,
        //then we add element from nums1 and remove element from nums2. Whenever we add or remove
        //element, we always calculate the maximum result for the selected elements
        for(int i = max(0, k - len2); i <= min(k, len1); ++i){
            vector<int> tempVector = mergeVector(calMaxVector(nums1, i), calMaxVector(nums2, k-i));
            //tempVector and res now have the same length k
            if(compareGreaterRes(tempVector, 0, res, 0)) res.swap(tempVector);
        }
        return res;
    }
};

//349. Intersection of Two Arrays
//https://leetcode.com/problems/intersection-of-two-arrays/
/* O(m+n) */
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> set1;
        int len1 = nums1.size(), len2 = nums2.size();
        //Make sure that len1 will always be the shorter one
        if(len2 < len1) return intersection(nums2, nums1);
        for(int n : nums1)
            set1.insert(n);
        
        vector<int> res;
        
        for(int i = 0; i < len2; ++i){
            if(set1.count(nums2[i]) > 0) {
                res.push_back(nums2[i]);
                set1.erase(nums2[i]);
            }
        }
        return res;
    }
};

//350. Intersection of Two Arrays II
//https://leetcode.com/problems/intersection-of-two-arrays-ii/
class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> map;
        int len1 = nums1.size(), len2 = nums2.size();
        //Make sure that len1 will always be the shorter one
        if(len2 < len1) return intersect(nums2, nums1);
        for(int n : nums1){
            if(map.find(n) != map.end())
                map[n]++;
            else
                map[n] = 1;
        }
        vector<int> res;
        for(int i = 0; i < len2; ++i){
            if(map.count(nums2[i]) > 0) {
                res.push_back(nums2[i]);
                map[nums2[i]] --;
                if(map[nums2[i]] == 0)
                    map.erase(nums2[i]);
            }
        }
        return res;
    }
};

//Follow up: What if the arrays are sorted?
/*
Cases to take into consideration include:
duplicates, negative values, single value lists, 0's, and empty list arguments.
*/
//Two pointers, can save space
class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        //Assume the arrays are sorted
        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());
        int n1 = (int)nums1.size(), n2 = (int)nums2.size();
        int i1 = 0, i2 = 0;
        vector<int> res;
        while(i1 < n1 && i2 < n2){
            if(nums1[i1] == nums2[i2]) {
                res.push_back(nums1[i1]);
                i1++;
                i2++;
            }
            else if(nums1[i1] > nums2[i2]){
                i2++;
            }
            else{
                i1++;
            }
        }
        return res;
    }
};


//396. Rotate Function
//https://leetcode.com/problems/rotate-function/
/* A very naive approach will be reverse the array and
calculate all the sums. However, we can get the correct
result by two loops. The first loop we will calculate
the total sum of all elements in Array A. We also calculate
the F(0) in the first loop. For the second loop, we will
calculate the F(1) ... F(n-1) respectively. And keep track
of the maximum result.*/
class Solution {
public:
    int maxRotateFunction(vector<int>& A) {
        int len = A.size();
        long totalSum = 0, FSum = 0;
        for(int i = 0; i < len; ++i){
            totalSum += A[i];
            FSum += i * A[i];
        }
        int res = FSum;
        //We should start from 0, in the first round,
        //we calculate F(1)
        for(int i = 0; i < len; ++i){
            FSum -= totalSum;
            FSum += A[i];
            FSum += (len - 1) * A[i];
            res = max(long(res), FSum);
        }
        return res;
    }
};


//357. Count Numbers with Unique Digits
//https://leetcode.com/problems/count-numbers-with-unique-digits/
class Solution {
public:
    int countNumbersWithUniqueDigits(int n) {
        //We have only 1 valid result
        if(n == 0) return 1;
        //when n == 1, then sum = 10
        int sum = 10;
        int base = 9;
        //note when n > 10, we actually go to a situation that
        //we cannot have any valid possible result with 11 digits
        //because we only have 0-9 10 digits in total. so when 
        //n == 10, the sum will be maximum
        for(int i = 1; i < n && i < 10; ++i){
            base = base * (10 - i);
            sum += base;
        }
        
        return sum;
    }
};


//57. Insert Interval
//https://leetcode.com/problems/insert-interval/
/* Needs sometime to get it right! I do not really like it */
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        if(intervals.empty()){
            intervals.push_back(newInterval);
            return intervals;
        }
        int nS = newInterval[0], nE = newInterval[1];
        int len = intervals.size();
        int startIndex = 0;
        int deleteLen = 0;
        for(int i = 0; i < len; ++i){
            if(nS <= intervals[i][1]){
                if(nE >= intervals[i][0])
                    intervals[i][0] = min(nS, intervals[i][0]);
                if(nE < intervals[i][0]){
                    intervals.insert(intervals.begin() + i, newInterval);
                    return intervals;
                }
                else if(nE <= intervals[i][1])
                    break;
                else if(nE > intervals[i][1]){
                    startIndex = i;
                    intervals[i][1] = nE;
                    i++;
                    while(i < len && intervals[i][0] <= nE){
                        intervals[startIndex][1] = max(nE, intervals[i][1]);
                        deleteLen ++;
                        i++;
                    }
                    break;
                }
            }else if(nS > intervals[i][1]){
                if(i + 1 == len || (i+1 < len && nE < intervals[i+1][0])){
                    intervals.insert(intervals.begin() + i + 1, newInterval);
                    return intervals;
                }
            }
        }
        int j = startIndex+1;
        for(; j < len && (j+deleteLen) < len; ++j){
            intervals[j] = intervals[j + deleteLen];
        }
        while(j < len){
            intervals.pop_back();
            j++;
        }
        return intervals;
    }
};


//Another implementation
/**
 * Definition for an interval.
 * struct Interval {
 *     int start;
 *     int end;
 *     Interval() : start(0), end(0) {}
 *     Interval(int s, int e) : start(s), end(e) {}
 * };
 */
class Solution {
public:
    vector<Interval> insert(vector<Interval>& intervals, Interval newInterval) {
        vector<Interval> finalRes;       
        int len = intervals.size();
        int i = 0;
        while(i < len && newInterval.start > intervals[i].end)
            finalRes.push_back(intervals[i++]);
        
        while(i < len && intervals[i].start <= newInterval.end){
            newInterval.start = min(intervals[i].start, newInterval.start);
            newInterval.end = max(intervals[i].end, newInterval.end);
            i++;
        }
        finalRes.push_back(newInterval);
        
        while(i<len)
            finalRes.push_back(intervals[i++]);
        
        return finalRes;
        
    }
};


//414. Third Maximum Number
//https://leetcode.com/problems/third-maximum-number/
/* Set solution: get rid of duplicate */
class Solution {
public:
    int thirdMax(vector<int>& nums) {
        //Cannot use priority queue here because pq allows duplicate elements
        //[2, 2, 3, 1] will fail
        set<int> maxThird;
        for(int num : nums){
            maxThird.insert(num);
            if(maxThird.size() > 3)
                maxThird.erase(maxThird.begin());
        }
        //How to use iterator is critical!!!
        return maxThird.size() == 3 ? (*maxThird.begin()) : (*maxThird.rbegin());
    }
};


/* Three pointers solution. I like it! */
class Solution {
public:
    int thirdMax(vector<int>& nums) {
        //one - largest element, three - third largest element
        //handle [1,2,-2147483648]
        long three = LONG_MIN, two = LONG_MIN, one = LONG_MIN;
        //Note when we have duplicates, we ignore it in our loop
        for(int n : nums){
            if(n > one){
                three = two;
                two = one;
                one = n;
            }
            else if(n < one && n > two){
                three = two;
                two = n;
            }else if(n < two && n > three){
                three = n;
            }
        }
        //if we have less than 3 elements
        return three == LONG_MIN ? one : three;
    }
};

//152. Maximum Product Subarray
//https://leetcode.com/problems/maximum-product-subarray/
//Naive implementation by me
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        int minDP[len] = {0};
        int maxDP[len] = {0};
        minDP[0] = nums[0];
        maxDP[0] = nums[0];
        int res = max(INT_MIN, nums[0]);
        for(int i = 1; i < len; ++i){
            if(nums[i] == 0){
                minDP[i] = maxDP[i] = 0;
                res = max(res, 0);
            }else if(nums[i] > 0){
                maxDP[i] = max(nums[i], maxDP[i-1] * nums[i]);
                minDP[i] = min(nums[i], minDP[i-1] * nums[i]);
                res = max(res, maxDP[i]);
            }else{
                minDP[i] = min(nums[i], maxDP[i-1] * nums[i]);
                maxDP[i] = max(nums[i], minDP[i-1] * nums[i]);
                res = max(res, maxDP[i]);
            }
        }
        return res;
    }
};

//Note maximum positive and negative value can swap by multiplying negative value
//Very tricky implementation! Similar to Kadane's algorithm
//We will keep track of the potential maximum and minimum production for each entry
//i. Actually, pos and neg can be an arra, so the formula will be
//pos[i] = max(nums[i], max(pos[i-1]*nums[i], neg[i-1]*nums[i]));
//neg[i] = min(nums[i], min(pos[i-1]*nums[i], neg[i-1]*nums[i]));
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        int pos = nums[0], neg = nums[0];
        //Set res to nums[0], handles the array which contains only 1 element
        int res = nums[0];
        for(int i = 1; i < len; ++i){
            int p = pos * nums[i];
            int q = neg * nums[i];
            //Note when neg multiply by negative value, q will be positive and 
            //potentially the largest value. For index i, we need to always 
            //save the potential minimum value in neg.
            pos = max(nums[i], max(p, q));
            neg = min(nums[i], min(p, q));
            res = max(res, pos);
        }
        return res;
    }
};

//We only need to consider 3 situations:
/*
There is no 0 in the array:
1. if we contains even number of negative numbers, basically, the max product will be the product of all elements;
2. If we have odd number of negative numbers, we need to consider whether we 
drop the first negative number or the last.
3.With 0, we only need to update the result to be 1 after comparison
Then the general idea is to product from both end and handle 0 separately!
*/
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int forwardProduct = 1, backwardProduct = 1;
        int res = INT_MIN;
        int len = nums.size();
        for(int i = 0; i < len; ++i){
            forwardProduct *= nums[i];
            backwardProduct *= nums[len- 1 - i];
            res = max(res, max(forwardProduct, backwardProduct));
            forwardProduct = forwardProduct ? forwardProduct : 1;
            backwardProduct = backwardProduct ? backwardProduct : 1;
        }
        return res;
    }
};


//456. 132 pattern
//https://leetcode.com/problems/132-pattern/
/* Brueforce algorithm */
class Solution {
public:
    bool find132pattern(vector<int>& nums) {
        int len = nums.size();
        if(len < 3) return false;
        vector<int> minLeft(len, INT_MAX);
        for(int i = 1; i < len; ++i){
            minLeft[i] = min(minLeft[i-1], nums[i-1]);
        }
        
        for(int j = 1; j < len - 1; ++j){
            if(nums[j] < minLeft[j]) continue;
            int k = j + 1;
            for(; k< len; ++k){
                if(nums[k] < nums[j] && nums[k] > minLeft[j])
                    return true;
            }
        }
        return false;
    }
};

/*
Explanation: https://leetcode.com/problems/132-pattern/discuss/94071/Single-pass-C%2B%2B-O(n)-space-and-time-solution-(8-lines)-with-detailed-explanation.
The requirement is that we need to find i < j < k and s1 < s3 < s2. Then we can
start from the end of the array and always maintain the maximum possible value
for s3, and we maintain a stack to keep track of all values of s2 (s2 must be 
greater than s3). Once we know that s2 is greater than s3, and if we find any
nums[i] < s3, then we have s1 < s3 < s2, we can return true.
Very tricky problem!
*/
class Solution {
public:
    bool find132pattern(vector<int>& nums) {
        int len = nums.size();
        if(len < 3) return false;
        stack<int> s2St;
        int s3 = INT_MIN;
        for(int i = len - 1;  i >= 0; --i){
            if(s3 > nums[i]) return true;
            else{
                while(!s2St.empty() && nums[i] > s2St.top()){
                    s3 = s2St.top();
                    s2St.pop();
                }
            }
            //nums[i] is greater than s3 at this point (guaranteed!)
            s2St.push(nums[i]);
        }
        return false;
    }
};


//765. Couples Holding Hands
//https://leetcode.com/problems/couples-holding-hands/
//Greedy: https://leetcode.com/problems/couples-holding-hands/discuss/113358/Easy-to-understand-C%2B%2B-O(n)-hashmap-solution-with-explanation
/*
A key observation is, a couple must occupies the seats 2 * i and 2 * i + 1 
(e.g. seats 0 & 1), they must not take seats 2 * i - 1 and 2 * i (e.g. seats 1 
and 2), or the fisrt and last seat will be left empty and the last one couple
cannot sit together.
Then we can scan from the begining of the array, and check whether each spouse of
that person is sitting beside him/her. In order to quickly check the relationship
with people and the seats, we can build a table to store the relationship.
*/
class Solution {
public:
    int minSwapsCouples(vector<int>& row){
        int len = row.size();
        if(len <= 2) return 0;
        vector<int> dict(len);
        for(int i = 0; i < len; ++i){
            //person row[i] sits in seat i
            dict[row[i]] = i;
        }
        int count = 0;
        //then we can greedy swap one couple and make the local optimal greedily
        for(int i = 0; i < len; i = i+2){
            int me = row[i];
            int spouse = (me&1) ? me - 1 : me + 1;
            int neighbour = row[i+1];
            if(spouse != neighbour){
                int seatSpouse = dict[spouse];
                swap(row[i+1], row[seatSpouse]);
                count ++;
                //update the swapped person's seat
                dict[neighbour] = seatSpouse;
            }
        }
        return count;
    }
};


//421. Maximum XOR of Two Numbers in an Array
//https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/
class Solution {
    //Note we are trying to build a trie, that the root represents the most 
    //significant bits (left most bit)
    //We can do better by preprocessing the array n and find the position of
    //the most significant 1 in all numbers by:
    //int pos = (int)(log2(n));
    //Then we can start the inner loop from the left most pos instead of 31
    //for [3, 10, 5, 25, 2, 8], the left most pos is 6, from 25
    TreeNode* buildTree(const vector<int>& n, const int maxPos){
        TreeNode* root = new TreeNode(0);
        for(int i = 0; i < n.size(); ++i){
            TreeNode* cur = root;
            for(int j = maxPos; j >= 0; --j){
                int bit = (n[i] & (1 << j)) ? 1 : 0;
                //cout << bit << " ";
                if(bit == 1){
                    if(!cur->right) cur->right = new TreeNode(bit);
                    cur = cur->right;
                }else{
                    if(!cur->left) cur->left = new TreeNode(bit);
                    cur = cur->left;
                }
            }
        }
        return root;
    }
public:
    int findMaximumXOR(vector<int>& nums) {
        int maxPos = 0;
        for(int n : nums){
            maxPos = max(maxPos, static_cast<int>(log2(n)));
            //cout << maxPos << " ";
        }
        TreeNode* root = buildTree(nums, maxPos);
        int res = 0;
        for(int i = 0; i < nums.size(); ++i){
            int curNum = nums[i];
            TreeNode* cur = root;
            int tempNum = 0;
            for(int j = maxPos; j >= 0; --j){
                int bit = (curNum & (1 << j)) ? 1 : 0;
                if(cur->left && cur->right){
                    cur = (bit == 1) ? cur->left : cur->right;
                }else
                    cur = (cur->left) ? cur->left : cur->right;
                //cout << cur->val << " ";
                tempNum |= (cur->val << j);
            }
            //cout << endl;
            res = max(curNum ^ tempNum, res);
        }
        return res;
    }
};



//413. Arithmetic Slices
//https://leetcode.com/problems/arithmetic-slices/
//Not hard, just try to find the right rule!
//(n*n - 3*n) / 2 + 1
class Solution {
private:
    //n is the length of arithmetic slice
    int calCulateArithmetic(int n){
        if(n < 3) return 0;
        return (n*n - 3*n) / 2 + 1;
    }
public:
    int numberOfArithmeticSlices(vector<int>& A) {
        unsigned int len = A.size();
        if(len < 3) return 0;
        int tempSub = 0;
        int res = 0, localMaxLen = 2;
        for(int j = 1; j < len; ++j){
            if(j == 1) {
                tempSub = A[j] - A[j-1];
                continue;
            }
            if(A[j-1] + tempSub == A[j])
                localMaxLen ++;
            else{
                res += calCulateArithmetic(localMaxLen);
                localMaxLen = 2;
                tempSub = A[j] - A[j-1];
            }
        }
        res += calCulateArithmetic(localMaxLen);
        return res;
    }
};


//DP solution, note dp[i] = dp[i-1] + 1
//Actually we can get rid of the array
class Solution {
public:
    int numberOfArithmeticSlices(vector<int>& A) {
        int n = A.size();
        if (n < 3) return 0;
        vector<int> dp(n, 0); // dp[i] means the number of arithmetic slices ending with A[i]
        if (A[2]-A[1] == A[1]-A[0]) dp[2] = 1; // if the first three numbers are arithmetic or not
        int result = dp[2];
        for (int i = 3; i < n; ++i) {
            // if A[i-2], A[i-1], A[i] are arithmetic, then the number of arithmetic slices ending with A[i] (dp[i])
            // equals to:
            //      the number of arithmetic slices ending with A[i-1] (dp[i-1], all these arithmetic slices appending A[i] are also arithmetic)
            //      +
            //      A[i-2], A[i-1], A[i] (a brand new arithmetic slice)
            // it is how dp[i] = dp[i-1] + 1 comes
            if (A[i]-A[i-1] == A[i-1]-A[i-2]) 
                dp[i] = dp[i-1] + 1;
            result += dp[i]; // accumulate all valid slices
        }
        return result;
    }
};


//416. Partition Equal Subset Sum
//https://leetcode.com/problems/partition-equal-subset-sum/
//DFS + Memorization! Note in order to save time, we use the set to track
//pos and target values! Not that efficient!
class Solution {
private:
    int helper(int pos, int target, vector<int>& n, unordered_set<string>& memo){
        if(target == 0) return 1;
        if(target < 0 || pos == n.size()) return 0;
        if(memo.count(to_string(pos) + '_' + to_string(target))>0) return false;
        int res;
        for(int i = pos; i < n.size(); ++i){
            res = helper(i+1, target - n[i], n, memo);
            if(res) return 1;
        }
        //we only record failed paths!
        memo.insert(to_string(pos) + '_' + to_string(target));
        return 0;
    }
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for(int n : nums){
            sum += n;
        }
        if(sum % 2 == 1) return false;
        int target = sum / 2;
        //The general vector<int> memo here won't work since
        //we need to keep track of the failed path! or we need 
        //to allocate a 2D array to keep track of both pos and 
        //target. That won't pass the test case!
        //So we use a uset to record the path
        unordered_set<string> memo;
        return helper(0, target, nums, memo);
    }
};

//Time exceed limit: early trimming DFS
//failed case: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,100]
class Solution {
private:
    int helper(int pos, int target, vector<int>& n){
        if(target == 0) return 1;
        if(target < 0 || pos == n.size()) return 0;
        for(int i = pos; i < n.size(); ++i){
            //early trim: since the array is sorted
            if(target < n[i])
                break;
            if(helper(i+1, target - n[i], n))
                return 1;
        }
        return 0;
    }
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for(int n : nums){
            sum += n;
        }
        if(sum % 2 == 1) return false;
        int target = sum / 2;
        //we need to sort the array for early trim!
        sort(nums.begin(), nums.end());
        return helper(0, target, nums);
    }
};


//DP solution
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int len = nums.size();
        int sum = 0;
        for(int i : nums)
            sum += i;
        if(sum & 1) return false;
        int target  = sum / 2;
        vector<int> dp(target+1, 0);
        //when target reaches 0
        dp[0] = 1;

        //Please draw the dp table, note we cannot swap the two loops 
        //here since when we calculate dp[i][j], we need the value for
        //dp[i][j-1], so we need to calculate dp[i][j-1] first. And we
        //also need to start from i = target, since dp[i-nums[j]][j-1]
        //should be preseved before update!
        //A safe way is to use 2D array to get the solution and then 
        //optimize it.
        for(int j = 0; j < len-1; ++j){
            for(int i = target; i >= nums[j]; --i){
                 if(i >= nums[j])
                    dp[i] = dp[i] || dp[i - nums[j]];
                //Cannot add dp[i] break here, since we need to reuse
                //the result for next round, so we need to calculate
                //the result
                //if(dp[i]) break;
            }
        }
        return dp[target];
    }
};

//Bit Solution
//An elegant bitset solution! very clever!
//Since each element will not be able to exceeds 100, and the array size
//will not exceed 200, we can allocate the bitset which represents the 
//total Sum.
/*
Size of bitset is the maximum sum that can be achieved by array + 1.
Ex. [5,2,4] ---> bitset of size 12 ==> 000000000001
That means initially we can achieve sum 0 with empty array subset [ ].
We have only 0's bit set.

num = 5
0 -> 5 (set 5's bit, since we can achieve sum 5.)
Now we can achieve 0 and 5 with [ ] and [ 5 ]. So by the union of both, we have 000000100001

num = 2
0->2
5->7
We can achieve 0,2,5,7 from [5,2] ==> [ ], [5], [2], [5,2]
After union our bitset is 000010100101

num = 4
0->4
2->6
5->9
7->11
We can achieve 0,2,4,5,6,7,11 from [5,2] ==> [ ], [5], [2], [4], [5,2], [2,4], [5,4], [5,2,4]
After union our bitset is 101011110101
*/
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        //Initialize the first bit to be 1 (rightmost)
        bitset<20001> bits(1);
        int sum = 0;
        for(int n : nums){
            sum += n;
            bits |= (bits << n);
        }
        return (!(sum & 1)) && bits[sum >> 1];
    }
};


//904. Fruit Into Baskets
//https://leetcode.com/problems/fruit-into-baskets/
//Sliding window. The first pointer always includes new elements
//When our basket is full, we need to move the second pointer 
//forward. In general, the time complexity is O(n)
class Solution {
public:
    int totalFruit(vector<int>& tree) {
        unsigned int len = tree.size();
        if(len == 0) return 0;
        unordered_map<int, int> uMap;
        uMap[tree[0]] = 1;
        int maxFruit = 1;
        int i = 0;
        int j = 1;
        for(; i < len; ){
            while(j < len){
                if(uMap.count(tree[j]) == 0){
                    uMap[tree[j]] = 1;
                    if(uMap.size() > 2){
                        maxFruit = max(maxFruit, j - i);
                        j++;
                        break;
                    }
                }else{
                    uMap[tree[j]]++;
                }
                j++;
            }    
            while(uMap.size() > 2){
                uMap[tree[i]]--;
                if(uMap[tree[i]] == 0)
                    uMap.erase(tree[i]);
                i++;
            }
            if(j == len && uMap.size() <= 2)
                break;
        }
        maxFruit = max(maxFruit, j - i);
        return maxFruit;
    }
};

//Optimized sliding window!
//Sliding window. The first pointer always includes new elements
//When our basket is full, we need to move the second pointer 
//forward. In general, the time complexity is O(n)
//Exactly the same idea! However, this implementation is elegant!
//Note we do not need to move i to the designated position and 
//update each time, since i and j move with the same speed, we can
//update i only when count.size() > 2
class Solution {
public:
    int totalFruit(vector<int> &tree) {
        unordered_map<int, int> count;
        int i, j;
        for (i = 0, j = 0; j < tree.size(); ++j) {
            count[tree[j]]++;
            if (count.size() > 2) {
                if (--count[tree[i]] == 0)count.erase(tree[i]);
                i++;
            }
        }
        return j - i;
    }
};


//857. Minimum Cost to Hire K Workers
//https://leetcode.com/problems/minimum-cost-to-hire-k-workers/
//A very good explanation:
//https://leetcode.com/problems/minimum-cost-to-hire-k-workers/discuss/141768/Detailed-explanation-O(NlogN)
//The key insight is to find that in order to make the minimum 
//payment, there is at least one worker which will be paid the his
//expected minimum wage. Another important facts is that if we 
//select a person whose ratio is the smallest, we can never make 
//any other people happy. For example: [10, 20, 5] - [70, 50, 30]
//The second worker has the smallest ratio, which is 2.5, if we 
//pick him, then according to the first rule, the first person 
//will be paid 25, and the third will be paid 12.5. Which does not
//satisfy the second rule. If we pick up the person with the largest
//ratio to pay him the expected minimum wage, say first person. 
//we can safely pick either second one or third one. 
//That is to say, if person i has raotio R[i], then any one whose
//ratio greater than R[i] cannot be included!
//Then we can see a clear strategy is to greedily select a person
//with lower ratio, and pick up k people with smallest possible 
//qualities, and we check for each person from samller ratio to
//higher ratio.
//Time: O(nlogn)
class Solution {
private:
    struct Workers{
        int quality, wage;
        double ratio;
        Workers(int q, int w){
            quality = q;
            wage = w;
            ratio = double(wage) / double(quality);
        }
    };
    
    struct myComp{
        bool operator()(Workers& w1, Workers& w2){
            return w1.ratio < w2.ratio;
        }
    } Comparator;
    
public:
    double mincostToHireWorkers(vector<int>& quality, vector<int>& wage, int K) {
        int len = quality.size();
        vector<Workers> workerList;
        //build the workers list
        for(int i = 0; i < len; ++i){
            workerList.push_back(Workers(quality[i], wage[i]));
        }
        double res = DBL_MAX;
        //Calculate the current total quality can save us time to calculate
        //current payment when we select worker[i]
        int sumQuality = 0;
        //sort the ratio from the smallest to the lagest
        sort(workerList.begin(), workerList.end(), Comparator);
        //save the quality to max queue, we only care about workers
        //with smaller quality
        priority_queue<int> pq;
        for(auto& w : workerList){
            sumQuality += w.quality;
            pq.push(w.quality);
            if(pq.size() > K){
                sumQuality -= pq.top();
                pq.pop();
            } 
            if(pq.size() == K) res = min(res, sumQuality * w.ratio);
        }
        
        return res;
    }
};


//410. Split Array Largest Sum
//https://leetcode.com/problems/split-array-largest-sum/
//My implementation, unoptimized!
class Solution {
public:
    int splitArray(vector<int>& nums, int m) {
        if(nums.empty()) return 0;
        int len = nums.size();
        long dp[m][len] = {0};
        long preSum[len] = {0};
        dp[0][0] = nums[0];
        preSum[0] = nums[0];
        
        for(int i = 1; i < len; ++i){
            dp[0][i] = dp[0][i-1] + nums[i];
            preSum[i] = preSum[i-1] + nums[i];
        }
        
        for(int i = 1; i < m; ++i){
            for(int j = 0; j < len; ++j){
                long localMin = LONG_MAX;
                for(int k = 0; k < j; ++k){
                    localMin = min(max(dp[i-1][k], preSum[j] - preSum[k]), localMin);
                }
                dp[i][j] = j == 0 ? preSum[j] : localMin;
            }
        }
        return dp[m-1][len-1];
    }
};


//Optimized version
class Solution {
public:
    int splitArray(vector<int>& nums, int m) {
        if(nums.empty()) return 0;
        int len = nums.size();
        //long dp[m][len] = {0};
        long* pre = new long [len];
        long* cur = new long [len];
        
        long preSum[len] = {0};
        //dp[0][0] = nums[0];
        pre[0] = nums[0];
        preSum[0] = nums[0];
        
        for(int i = 1; i < len; ++i){
            //dp[0][i] = dp[0][i-1] + nums[i];
            pre[i] = pre[i-1] + nums[i];
            preSum[i] = preSum[i-1] + nums[i];
        }
        
        for(int i = 1; i < m; ++i){
            for(int j = 0; j < len; ++j){
                long localMin = LONG_MAX;
                for(int k = j-1; k >= 0; --k){
                    //No need to traverse further, since preSum[k] will become smaller and
                    //smaller! so preSum[j] - preSum[k] will be larger and larger
                    if(preSum[j] - preSum[k] >= localMin) break;
                    //localMin = min(max(dp[i-1][k], preSum[j] - preSum[k]), localMin);
                    localMin = min(max(pre[k], preSum[j] - preSum[k]), localMin);
                }
                //dp[i][j] = j == 0 ? preSum[j] : localMin;
                cur[j] = j == 0 ? preSum[j] : localMin;
            }
            swap(pre, cur);
        }
        
        int res = pre[len-1];
        delete[] pre;
        delete[] cur;
        //return dp[m-1][len-1];
        return res;
    }
};


//Good explanation:
//https://leetcode.com/problems/split-array-largest-sum/discuss/89819/C%2B%2B-Fast-Very-clear-explanation-Clean-Code-Solution-with-Greedy-Algorithm-and-Binary-Search
//Very tricky solution: Greedy + Binary search!
class Solution {
private:
    //this function answers a question: if given cuts number of 
    //cuts, can we split the array nums with cuts+1 subarray, with
    //each subarray sum <= upperBound ?
    bool canSplit(vector<int>&nums, int cuts, int upperBound){
        //we use acc to record whether we can put this element to
        //current subarray, if we can, acc += n; else we have to
        //use one more cuts, and start over with acc = n
        long long acc = 0;
        for(int n : nums){
            //impossible if upperbound is greater than individual 
            //element
            if(n > upperBound) return false;
            else if(acc + n <= upperBound){
                acc += n;
            }else{
                acc = n;
                cuts--;
                if(cuts < 0) return false;
            }
        }
        return true;
    }
public:
    int splitArray(vector<int>& nums, int m) {
        int len = nums.size();
        //l and r represents the lower and upperbound of any 
        //array. For example: [1, 2, 3, 4, 5], the bound should
        //be [l, r] = [5, 15]
        long long l = 0, r = 0;
        for(int n : nums){
            l = max(l, (long long)n);
            r += n;
        }
        //Actually, from lowerbound to upperbound, we will have a
        //boolean array which represents wheather we can split 
        //the array to m-1 subarray if the given bound is from
        //[l, r]. For example, [1, 2, 3, 4, 5]. If m = 3 (2 cuts).
        //then we will have an array with [5, 6, ... 15] to search
        //we know [5:false, 6:true, 7:true...15:true].Then we need
        //to find the first true index in this array. We can apply
        //binary search here!
        while(l < r){
            int mid = l + (r - l) / 2;
            if(canSplit(nums, m-1, mid)) r = mid;
            else
                l = mid + 1;
        }
        return l;

    }
};

//DP solution: DP formula is the key to success
//dp[s,j] is the solution for splitting subarray n[j]...n[L-1] into 
//s parts.
//dp[s+1,i] = min{ max(dp[s,j], n[i]+...+n[j-1]) }, i+1 <= j <= L-s
class Solution {
public:
    int splitArray(vector<int>& nums, int m) {
        int len = nums.size();
        long long preSum[len+1] = {0};
        //dp array, we can do dp[s][i] here.
        //but dp[s][i] only depends on dp[s-1][j], we can compress
        //the result here
        long long dp[len] = {0};
        for(int i = 1; i <= len; ++i){
            preSum[i] = preSum[i-1] + nums[i-1];
        }
        for(int i = 0; i < len; ++i){
            //the first dp[i] means we have dp[0][i] for 
            //[i+1...len-1], which is simply the sum of i+1 to len-1
            dp[i] = preSum[len] - preSum[i];
        }
        //since we split s segments from [j..len-1], then we can 
        //have at most m-1 segments. (we will have one more segment
        //[i ... j-1])
        for(int s = 1; s < m; ++s){
            //we need to at least leave (s-1) elements for dp[s][j]
            //so we can partition it to s-1 subarrays
            for(int i = 0; i < len - s; ++i){
                dp[i] = LONG_MAX;
                for(int j = i+1; j <= len-s; ++j){
                    //preSum[j] - preSum[i]: [i .. j-1]
                    //dp[j]: dp[s-1][j]
                    //maximum potential split if we split in j 
                    //index
                    int t = max(preSum[j] - preSum[i], dp[j]); 
                    //since the array contains all positive numbers
                    //if t == dp[i], in the next loop, 
                    //preSum[j] - preSum[i] will be even larger, so
                    //we can break here!
                    //here we should have t <= dp[i] instead of < 
                    //dp[i]. The reason is dp[j] could be maximum
                    //and have dp[i] == dp[j], if we break earlier
                    //we lost the case that in the future roud, 
                    //we have preSum[j] - preSum[i] < dp[i]
                    if(t <= dp[i]) dp[i] = t;
                    else break;
                }
            }
        }
        //actually return dp[m][0]
        return dp[0];
    }
};


//DFS + Memorization: if you know how to implement DP solution
//Slow but easy to understand!
class Solution {
private:
    long long DFS(int pos, vector<int>& nums, int m, vector<vector<long long>>& memo, long long * preSum){
        if(m == 0) return preSum[nums.size()] - preSum[pos];
        if(pos >= nums.size() || m < 0) return LONG_MAX;
        if(memo[m][pos] != -1) return memo[m][pos];
        
        long long res = LONG_MAX;
        for(int i = pos; i < nums.size(); ++i){
            long long leftMax = preSum[i+1] - preSum[pos];
            //Note we have already cut 1 time here
            long long rightMax = DFS(i+1, nums, m-1, memo, preSum);
            res = min(res, max(leftMax, rightMax));
        }
        memo[m][pos] = res;
        return res;
    }
public:
    int splitArray(vector<int>& nums, int m) {
        int len = nums.size();
        long long preSum[len+1] = {0};
        for(int i = 0; i < len; ++i){
            preSum[i+1] = preSum[i] + nums[i];
        }
        //memo[s][j] means that for [j...len-1], if we can split 
        //m times, what is the minimum largest sum of different
        //partitions? s = [0 .. m-1]
        vector<vector<long long>> memo(m, vector<long long>(len, -1));
        //here m-1 means m-1 times cuts
        return DFS(0, nums, m-1, memo, preSum);
    }
};


//1007. Minimum Domino Rotations For Equal Row
//https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/
//First try! It's wrong!!!!! we need to count both swap from A to B and B to A.
//Using only one countA will not be sufficient! len - countA does not represent
//swap from A to B. Be careful!
//counter example: [1,2,2,1,1,2] [2,1,1,1,2,1]
/*
class Solution {
public:
    int minDominoRotations(vector<int>& A, vector<int>& B) {
        if(A.empty() || A.size() != B.size()) return -1;
        int len = A.size();
        int a = A[0], b = B[0];
        int countA = 0;
        int countB = 0;
        int i = 0;
        for(; i < len; ++i){
            if(A[i] != a && B[i] != a) {
                i = -1; break;
            }
            //Initial thought! it was wrong!!!!!
            if(A[i] != a && B[i] == a) countA++;
        }

        //it is possible that we swap len - countA times to formulate a
        //valid B array
        countA = i == -1 ? INT_MAX : (min(countA, len - countA));
        
        int j = 0;
        for(; j < len; ++j){
            if(A[j] != b && B[j] != b) {
                j = -1; break;
            }
            if(A[j] == b && B[j] != b) countB++;
        }
        countB = j == -1 ? INT_MAX : (min(countB, len - countB));
        if(i == -1 && j == -1) return -1;
        return min(countA, countB);
    }
};
*/

//A natural approach is to check x from 1... 6 and to see for each A[i],
//B[i], wheather either A[i] == x or B[i] == x. Actually we can do better,
//we can check A[0] and B[0] separately.  We know if either A or B satisfy 
//the requirement, then A[0] or B[0] must be included. We can reduce the
//search!
class Solution {
public:
    int minDominoRotations(vector<int>& A, vector<int>& B) {
        if(A.empty() || A.size() != B.size()) return -1;
        int len = A.size();
        int a = A[0], b = B[0];
        int countA = 0;
        int countB = 0;
        int i = 0;
        for(; i < len; ++i){
            if(A[i] != a && B[i] != a) {
                break;
            }
            //We need to count both A and B, which represents wheather we
            //swap B[i] to A[i] or swap A[i] to B[i]
            //Note it's not equivalent
            if(A[i] != a) countA++;
            if(B[i] != a) countB++;
            if(i == len-1) return min(countA, countB);
        }
        
        countA = countB = 0;
        int j = 0;
        for(; j < len; ++j){
            if(A[j] != b && B[j] != b) {
               break;
            }
            if(A[j] != b) countA++;
            if(B[j] != b) countB++;
            if(j == len-1) return min(countA, countB);
        }
        //If we do not terminate earlier, which means we cannot have valid
        //swap
        return -1;
    }
};


//973. K Closest Points to Origin
//https://leetcode.com/problems/k-closest-points-to-origin/
//O(nlogk)
class Solution {
private:
    long calEuclideanDist(vector<int>& p){
        return p[0] * p[0] + p[1] * p[1];
    }
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        int len = points.size();
        if(K >= len) return points;
        //pq will save the smallest element and its index!
        priority_queue<pair<long, int>> pq;
        for(int i = 0; i < len; ++i){
            long dist = calEuclideanDist(points[i]);
            pq.push(make_pair(dist, i));
            if(pq.size() > K) pq.pop();
        }
        vector<vector<int>> res;
        while(!pq.empty()){
            res.push_back(points[pq.top().second]);
            pq.pop();
        }
        return res;
    }
};


//partial_sort library
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        partial_sort(points.begin(), points.begin() + K, points.end(), [](vector<int>& a, vector<int>& b){ return a[0] * a[0] + a[1] * a[1] < b[0] * b[0] + b[1] * b[1];} );
        return vector<vector<int>>(points.begin(), points.begin() + K);
    }
};

//nth_element library!
//quick select, amortized O(n). Most efficient!
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        auto comp = [](vector<int>& a, vector<int>& b){
            return a[0]*a[0] + a[1]*a[1] < b[0]*b[0] + b[1]*b[1];
        };
        //quick select algorithm!
        nth_element(points.begin(), points.begin()+K, points.end(), comp);
        return vector<vector<int>>(points.begin(), points.begin()+K);
    }
};


//quick select. Implemented by me!
class Solution {
    //p0 will be the pivot point
    bool farther(vector<int>& p0, vector<int>& p1){
        return p0[0]*p0[0] + p0[1]*p0[1] < p1[0]*p1[0] + p1[1]*p1[1];
    }
    bool closer(vector<int>& p0, vector<int>& p1){
        return p0[0]*p0[0] + p0[1]*p0[1] > p1[0]*p1[0] + p1[1]*p1[1];
    } 
    //put elements smaller than p[index] before p[index], greater than 
    int partition(vector<vector<int>>& p, int l, int r){
        int index = l;
        l = l + 1;
        while(l <= r){
            if(farther(p[index], p[l]) && closer(p[index], p[r])){
                swap(p[l], p[r]);
                l++; r--;
                continue;
            }
            if(!farther(p[index], p[l])){
                l++;
            }
            if(!closer(p[index], p[r])){
                r--;
            }
        }
        swap(p[index], p[r]);
        return r;
    }
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        int l = 0, r = points.size() - 1;
        if(k > r) return points;
        while(l < r){
            int index = partition(points, l, r);
            //we find kth element
            if(index == k-1) break;
            //we have less than k elements in the left
            if(index < k-1)
                l = index + 1;
            else
                r = index;
        }
        return vector<vector<int>>(points.begin(), points.begin() + k);
    }
};


//475. Heaters
//https://leetcode.com/problems/heaters/
//Set solution!
//using set to store all the heaters.
class Solution {
public:
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        set<int> heaterSet(heaters.begin(), heaters.end());
        int miniRange = INT_MIN;
        for(int i = 0; i < houses.size(); ++i){
            int localMin = 0;
            int housePos = houses[i];
            auto it = heaterSet.lower_bound(housePos);
            if(it == heaterSet.begin()) 
                localMin = abs(*it - housePos);
            else{
                localMin = min(abs(*(--it) - housePos), abs(*it - housePos));
            }
            miniRange = max(miniRange, localMin);
            
        }
        return miniRange;
    }
};


/*
Example:    h = house,  * = heater  M = INT_MAX

        h   h   h   h   h   h   h   h   h    houses
        1   2   3   4   5   6   7   8   9    index
        *           *       *                heaters
                
        0   2   1   0   1   0   -   -   -    (distance to nearest RHS heater)
        0   1   2   0   1   0   1   2   3    (distance to nearest LHS heater)

        0   1   1   0   1   0   1   2   3    (res = minimum of above two)

Result is maximum value in res, which is 3.
*/
//Very good two pass cache solution! 
class Solution {
public:
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        int len = houses.size();
        sort(houses.begin(), houses.end());
        sort(heaters.begin(), heaters.end());
        vector<int> minimumRange(len, INT_MAX);
        for(int i = 0, j = 0; i < len && j < heaters.size();){
            if(houses[i] <= heaters[j]){
                minimumRange[i] = heaters[j] - houses[i];
                i++;
            }
            else
                j++;
        }
        for(int i = len-1, j = heaters.size()-1; i>=0 &&j >=0;){
            if(houses[i] >= heaters[j]){
                minimumRange[i] = min(houses[i] - heaters[j], minimumRange[i]);
                i--;
            }else
                j--;
        }
        
        return *max_element(minimumRange.begin(), minimumRange.end());
    }
};


//Binary search, with no auxiliary array
//O(nlogn) time. O(1) space.
class Solution {
public:
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        int res = 0, n = heaters.size();
        sort(heaters.begin(), heaters.end());
        for (int house : houses) {
            int left = 0, right = n;
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (heaters[mid] < house) left = mid + 1;
                else right = mid;
            }
            int dist1 = (right == n) ? INT_MAX : heaters[right] - house;
            int dist2 = (right == 0) ? INT_MAX : house - heaters[right - 1];
            res = max(res, min(dist1, dist2));
        }
        return res;
    }
};

//118. Pascal's Triangle
//https://leetcode.com/problems/pascals-triangle/
//Iterative version is pretty straightforward
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> res;
        for(int i = 0; i < numRows; i++){
            vector<int> row(i+1, 1);
            res.push_back(row);
            for(int j = 1; j < i; j++){
                res[i][j] = res[i-1][j-1] + res[i-1][j];
            }
        }
        return res;        
    }
};

//Recursive version. Note how we retrieve the nums-1 result and reuse it for
//current row computing
//Recursive version is a little bit tricky! Be careful when 
//write in this form
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        if(numRows <= 0) return vector<vector<int>>();
        //Base case
        if(numRows == 1) return vector<vector<int>>(1, vector<int>(1,1));
        auto tempRes = generate(numRows-1);
        const auto& preRow = tempRes.back();
        //Build current row;
        vector<int> curRow(1, 1);
        for(int i = 0; i < preRow.size()-1; ++i){
            curRow.push_back(preRow[i] + preRow[i+1]);
        }
        curRow.push_back(1);
        tempRes.emplace_back(curRow);
        return tempRes;
    }
};


//119. Pascal's Triangle II
//https://leetcode.com/problems/pascals-triangle-ii/
//Note how we save the space for this problem
class Solution {
public:
    vector<int> getRow(int rowIndex) {
        //Since row starts from 0
        vector<int> res(rowIndex+1, 0);
        res[0] = 1;
        for(int i = 0; i < rowIndex + 1; ++i){
            //For ith row, we at most need to update [1..index] elements
            for(int j = i; j >= 1; --j){
                res[j] += res[j-1];
            }
        }
        return res;
    }
};

//Recursive version is the same as before
class Solution {
public:
    vector<int> getRow(int rowIndex) {
        if(rowIndex < 0) return vector<int>();
        if(rowIndex == 0) return vector<int>(1, 1);
        auto tempRes = getRow(rowIndex-1);
        vector<int> curRes(rowIndex+1, 0);
        curRes[0] = 1;
        curRes.back() = 1;
        for(int i = 1; i < tempRes.size(); ++i){
            curRes[i] = tempRes[i-1] + tempRes[i];
        }
        return curRes;
    }
};


//912. Sort an Array
//https://leetcode.com/problems/sort-an-array/
//Wrong! first try!
/*
class Solution {
private:
    vector<int> mergeSort(vector<int>& nums, int l, int r){
        if(l > r) return vector<int>();
        if(l == r) return vector<int>(nums[l]);
        
        int mid = l + (r - l) / 2;
        vector<int> leftHalf = mergeSort(nums, l, mid);
        vector<int> rightHalf = mergeSort(nums, mid+1, r);
        if(!leftHalf.empty()) cout << leftHalf[0] << endl;
        vector<int> temp = merge(leftHalf, rightHalf);
        int j = 0;
        for(int i = l; i <= r; ++l)
            nums[l] = temp[j++];
        
        return nums;
    }
    
    vector<int> merge(vector<int>& n1, vector<int>& n2){
        vector<int> res;
        int len1 = n1.size(), len2 = n2.size();
        int i = 0, j = 0;
        for(; i < len1 && j < len2;){
            if(n1[i] <= n2[j]) {
                res.push_back(n1[i++]);
            }
            else res.push_back(n2[j++]);
        }
        while(i < len1) res.push_back(n1[i++]);
        while(j < len2) res.push_back(n2[j++]);
        return res;
    }
    
public:
    vector<int> sortArray(vector<int>& nums) {
        int len = nums.size();
        if(len <= 1) return nums;
        int l = 0, r = len-1;
        return mergeSort(nums, l, r);
    }
};
*/

//Merge sort algorithm!
//You spent too much time implementing this algorithm! Tricky, please be careful
class Solution {
private:
    void mergeSort(vector<int>& nums, int l, int r){
        if(l >= r) return;
        int mid = l + (r - l) / 2;
        mergeSort(nums, l, mid);
        mergeSort(nums, mid+1, r);
        merge(nums, l, mid, r);
    }
    void merge(vector<int>& nums, int l, int mid, int r){
        vector<int> res(r - l + 1, 0);
        int i = l, j = mid+1, ret = 0;
        while(i <= mid && j <= r){
            if(nums[i] <= nums[j]){
                res[ret++] = nums[i++];
            }else
                res[ret++] = nums[j++];
        }
        while(i <= mid) res[ret++] = nums[i++];
        while(j <= r) res[ret++] = nums[j++];
        i = 0, ret = 0;
        //should be < r-l+1
        for(; i < r - l + 1; ++i){
            nums[i + l] = res[ret++];
        }
    }
public:
    vector<int> sortArray(vector<int>& nums) {
        int len = nums.size();
        if(len <= 1) return nums;
        mergeSort(nums, 0, len-1);
        return nums;
    }
};

//Merge sort in-place! O(n^2logn) (not very useful!)
class Solution {
private:
    void mergeSort(vector<int>& nums, int l, int r){
        if(l >= r) return;
        int mid = l + (r - l) / 2;
        mergeSort(nums, l, mid);
        mergeSort(nums, mid+1, r);
        merge(nums, l, mid, r);
    }
    //In-place merge!
    void merge(vector<int>& nums, int l, int mid, int r){
        int s1 = l, s2 = mid + 1;
        //already sorted!
        if(nums[mid] <= nums[s2]) return; 
        
        while(s1 <= mid && s2 <= r){
            if(nums[s1] <= nums[s2])
                s1++;
            else{
                int index = s2;
                int val = nums[s2];
                //shift all the elements between s1 - s2 right by 1
                //leave room for val
                while(index != s1){
                    nums[index] = nums[index-1];
                    index--;
                }
                nums[s1] = val;
                
                s1++;
                s2++;
                //We need to update mid here, because s1 <= mid is
                //termination condition!
                mid++;
            }
        }
        
    }
public:
    vector<int> sortArray(vector<int>& nums) {
        int len = nums.size();
        if(len <= 1) return nums;
        mergeSort(nums, 0, len-1);
        return nums;
    }
};

//quick sort algorithm!
//Quick sort!
class Solution {
private:
    void quickSort(vector<int>& nums, int left, int right){
        int index = left;
        int l = left + 1, r = right;
        //Make sure we will have the return condition here!
        if(l > r) return;
        while(l <= r){
            if(nums[l] > nums[index] && nums[r] < nums[index]){
                swap(nums[l], nums[r]);
                l++; r--;
            }
            
            if(nums[l] <= nums[index])
                l++;
            
            if(nums[r] >= nums[index])
                r--;
        }
        swap(nums[index], nums[r]);
        //we already sort r
        int mid = r;
        quickSort(nums, left, mid-1);
        quickSort(nums, mid+1, right);
    }
public:
    vector<int> sortArray(vector<int>& nums) {
        int len = nums.size();
        int l = 0, r = len - 1;
        quickSort(nums, l, r);
        return nums;
    }
};



//218. The Skyline Problem
//https://leetcode.com/problems/the-skyline-problem/
//A very hard problem, the solution is tricky!
//The key idea is to keep track of each boundry, and record the highest height within at that point. For example:
//[2 9 10][3 7 15][5 12 12][15 20 10][19, 24 8]
//At each horizontal point, we are going to record the highest height and get a couple of pairs like below (we use multiset and multimap to do this):
//[2 10][3, 15][5,15][7 12][9 12][12 0][15 10][19 10][20 8][24 0]
//Then we remove all the consecutive pairs which have the same height (we only keep the first one). 
class Solution {
public:
    vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
        vector<vector<int>> res;
        int len = buildings.size();
        
        multimap<int, int> mMap;
        for(auto& v : buildings){
            //set second point with negative height, then we know
            //which point is the first
            mMap.emplace(v[0], v[2]);
            mMap.emplace(v[1], -v[2]);
        }
        
        //Build the x coordinate and height relationship
        //We need to insert 0 to heights as the base case! or when we have
        //0 heights, we cannot remove any elements in the set, will cause
        //problem!
        multiset<int> heights{0};
        //build the relationship of current x and current max height!
        //do not need multimap here, since the smae x will have the same
        //result. Save time and space
        map<int, int> xHMapping;
        for(auto& v : mMap){
            //When we encounter first point, we push the height!
            if(v.second > 0){
                heights.insert(v.second);
            }else{
                //cannot just say heights.erase(-v.second). we will erase
                //all the -v.second in heights
                heights.erase(heights.find(-v.second));
            }
            xHMapping[v.first] = (*heights.crbegin());
            
        }

        for(auto& v : xHMapping){
            if(res.empty() || res.back()[1] != v.second)
                res.push_back(vector<int>({v.first, v.second}));
        }
        return res;
    }
};


//A little bit optimization!
//This solution is more efficient, however it's a solution relies on sort rule...
//In general, it's bad!!

class Solution {
public:
    vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
        
        vector<pair<int, int>> dict;
        for(vector<int>& b:buildings){
            //We have to make the negative value go first, or the following if(cmaxHeight != preMaxHeight) will not work.
            dict.push_back({b[0], -b[2]});
            dict.push_back({b[1], b[2]});
        }
        //Sorting here is different compared with using multimap. For example, in multimap,
        //We will have sorting like [0 -3][2 3][2 -3][5 3]
        //Here, we will have [0 -3][2 -3][2 3][5 3]
        //It's critical when we try to get rid of duplicates from the array
        sort(dict.begin(), dict.end());
        //Note elements inserted to height are also sorted
        multiset<int> heights{0};
        int preMaxHeight = 0, cmaxHeight;
        vector<vector<int>> res;
        for(const pair<int, int>& p : dict){
            //we meet the first boundry
            if(p.second < 0){
                heights.insert(-p.second);
            }
            else 
                heights.erase(heights.find(p.second));
            //We update the max height for current position
            cmaxHeight = *heights.crbegin();
            //We keep track of the previous maxHeight, and get rid of the duplicate elments
            //Note this only works when we implement dict using vector and put negative height to our left boundry... Very weird...
            if(cmaxHeight != preMaxHeight){
                vector<int> t{p.first, cmaxHeight};
                res.push_back(t);
                preMaxHeight = cmaxHeight;
            }
        }
        return res;
    }
};



//442. Find All Duplicates in an Array
//https://leetcode.com/problems/find-all-duplicates-in-an-array/
//pingeon hole theory!
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> res;
        for(int i = 0; i < nums.size(); ){
            if(nums[i] == -1) {
                i++;
                continue;
            }
            if(nums[i] - 1 == i)
                i++;
            else if(nums[i] - 1 != i && nums[nums[i]-1] != nums[i]){
                swap(nums[i], nums[nums[i]-1]);
            }
            else if(nums[i] - 1 != i && nums[nums[i] - 1] == nums[i]){
                res.push_back(nums[i]);
                nums[i] = -1;
                i++;
            }
        }
        return res;
    }
};


//A more tricky implementation
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> res;
        for(int i = 0; i < nums.size(); ++i){
            //need absolute here
            int k = abs(nums[i]) - 1;
            //using negative number as a flag!
            if(nums[k] > 0){
                nums[k] = -1 *  nums[k];
            }
            else
                res.push_back(k+1);
        }
        return res;
    }
};


//406. Queue Reconstruction by Height
//https://leetcode.com/problems/queue-reconstruction-by-height/
/*
//https://leetcode.com/problems/queue-reconstruction-by-height/discuss/
Pick out tallest group of people and sort them in a subarray (S). Since there's no other groups of people taller than them, therefore each guy's index will be just as same as his k value.
For 2nd tallest group (and the rest), insert each one of them into (S) by k value. So on and so forth.

E.g.
input: [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
subarray after step 1: [[7,0], [7,1]]
subarray after step 2: [[7,0], [6,1], [7,1]]

Very tricky problem!

*/

class Solution {
public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        //Note we will put people with higher height and fewer k to the 
        //front area of our array
        //For the example input, after sort:
        //[[7,0], [7,1], [6,1], [5,0], [5,2], [4,4]]
        sort(people.begin(), people.end(), [](vector<int>& v1, vector<int>& v2){ return v1[0] > v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]); });
        //Note we cannot use list<vector<int>> here because list only 
        //supports bidirectional iterator, we cannot have list.begin()+k
        vector<vector<int>> res;
        for(auto& p : people){
            res.insert(res.begin() + p[1], p);
        }
        return res;
    }
};


//447. Number of Boomerangs
//https://leetcode.com/problems/number-of-boomerangs/
//straightforward O(n^2) solution!
class Solution {
public:
    int numberOfBoomerangs(vector<vector<int>>& points) {
        int len = points.size();
        if(len < 3) return 0;
        int res = 0;
        for(int i = 0; i < len; ++i){
            //map distance and number of elements with the distance
            unordered_map<long, int> uMap;
            for(int j = 0; j < len; ++j){
                if(j == i) continue;
                long distX = points[j][0] - points[i][0];
                long distY = points[j][1] - points[i][1];
                
                long finalDist = distX * distX + distY * distY;
                
                uMap[finalDist]++;
            }
            
            for(auto& it : uMap){
                //if we have more than one element with the same distance
                if(it.second > 1){
                    //Note we pick up 2 elements from n elements
                    //And the order matters!
                    res += (it.second) * (it.second-1);
                }
            }
        }
        return res;
    }
};



//448. Find All Numbers Disappeared in an Array
//https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int len = nums.size();
        vector<int> res;
        for(int i = 0; i < len; ++i){
            if(nums[i] == i + 1) continue;

            while(nums[i] != i+1 && nums[i] != nums[nums[i]-1]){
                swap(nums[i], nums[nums[i]-1]);
            }
            //if(nums[i] != i+1) res.push_back(i+1);
        }
        for(int i = 0; i < len; ++i){
            if(nums[i] != i+1) res.push_back(i+1);
        }
        return res;
    }
};


//435. Non-overlapping Intervals
//https://leetcode.com/problems/non-overlapping-intervals/
/*
//Original idea, which is wrong. 
//Cannot pass the test case like: [[2,4],[3,6],[7,8],[1,8]]
//The only part we are missing is to update previous interval!
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        if(intervals.empty()) return 0;
        auto comp = [](vector<int>& p1, vector<int>& p2){
            return p1[0] < p2[0] || (p1[0] == p2[0] && p1[1] < p2[1]);
        };
        sort(intervals.begin(), intervals.end(), comp);
        vector<int> tempV = intervals[0];
        int res = 0;
        for(int i = 1; i < intervals.size(); ++i){
            if(intervals[i][0] < tempV[1]){
                res++;
            }else{
                tempV = intervals[i];
            }
       }
        
       return res; 
    }   
};
*/
//Almost the same idea. The only thing we are missing is 
//if(intervals[i][1] < tempV[1]) tempV = intervals[i];
class Solution {
public:
    //Original idea about the problem!
    //Cannot pass the test case like: [[2,4],[3,6],[7,8],[1,8]]
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        if(intervals.empty()) return 0;
        auto comp = [](vector<int>& p1, vector<int>& p2){
            return p1[0] < p2[0];
        };
        sort(intervals.begin(), intervals.end(), comp);

        vector<int> tempV = intervals[0];
        int res = 0;
        for(int i = 1; i < intervals.size(); ++i){
            if(intervals[i][0] < tempV[1]){
                res++;
                //Note we need update tempV to be the overlapped interval
                //with larger right bound!
                if(intervals[i][1] < tempV[1]) tempV = intervals[i];
            }else{
                tempV = intervals[i];
            }
       }
        
       return res; 
    }
        
};


//452. Minimum Number of Arrows to Burst Balloons
//https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        if(points.empty()) return 0;
        //without comp, the default sorting is slow! with comp, 
        //much faster!
        auto comp = [](vector<int>& p1, vector<int>& p2){
            return p1[0] < p2[0];
        };
        sort(points.begin(), points.end(), comp);
        int pre = 0;
        int shot = 1;
        for(int i = 1; i < points.size(); ++i){
            if(points[i][0] > points[pre][1]){
                shot++;
                pre = i;
            }
            //keep track of the smaller right bound if 
            //points[i][0] <= points[pre][1]
            if(points[i][1] < points[pre][1])
                pre = i;
        }
        return shot;
    }
};



//436. Find Right Interval
//https://leetcode.com/problems/find-right-interval/
//Tree map solution!
class Solution {
public:
    vector<int> findRightInterval(vector<vector<int>>& intervals) {
        int len = intervals.size();
        //map the left boundry with interval index
        map<int, int> uMap;
        for(int i = 0; i < len; ++i){
            uMap[intervals[i][0]] = i;
        }
        
        vector<int> res;
        for(int i = 0; i < len; ++i){
            int rightBoundry = intervals[i][1];
            auto it = uMap.lower_bound(rightBoundry);
            if(it == uMap.end()) res.push_back(-1);
            else{
                res.push_back(it->second);
            }
        }
        return res;
    }
};


//454. 4Sum II
//https://leetcode.com/problems/4sum-ii/
//Not easy to get the idea!
class Solution {
public:
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        unordered_map<int, int> ABMap;
        int res = 0;
        for(int a : A){
            for(int b : B){
                ABMap[a + b] ++;
            }
        }
        //Note for any combination of AB, -c-d will contribute one possible
        //result
        for(int c : C){
            for(int d : D){
                res += ABMap[-c-d];
            }
        }
        
        return res;
    }
};



//453. Minimum Moves to Equal Array Elements
//https://leetcode.com/problems/minimum-moves-to-equal-array-elements/
class Solution {
public:
    int minMoves(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int res = 0;
        for(int i = nums.size()-1; i >= 1; --i){
            res += nums[i] - nums[0];
        }
        return res;
    }
};

/*Actually, we only need to calculate the diff between each element and
minimum element. Let's consider a sorted array, if we add num[i] - num[i-1]
to num[i-1], then num[i] == num[i-1]. Then we consider add num[i+1] - 
num[i-1] to num[i-1] and num[i], then we have num[i] == num[i+1] == num[i-1]. 
Keep going. Which means we only need to find the minimum element, then do
the calculation! */
class Solution {
public:
    int minMoves(vector<int>& nums) {
        int smallest = *min_element(nums.begin(), nums.end());
        int res = 0;
        for(int i = nums.size()-1; i >= 0; --i){
            res += nums[i] - smallest;
        }
        return res;
    }
};


//252. Meeting Rooms
//https://leetcode.com/problems/meeting-rooms/
class Solution {
public:
    bool canAttendMeetings(vector<vector<int>>& intervals) {
        auto comp = [](vector<int>& v1, vector<int>& v2){
            return v1[0] < v2[0];
        };
        sort(intervals.begin(), intervals.end(), comp);
        int len = intervals.size();
        for(int i = 1; i < len; ++i){
            if(intervals[i][0] < intervals[i-1][1]) return false;
        }
        return true;
    }
};



//253. Meeting Rooms II
//https://leetcode.com/problems/meeting-rooms-ii/
/* 
Good solution: Add tag at the beginning and the end of each interval. And
calculate the number of rooms needed accordingly. We keep track of the 
maximum room we need on the fly!
*/
class Solution {
public:
    int minMeetingRooms(vector<vector<int>>& intervals) {
        vector<pair<int, int>> timeSets;
        int len = intervals.size();
        for(int i = 0; i < len; ++i){
            timeSets.push_back({intervals[i][0], 1});
            timeSets.push_back({intervals[i][1], -1});
        }
        //when p1.first == p2.first, then p1.second must be positive 
        //in order to go beyond. customize comparator improves the 
        //efficiency
        auto comp = [](pair<int, int>& p1, pair<int, int>& p2){
            return p1.first < p2.first || (p1.first == p2.first && p1.second < p2.second);
        };
        sort(timeSets.begin(), timeSets.end(), comp);
        int res = 0;
        int curRoom = 0;
        for(int i = 0; i < timeSets.size(); ++i){
            curRoom += timeSets[i].second;
            res = max(res, curRoom);
        }
        return res;
    }
};

/* Priority queue implementation! */
class Solution {
public:
    int minMeetingRooms(vector<vector<int>>& intervals) {
        if(intervals.empty()) return 0;
        auto comp = [](vector<int>& v1, vector<int>& v2){
            return v1[0] < v2[0];
        };
        sort(intervals.begin(), intervals.end(), comp);
        //at least 1
        int res = 1;
        priority_queue<int, vector<int>, greater<int>> pq;
        //the end point of the first interval
        pq.push(intervals[0][1]);
        
        for(int i = 1; i < intervals.size(); ++i){
            while(!pq.empty() && intervals[i][0] >= pq.top()){
                pq.pop();
            }
            pq.push(intervals[i][1]);
            res = max(res, static_cast<int>(pq.size()));
        }
        return res;
    }
};



//477. Total Hamming Distance
//https://leetcode.com/problems/total-hamming-distance/
// standard
class Solution {
public:
    int totalHammingDistance(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        int res = 0;
        //calculate how many 1's and 0's for each bit. The total difference
        //will be countOne * countZero
        for(int i = 0; i < 32; ++i){
            int countOne = 0;
            for(int j = 0; j < len; ++j){
                countOne += ((nums[j] & (1 << i)) == 0) ? 0 : 1;
            }
            res += countOne * (len - countOne);
        }
        return res;
    }
};



//462. Minimum Moves to Equal Array Elements II
//https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/
//Find the median val
class Solution {
public:
    int minMoves2(vector<int>& nums) {
        if(nums.empty()) return 0;
        sort(nums.begin(), nums.end());
        int len = nums.size();
        int mid = len / 2;
        int val = nums[mid];
        int res = 0;
        for(int element : nums){
            res += abs(element - val);
        }
        return res;
    }
};

//O(n)
class Solution {
public:
    int minMoves2(vector<int>& nums) {
        if(nums.empty()) return 0;
        random_shuffle(nums.begin(), nums.end());
        int mid = nums.size() / 2;
        nth_element(nums.begin(), nums.begin() + mid, nums.end());
        int val = nums[mid];
        int res = 0;
        for(int element : nums){
            res += abs(element - val);
        }
        
        return res;
    }
};



//66. Plus One
//https://leetcode.com/problems/plus-one/
//Love the code
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        int len = digits.size();
        for(int i = len-1; i>=0; i--){
            if(digits[i] == 9)
                digits[i] = 0;
            else{
                digits[i]++;
                return digits;
            }    
        }
        digits[0] = 1;
        digits.push_back(0);
        return digits;
    }
};



//163. Missing Ranges
//https://leetcode.com/problems/missing-ranges/
/* Naive implementation! Note we need to handle test case */
class Solution {
public:
    vector<string> findMissingRanges(vector<int>& nums, int lower, int upper) {
        vector<string> res;
        int len = nums.size();
        if(len == 0){
            if(upper > lower)
                res.push_back(to_string(lower) + "->" + to_string(upper));
            else if(upper == lower)
                res.push_back(to_string(lower));
            return res;
        }
        
        if(lower != nums.front()){
            if(lower == nums.front()-1){
                res.push_back(to_string(lower));
            }else if(lower < nums.front()-1){
                res.push_back(to_string(lower) + "->" + to_string(nums.front()-1));
            }
        }

        
        for(int i = 1; i < len; ++i){
            if(nums[i] == nums[i-1] || nums[i] == nums[i-1]+1)
                continue;
            if(nums[i] == nums[i-1] + 2)
                res.push_back(to_string(nums[i-1]+1));
            else if(nums[i] > nums[i-1] + 2){
                res.push_back(to_string(nums[i-1]+1) + "->" + to_string(nums[i]-1));
            }
        }
        
        if(upper == nums.back()) return res;
        else if( upper == nums.back() + 1){
            res.push_back(to_string(upper));
        }else if(upper > nums.back() + 1){
            res.push_back(to_string(nums.back()+1) + "->" + to_string(upper));
        }
        
        return res;
    }
};

/* Other's code. He did not handle integer overflow. The code is simple and 
cocise. */
public class Solution {
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> list = new ArrayList<String>();
        for(int n : nums){
            int justBelow = n - 1;
            if(lower == justBelow) list.add(lower+"");
            else if(lower < justBelow) list.add(lower + "->" + justBelow);
            lower = n+1;
        }
        if(lower == upper) list.add(lower+"");
        else if(lower < upper) list.add(lower + "->" + upper);
        return list;
    }
}


//849. Maximize Distance to Closest Person
//https://leetcode.com/problems/maximize-distance-to-closest-person/
/* Consider 3 different intervals separately! */
class Solution {
public:
    int maxDistToClosest(vector<int>& seats) {
        int len = seats.size();
        int preFirst = 0;
        int afterLast = 0;
        int inBetween = 0;
        int i = 0, j = len-1;
        while(i < len && seats[i] == 0) i++;
        preFirst = i;
        while(j >= 0 && seats[j] == 0) j--;
        afterLast = len - j - 1;
        
        
        for(int k = i+1; k < j; ++k){
            if(seats[k] == 0){
                inBetween = max(inBetween, k - i);
            }else{
                i = k;
            }
        }
        
        return max(preFirst, max(afterLast, (inBetween + 1)/2));
    }
};

//Much elegant implementation. Calculate all three intervals on the fly!
int maxDistToClosest(vector<int> seats) {
    int res = 0, n = seats.size(), last = -1;
    for (int i = 0; i < n; ++i) {
        if (seats[i] == 1) {
            res = last < 0 ? i : max(res, (i - last) / 2);
            last = i;
        }
    }
    res = max(res, n - last - 1);
    return res;
}


//135. Candy
//https://leetcode.com/problems/candy/
//Initial idea: priority queue! However, slow.
class Solution {
public:
    int candy(vector<int>& ratings) {
        if(ratings.empty()) return 0;
        int len = ratings.size();
        if(len == 1) return 1;
        
        priority_queue<pair<int, int>, vector<pair<int,int>>, greater<pair<int, int>>> pq;
        int candyList[len] = {0};
        for(int i = 0; i < len; ++i){
            pq.push({ratings[i], i});
        }
        //At least 2 elements here
        while(!pq.empty()){
            auto p = pq.top();
            pq.pop();
            int leftNeighbour = p.second == 0 ? 0 : p.second - 1;
            int rightNeighbour = p.second == len-1 ? len-1 : p.second+1;
            int numCandy = 1;
            
            if(p.first <= ratings[leftNeighbour] && p.first <= ratings[rightNeighbour]){
                candyList[p.second] = numCandy;
                continue;
            }

            if(p.first > ratings[leftNeighbour])
                numCandy = max(numCandy, candyList[leftNeighbour] + 1);
            
            if(p.first > ratings[rightNeighbour])
                numCandy = max(numCandy, candyList[rightNeighbour] + 1);
            
            candyList[p.second] = numCandy;
        }
        
        int res = 0;
        for(int n : candyList){
            //cout << n << endl;
            res += n;
        }
        return res;
        
    }
};


//Actually, we can do 3 passes. Much elegant!
class Solution {
public:
    int candy(vector<int>& ratings) {
        int len = ratings.size();
        if(len == 0) return 0;
        vector<int> candy(len, 1);
        int sum = 0;
        for(int i = 1; i < len; i++){
            if(ratings[i] > ratings[i-1]){
                candy[i] = candy[i-1] + 1;
            }
        }
        for(int j = len-2; j >= 0; j--){
            if(ratings[j] > ratings[j+1])
                candy[j] = max(candy[j], candy[j+1]+1);
        }
        for(int num : candy){
            sum += num;
        }
        return sum;
    }
};


// Space O(1) solution. Clever, but not very useful!
// Hard to get and implement.
class Solution {
public:
    int count(int n){
        return (n*(n+1))/2;
    }
    int candy(vector<int>& ratings) {
        int len = ratings.size();
        if(len == 0) return 0;
        int up = 0, down = 0, oldSlope = 0;
        int candy = 0;
        for(int i = 1; i < len; i++){
            int nSlope = (ratings[i]>ratings[i-1]) ? 1: (ratings[i]<ratings[i-1] ? -1 : 0);
            if((oldSlope>0 && nSlope == 0) || (oldSlope<0 && nSlope >= 0)){
                candy += count(up) + count(down) + max(up, down);
                //cout << candy << " ";
                up = 0;
                down = 0;
            }
            if(nSlope > 0) up++;
            if(nSlope < 0) down++;
            if(nSlope == 0) candy++;
            
            oldSlope = nSlope;
        }
        candy += count(up) + count(down) + max(up, down) + 1;
        return candy;
    }
};


//845. Longest Mountain in Array
//https://leetcode.com/problems/longest-mountain-in-array/
//One pass, track everything
class Solution {
public:
    int longestMountain(vector<int>& A) {
        if(A.size() < 3) return 0;
        int res = 0;
        int len = A.size();
        int i = 1;
        while(i < A.size()){
            int tempUp = 0;
            int tempDown = 0;
            while(i < A.size() && A[i] == A[i-1]) i++;
            while(i < A.size() && A[i] > A[i-1]) {
                tempUp ++;
                i++;
            }
            while(i < A.size() && A[i] < A[i-1]){
                tempDown++;
                i++;
            }
            if(tempUp > 0 && tempDown > 0){
                res = max(res, 1 + tempUp + tempDown);
            }
        }
        return res;
    }
};


//1095. Find in Mountain Array
//https://leetcode.com/problems/find-in-mountain-array/
/**
 * // This is the MountainArray's API interface.
 * // You should not implement it, or speculate about its implementation
 * class MountainArray {
 *   public:
 *     int get(int index);
 *     int length();
 * };
 */
class Solution {
public:
    int findInMountainArray(int target, MountainArray &mountainArr) {
        int len = mountainArr.length();
        int l = 0;
        int r = len-1;
        int peak = 0;
        //find the peak. Binary search.
        while(l < r){
            int m = l + (r - l) / 2;
            if(mountainArr.get(m) > mountainArr.get(m-1)){
                l = m + 1;
            }else{
                r = m;
            }
        }
        peak = l;
        
        //search left
        l = 0;
        r = peak;
        while(l <= r){
            int m = l + (r - l) / 2;
            if(mountainArr.get(m) == target) return m;
            else if (mountainArr.get(m) < target) l = m+1;
            else r = m - 1;
        }
        
        //search right
        l = peak;
        r = len-1;
        while(l <= r){
            int m = l + (r - l) / 2;
            if(mountainArr.get(m) == target) return m;
            else if (mountainArr.get(m) > target) l = m+1;
            else r = m - 1;
        }
        
        return -1;
    }
};



//315. Count of Smaller Numbers After Self
//https://leetcode.com/problems/count-of-smaller-numbers-after-self
//Very elegant solution, I am afraid I cannot get it right during the interview!
//O(nlogn)
class Solution {
private:
    void sort_count(vector<pair<int, int>>::iterator l, 
                           vector<pair<int, int>>::iterator r, 
                           vector<int>& cnt){
        if(l + 1 >= r) return;
        auto m = l + (r - l) / 2;
        //[l, m)
        //Now [l,m) [m, r) have already been sorted
        sort_count(l, m, cnt);
        sort_count(m, r, cnt);
        
        for(auto i = l, j = m; i < m; ++i){
            while(j < r && ((*i).first > (*j).first)) j++;
            //Note we need to include j-m, not j-i, since i..m-1 has already been included
            cnt[(*i).second] += (j - m);
        }
        //implce_merge(l, m, r) is easy to use, it will merge the array in place. given 3
        //boundries
        inplace_merge(l, m, r);
        
    }
public:
    vector<int> countSmaller(vector<int>& nums) {
        int len = nums.size(); 
        if(len == 0) return vector<int>();
        //Allocate sufficient space here
        vector<int> count(len, 0);
        vector<pair<int, int>> hold;
        for(int i = 0; i < len; ++i){
            hold.push_back({nums[i], i});
        }
        
        sort_count(hold.begin(), hold.end(), count);
        return count;
    }
};

//Binary search tree solution. Not implemented by me!
struct node{
    int val,copy,leftCnt;
    node *left,*right;
    node(int x){val=x;copy=1;leftCnt=0;left=NULL;right=NULL;}
};

int insert(node* root,int x){
    if (root->val==x){
        root->copy++;
        return root->leftCnt;
    }else if (root->val>x){
        root->leftCnt++;
        if (root->left==NULL) {
            root->left = new node(x);
            return 0;
        }else  return insert(root->left,x);
    }else{
        if (root->right==NULL){
            root->right = new node(x);
            return root->leftCnt+root->copy;
        }else return root->leftCnt+root->copy+insert(root->right,x);
    }
}

class Solution {
public:
    vector<int> countSmaller(vector<int>& nums) {
        int sz=nums.size();
        vector<int> res(sz,0);
        if (sz<=1) return res;
        node *root = new node(nums[sz-1]);
        for (int i=sz-2;i>=0;i--){
            res[i] = insert(root,nums[i]);
        }
        return res;
    }
};


//739. Daily Temperatures
//https://leetcode.com/problems/daily-temperatures/
//Map solution: implemented by me O(nlogn)
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        map<int, int> Map;
        int len = T.size();
        vector<int> res;
        if(len == 0) return res;
        for(int i = len-1; i >= 0; --i){
            int temp = T[i];
            auto it = Map.upper_bound(temp);
            if(it == Map.end()) res.push_back(0);
            else{
                res.push_back(it->second - i);
            }
            //cannot reuse it here
            while(!Map.empty() && Map.upper_bound(temp)!= Map.begin()){
                int key = (--Map.upper_bound(temp))->first;
                Map.erase(key); 
            }
            Map[temp] = i;
        }
        reverse(res.begin(), res.end());
        return res;
    }
};

//Using stack. Same idea as above. There is no need to maintain a map!
//We can also utilize the fact that T[i] is [30 - 100]. So we can do the 
//calculation one by one, 30, 31, 32 ... 100 etc.
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        stack<int> st;
        int len = T.size();
        if(len == 0) return vector<int>();
        vector<int> res(len, 0);
        
        for(int i = len-1; i >= 0; --i){
            int temp = T[i];
            //<= to handle duplicate numbers
            while(!st.empty() && T[st.top()] <= temp)
                st.pop();
            int next = st.empty() ? i : st.top();
            res[i] = next - i;
            st.push(i);
        }
        return res;
    }
};


//581. Shortest Unsorted Continuous Subarray
//https://leetcode.com/problems/shortest-unsorted-continuous-subarray/
//Sorting: O(nlogn) I get this idea
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        vector<int> nums_sorted = nums;
        sort(nums_sorted.begin(), nums_sorted.end());
        int l = 0, r = nums.size()-1;
        int len = nums.size();
        while(l < len && nums[l] == nums_sorted[l])
            l++;
        while(r >= l && nums[r] == nums_sorted[r])
            r--;
        
        return r - l + 1;
    }
};


//O(n) with stack. Not easy to get!
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        int len = nums.size();
        int startL = len-1, startR = 0;
        //We need to use the stack to keep track of the boundry of left and
        //right. we need to find the minimum elements from the unsorted array
        //which determine the leftmost boundry, and maximum element which 
        //determines the rightmost boundry.
        stack<int> st;
        for(int i = 0; i < len; ++i){
            while(!st.empty() && nums[st.top()] > nums[i]){
                startL = min(startL, st.top());
                st.pop();
            }
            st.push(i);
        }
        
        st = stack<int>();
        
        for(int i = len-1; i >= 0; --i){
            while(!st.empty() && nums[st.top()] < nums[i]){
                startR = max(startR, st.top());
                st.pop();
            }
            st.push(i);
        }
        
        return startR - startL <= 0 ? 0 : startR - startL + 1;
    }
};


//O(n) time, O(1) space. Hard to get it right.
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        int len = nums.size();
        int startL = len-1, startR = 0;
        int minElement = INT_MAX, maxElement = INT_MIN;
         
        bool flag = false;
        for(int i = 1; i < len; ++i){
            if(nums[i] < nums[i-1]) flag = true;
            if(flag) minElement = min(minElement, nums[i]);
        }
        
        flag = false;
        for(int i = len-2; i >= 0; --i){
            if(nums[i] > nums[i+1]) flag = true;
            if(flag) maxElement = max(maxElement, nums[i]);
        }
        
        for(int i = 0; i < len; ++i){
            if(nums[i] > minElement) {
                startL = i;
                break;
            }
        }
        
        for(int i = len-1; i >= 0; --i){
            if(nums[i] < maxElement){
                startR = i;
                break;
            }
        }
        
        return startR - startL <= 0 ? 0 : startR - startL + 1;
    }
};



//560. Subarray Sum Equals K
//https://leetcode.com/problems/subarray-sum-equals-k/
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int len = nums.size();
        int sum = 0;
        int cnt = 0;
        for(int i = 0; i < len; ++i){
            sum = 0;
            for(int j = i; j < len; ++j){
                sum += nums[j];
                if(sum == k) cnt++;
            }
        }
        return cnt;
    }
};


//A very clever solution and hard to get it right!
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int len = nums.size();
        unordered_map<int, int> uMap;
        //The base case, when the sum-k == 0, we need to add 1 to the cnt
        uMap[0] = 1;
        int cnt = 0, sum = 0;
        for(int i = 0; i < len; ++i){
            sum += nums[i];
            //if we have encountered sum - k before, we need to include it
            //to our final cnt
            if(uMap.count(sum - k) > 0)
                cnt += uMap[sum-k];
            uMap[sum] ++;
        }
        return cnt;
    }
};



//338. Counting Bits
//https://leetcode.com/problems/counting-bits/
class Solution {
private:
    int popBits(int x){
        int cnt = 0;
        while(x != 0){
            //x &= x-1, it will flip the lowest 1 bit to 0
            //Keep in mind!
            x &= x-1;
            cnt++;
        }
        return cnt;
    }
public:
    vector<int> countBits(int num) {
        vector<int> res(num+1, 0);
        for(int i = 1; i <= num; ++i){
            int bit = popBits(i);
            res[i] = bit;
        }
        return res;
    }
};


//DP solution: https://leetcode.com/articles/counting-bits/
//Excellent solution. Especially the last DP
//DP 1:
//dp[x + b] = dp[x] + 1. b = 2^k
//0 - 1, 01 - 23, 0123 - 4567, .... 
class Solution {
public:
    vector<int> countBits(int num) {
        int i = 0, b = 1;
        vector<int> dp(num+1, 0);
        while(b <= num){
            while(i <= num && i + b <= num){
                dp[i+b] = dp[i] + 1;
                i++;
            }
            i = 0;
            //doubles b here!
            b = b << 1;
        }
        return dp;
    }
};

//DP 2:
//dp[x] = dp[x/2] + x&1. Get rid of the least significant bit
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> res(num+1, 0);
        for(int i = 1; i <= num; ++i){
            res[i] = res[i/2] + (i & 1);
        }
        return res;
    }
};

//DP 3:
//dp[x] = dp[(x & (x-1))] + 1. Flip the least significant 1
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> res(num+1, 0);
        for(int i = 1; i <= num; ++i){
            res[i] = res[(i & (i-1))] + 1;
        }
        return res;
    }
};



//621. Task Scheduler
//https://leetcode.com/problems/task-scheduler/
//Tricky problem: https://leetcode.com/articles/task-scheduler/
//Excellent simulation idea! not easy to get it!
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        vector<int> dict(26, 0);
        for(char c : tasks){
            dict[c - 'A'] ++;
        }
        sort(dict.begin(), dict.end());
        int res = 0;
        //Keep sorting, the idea is beautiful!
        while(dict[25] != 0){
            int i = 0;
            while(i <= n){
                //when we add last task to CPU and find we no longer have
                //any tasks left
                if(dict[25] == 0) break;
                if(i < 26 && dict[25 - i] > 0)
                    dict[25-i] --;
                i++;
                res++;
            }
            sort(dict.begin(), dict.end());
        }
        
        return res;
    }
};


//priority_queue
//Excellent simulation idea! not easy to get it!
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        int res = 0;
        vector<int> dict(26, 0);
        for(char c : tasks){
            dict[c - 'A'] ++;
        }
        
        priority_queue<int> pq;
        for(int n : dict){
            if(n > 0) pq.push(n);
        }
        
        while(!pq.empty()){
            vector<int> vec;
            int i = 0;
            while(i <= n){
                //we only care about whose frequency
                if(!pq.empty()){
                    if(pq.top() > 1){
                        int freq = pq.top() - 1;
                        vec.push_back(freq);
                        pq.pop();
                    }else{
                        pq.pop();
                    }
                }
                i++;
                res++;
                if(pq.empty() && vec.empty()) break;
            }
            for(int k : vec) pq.push(k);
        }
        
        return res;
        
    }
};

//Finding the idle time first algorithm! Very elegant! Most efficient.
//Hard to get!
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        int len = tasks.size();
        vector<int> dict(26, 0);
        for(char c : tasks){
            dict[c-'A'] ++;
        }
        sort(dict.begin(), dict.end());
        //We need to calculate the empty intervals we need to add to the 
        //final result! So we need to make maxFreq -1. Look at below:
        /*
        A B C - - -
        A B C - - -
        A B - - - -
        A B
        */
        int maxFreq = dict[25]-1;
        int idleTimeSlots = maxFreq * n;
        //We need to start from 24, since n is the gap between two duplicate
        //tasks
        for(int i = 24; i >= 0 && dict[i] > 0; --i){
            idleTimeSlots -= min(dict[i], maxFreq);
        }
        return idleTimeSlots > 0 ? idleTimeSlots + len : len;
    }
};


//277. Find the Celebrity
//https://leetcode.com/problems/find-the-celebrity/
// Forward declaration of the knows API.
bool knows(int a, int b);

class Solution {
public:
    int findCelebrity(int n) {
        if(n < 0) return -1;
        if(n == 1) return 0;
        
        stack<int> st;
        for(int i = 0; i < n; ++i){
            st.push(i);
        }
        
        //Each iteration, we eliminate the one of the two persons from our
        //pool
        while(st.size() > 1){
            int a = st.top();
            st.pop();
            int b = st.top();
            st.pop();
            if(knows(a, b))
                st.push(b);
            else
                st.push(a);
        }
        
        //check the last one
        int c = st.top();
        for(int i = 0; i < n; ++i){
            if(i != c && (knows(c, i) || !knows(i, c)))
                return -1;
        }
        return c;
    }
};

//A much concise implementation. However, a little bit hard to reasoning.
bool knows(int a, int b);

class Solution {

public:

    int findCelebrity(int n) {
        if(n<=1) return n;
        
        int candidate = 0;
        
        for(int i=1; i<n; i++){
            
            if ( !knows(i,candidate) ){
                candidate = i;
            }
        }
        
    
        for(int j=0; j<n; j++){
            
            if(j== candidate) continue;
        
            if( !knows(j,candidate) || knows(candidate,j) ){
                //if j does not know candidate, or candidate knows j, return -1;
                return -1;
            }
    
        }
        return candidate;   
    }
};

