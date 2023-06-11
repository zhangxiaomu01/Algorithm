/**
 * @file Backtracking.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2022-06-09
 * 
 * A quick second pass with the common 'Backtracking' algorithm problems.
 * 
 */

 /*
    698. Partition to K Equal Sum Subsets
    https://leetcode.com/problems/partition-to-k-equal-sum-subsets/
 
    Given an integer array nums and an integer k, return true if it is possible to divide 
    this array into k non-empty subsets whose sums are all equal.


    Example 1:
    Input: nums = [4,3,2,3,5,2,1], k = 4
    Output: true
    Explanation: It is possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with
    equal sums.

    Example 2:
    Input: nums = [1,2,3,4], k = 3
    Output: false
    

    Constraints:
    1 <= k <= nums.length <= 16
    1 <= nums[i] <= 104
    The frequency of each element is in the range [1, 4].
 */
// Solution 01: backtracking search! Will get TLE on Leetcode.
class Solution {
public:
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int sum = 0;
        sum = accumulate(nums.begin(), nums.end(), sum);
        if (nums.size() < k || sum % k) return false;
        
        vector<int>visited(nums.size(), false);
        sort(begin(nums),end(nums),greater<int>());// For avoid extra calculation

        return backtrack(nums, visited, sum / k, 0, 0, k);
    }
    
    bool backtrack(vector<int>& nums,vector<int>visited, int target, int curr_sum, int i, int k) {
        if (k == 1) 
            return true;
        
        if (curr_sum == target) 
            return backtrack(nums, visited, target, 0, 0, k-1);
        
        for (int j = i; j < nums.size(); j++) {
            if (visited[j] || curr_sum + nums[j] > target) continue;
            
            visited[j] = true;
            if (backtrack(nums, visited, target, curr_sum + nums[j], j+1, k)) return true;
            visited[j] = false;
        }
        
        return false;
    }
};

// Bitmask + extensive search O(n*2^n)
// Hard to implement.
class Solution {    
public:
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int n = nums.size();
        int sum = 0;
        for (int e : nums) {
            sum += e;
        }
        if (n < k || sum % k != 0) return false;

        int targetSum = sum / k;

        /*
        dp[i] represents if we selected 'x' elements from nums, and the remainder of the current sum % targetSum.
        The 'x' is presented by 16 bits, like '100...' means that the first element is selected.
        In the end, if dp[(1<<n) - 1] ('111111...') is 0, which means we can find k sets which have the sum equals
        to targetSum.
        */
        int dp[1<<17]; 
        fill(dp, dp + (1<<17), -1);
        int mask = 1 << n;

        dp[0] = 0;

        for (int i = 0; i < mask; i++) {
            if (dp[i] == -1) continue; // Remove the unintialized state.
            for(int j = 0; j < n; j++) {
                // i & (1 << j) is used to test whether we have updated the combination of selecting current 
                // element.
                if (!(i&(1 << j)) && dp[i] + nums[j] <= targetSum) {
                    dp[i|(1 << j)] = (dp[i] + nums[j]) % targetSum;
                }
            }
        } 
        return  dp[(1 << n) - 1] == 0;
    }
};

 /*
    77. Combinations
    https://leetcode.com/problems/combinations/
 
   Given two integers n and k, return all possible combinations of k numbers chosen from 
   the range [1, n].

    You may return the answer in any order.

    

    Example 1:
    Input: n = 4, k = 2
    Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    Explanation: There are 4 choose 2 = 6 total combinations.
    Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to be the 
    same combination.

    Example 2:
    Input: n = 1, k = 1
    Output: [[1]]
    Explanation: There is 1 choose 1 = 1 total combination.
    

    Constraints:
        1 <= n <= 20
        1 <= k <= n
 */
// Backtracking
class Solution {
private:
    void backtracking(vector<vector<int>>& res, vector<int>& path, int k, int n, int next) {
        if (path.size() == k) {
            res.push_back(path);
            return;
        }

         // Note the optimization： we trim all the impossible paths if the rest of the elements
         // is not suffcient to make path size >= k.
        for (int i = next; i <= n - (k - path.size()) + 1; ++i) {
            path.push_back(i);
            backtracking(res, path, k, n, i+1);
            path.pop_back();
        }
    }
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> path;
        backtracking(res, path, k, n, 1);
        return res;
    }
}; 


 /*
    216. Combination Sum III
    https://leetcode.com/problems/combination-sum-iii/
    Find all valid combinations of k numbers that sum up to n such that the following conditions 
    are true:

    Only numbers 1 through 9 are used.
    Each number is used at most once.
    Return a list of all possible valid combinations. The list must not contain the same 
    combination twice, and the combinations may be returned in any order.

    

    Example 1:
    Input: k = 3, n = 7
    Output: [[1,2,4]]
    Explanation:
    1 + 2 + 4 = 7
    There are no other valid combinations.

    Example 2:
    Input: k = 3, n = 9
    Output: [[1,2,6],[1,3,5],[2,3,4]]
    Explanation:
    1 + 2 + 6 = 9
    1 + 3 + 5 = 9
    2 + 3 + 4 = 9
    There are no other valid combinations.

    Example 3:
    Input: k = 4, n = 1
    Output: []
    Explanation: There are no valid combinations.
    Using 4 different numbers in the range [1,9], the smallest sum we can get is 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.
    

    Constraints:
    2 <= k <= 9
    1 <= n <= 60
 */
// Backtracking
class Solution {
private:
    void backtracking(vector<vector<int>>& res, vector<int>& combo, int k, int n, int next) {
        if (combo.size() == k && n == 0) {
            res.push_back(combo);
            return;
        }

        // Trim
        for (int i = next; i <= 9 - (k - combo.size()) + 1; ++i) {
            if (n - i >= 0) {
                combo.push_back(i);
                backtracking(res, combo, k, n - i, i + 1);
                combo.pop_back();
            } else {
                break;
            }
        }
    }

public:
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> res;
        vector<int> combo;

        backtracking(res, combo, k, n, 1);
        return res;
    }
};

 /*
    17. Letter Combinations of a Phone Number
    https://leetcode.com/problems/letter-combinations-of-a-phone-number/
    Given a string containing digits from 2-9 inclusive, return all possible letter combinations
    that the number could represent. Return the answer in any order.

    A mapping of digits to letters (just like on the telephone buttons) is given below.
     Note that 1 does not map to any letters.


    Example 1:
    Input: digits = "23"
    Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

    Example 2:
    Input: digits = ""
    Output: []

    Example 3:
    Input: digits = "2"
    Output: ["a","b","c"]
    

    Constraints:
    0 <= digits.length <= 4
    digits[i] is a digit in the range ['2', '9'].
 */
class Solution {
private:
    const string letterMap[10] = {
        "", // 0
        "", // 1
        "abc", // 2
        "def", // 3
        "ghi", // 4
        "jkl", // 5
        "mno", // 6
        "pqrs", // 7
        "tuv", // 8
        "wxyz", // 9
    };

    void backtracking(vector<string>& res, string& combo, string& digits, int next) {
        if (next == digits.size()) {
            if (!combo.empty()) {
                res.push_back(combo);
            }
            return;
        }

        int index = digits[next] - '0';
        for (int i = 0; i < letterMap[index].size(); ++i) {
            combo.push_back(letterMap[index][i]);
            backtracking(res, combo, digits, next + 1);
            combo.pop_back();
        }
    }
public:
    vector<string> letterCombinations(string digits) {
        vector<string> res;
        string combo;
        backtracking(res, combo, digits, 0);
        return res;
    }
};

