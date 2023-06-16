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


 /*
    39. Combination Sum
    https://leetcode.com/problems/combination-sum/
    Given an array of distinct integers candidates and a target integer target, return a
    list of all unique combinations of candidates where the chosen numbers sum to target.
    You may return the combinations in any order.

    The same number may be chosen from candidates an unlimited number of times. Two combinations 
    are unique if the frequency of at least one of the chosen numbers is different.

    The test cases are generated such that the number of unique combinations that sum up to 
    target is less than 150 combinations for the given input.

    

    Example 1:
    Input: candidates = [2,3,6,7], target = 7
    Output: [[2,2,3],[7]]
    Explanation:
    2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
    7 is a candidate, and 7 = 7.
    These are the only two combinations.

    Example 2:
    Input: candidates = [2,3,5], target = 8
    Output: [[2,2,2,2],[2,3,3],[3,5]]

    Example 3:
    Input: candidates = [2], target = 1
    Output: []
    

    Constraints:
    1 <= candidates.length <= 30
    2 <= candidates[i] <= 40
    All elements of candidates are distinct.
    1 <= target <= 40
 */
class Solution {
private:
    void backtracking(vector<vector<int>>& res, vector<int>& combo, vector<int>& candidates, int sum, int target, int next) {
        if (sum > target) return;

        if (sum == target) {
            res.push_back(combo);
            return;
        }

        for (int i = next; i < candidates.size(); ++i) {
            combo.push_back(candidates[i]);
            backtracking(res, combo, candidates, sum + candidates[i], target, i);
            combo.pop_back();
        }
    }
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> combo;
        backtracking(res, combo, candidates, 0, target, 0);
        return res;
    }
};

 /*
    40. Combination Sum II
    https://leetcode.com/problems/combination-sum-ii/
    Given a collection of candidate numbers (candidates) and a target number (target),
    find all unique combinations in candidates where the candidate numbers sum to target.

    Each number in candidates may only be used once in the combination.

    Note: The solution set must not contain duplicate combinations.

    

    Example 1:
    Input: candidates = [10,1,2,7,6,1,5], target = 8
    Output: 
    [
    [1,1,6],
    [1,2,5],
    [1,7],
    [2,6]
    ]

    Example 2:
    Input: candidates = [2,5,2,1,2], target = 5
    Output: 
    [
    [1,2,2],
    [5]
    ]
    

    Constraints:
    1 <= candidates.length <= 100
    1 <= candidates[i] <= 50
    1 <= target <= 30
 */
class Solution {
private:
    void backtracking(vector<vector<int>>& res, vector<int>& candidates, vector<int>& combo, int target, int sum, int next) {
        if (sum > target) return;
        if (sum == target) {
            res.push_back(combo);
            return;
        }

        for (int i = next; i < candidates.size(); ++i) {
            // Note we need to skip the duplicates here. We will only skip duplicates in the level
            // order of the traversal tree.
            if (i > next && candidates[i] == candidates[i-1]) continue;
            if (sum + candidates[i] <= target) {
                combo.push_back(candidates[i]);
                backtracking(res, candidates, combo, target, sum + candidates[i], i + 1);
                combo.pop_back();
            }
        }

    }
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> res;
        vector<int> combo;
        backtracking(res, candidates, combo, target, 0, 0);
        return res;
    }
};

 /*
    131. Palindrome Partitioning
    https://leetcode.com/problems/palindrome-partitioning/
    Given a string s, partition s such that every 
    substring of the partition is a palindrome.
    . Return all possible palindrome partitioning of s.

    

    Example 1:
    Input: s = "aab"
    Output: [["a","a","b"],["aa","b"]]

    Example 2:
    Input: s = "a"
    Output: [["a"]]
    

    Constraints:
    1 <= s.length <= 16
    s contains only lowercase English letters.
 */
class Solution {
private:
    bool isPalindrome(string& s) {
        if (s.size() <= 1) return true;
        int l = 0, r = s.size() - 1;
        while (l < r) {
            if (s[l++] != s[r--]) return false; 
        }
        return true;
    }
    void backtracking(string& s, vector<string>& combo, vector<vector<string>>& res, int next) {
        if (next > s.size()) return;
        if (next == s.size()) {
            res.push_back(combo);
            return;
        }
        string temp = "";

        for (int i = next; i < s.size(); ++i) {
            temp.push_back(s[i]);
            if (isPalindrome(temp)) {
                combo.push_back(temp);
                backtracking(s, combo, res, i + 1);
                combo.pop_back();
            }
        }
    }
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> res;
        vector<string> combo;
        backtracking(s, combo, res, 0);
        return res;
    }
};

// Slightly optimized the palindrome calculation.
class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> res;
        vector<string> temp;
        int len = s.size();
        if(len == 0) return res;
        vector<vector<int>> dp(len, vector<int>(len, 0));
        for(int i = 0; i < len; i++){
            for(int j = 0; j <= i; j++){
                //Note dp[j+1][i-1] means the sequence from j+1 to i-1.
                if(s[i] == s[j] && (i-j <= 2 || dp[j+1][i-1] == 1))
                    dp[j][i] = 1;
            }
        }
        //cout<<dp[1][4] << " " << dp[2][3];
        rec(s, res, temp, dp, 0, len);
        return res;
        
        
    }
    void rec(string& s, vector<vector<string>>& res, vector<string>& temp, vector<vector<int>>& dp, int start, int len){
        if(start == len){
            res.push_back(temp);
            return;
        }
        for(int i=start; i<len; i++){
            if(dp[start][i] == 1){
                temp.push_back(s.substr(start, i-start+1));
                rec(s, res, temp, dp, i+1, len);
                temp.pop_back();
            }
        }  
        
    }
};

 /*
    93. Restore IP Addresses
    https://leetcode.com/problems/restore-ip-addresses/
    A valid IP address consists of exactly four integers separated by single dots.
    Each integer is between 0 and 255 (inclusive) and cannot have leading zeros.

    For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses, but "0.011.255.245", 
    "192.168.1.312" and "192.168@1.1" are invalid IP addresses.
    Given a string s containing only digits, return all possible valid IP addresses that 
    can be formed by inserting dots into s. You are not allowed to reorder or remove any 
    digits in s. You may return the valid IP addresses in any order.

    

    Example 1:
    Input: s = "25525511135"
    Output: ["255.255.11.135","255.255.111.35"]

    Example 2:
    Input: s = "0000"
    Output: ["0.0.0.0"]

    Example 3:
    Input: s = "101023"
    Output: ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
    

    Constraints:
    1 <= s.length <= 20
    s consists of digits only.
 */
class Solution {
private:
    bool isValid(string& s) {
        if (s.empty()) return false;

        if (s[0] == '0' && s.size() > 1) return false;

        if (stoi(s) > 255) return false;
        
        return true;
    }
    void backtracking(string& s, vector<string>& res, string& valid, int next, int count) {
        if (next > s.size() || count > 4) return;

        if (next == s.size() && count == 4) {
            if (!valid.empty()) valid.pop_back();
            res.push_back(valid);
            return;
        }

        string temp = "";
        string tempValid = valid;
        for (int i = next; i < s.size(); ++i) {
            temp.push_back(s[i]);
            if (isValid(temp)) {
                valid+= temp;
                valid.push_back('.');
                backtracking(s, res, valid, i + 1, count + 1);
                valid = tempValid;
            } else {
                valid = tempValid;
                break;
            }
        }
    }
public:
    vector<string> restoreIpAddresses(string s) {
        vector<string> res;
        string valid = "";
        backtracking(s, res, valid, 0, 0);
        return res;
    }
};


 /*
    78. Subsets
    https://leetcode.com/problems/subsets/
    Given an integer array nums of unique elements, return all possible 
    subsets (the power set).

    The solution set must not contain duplicate subsets. Return the solution in any order.

    

    Example 1:
    Input: nums = [1,2,3]
    Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

    Example 2:
    Input: nums = [0]
    Output: [[],[0]]
    

    Constraints:
    1 <= nums.length <= 10
    -10 <= nums[i] <= 10
    All the numbers of nums are unique.
 */
class Solution {
private:
    void backtracking(vector<int>& nums, vector<vector<int>>& res, vector<int>& set, int next) {
        res.push_back(set);
        if (next >= nums.size()) return;

        for(int i = next; i < nums.size(); ++i) {
            set.push_back(nums[i]);
            backtracking(nums, res, set, i + 1);
            set.pop_back();
        }

    }
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> set;
        backtracking(nums, res, set, 0);
        return res;
    }
};

/*
    We mimic the process of how to build a subsets. 
*/
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res(1, vector<int>());
        int len = nums.size();
        for(int i = 0; i < len; i++){
            int n = res.size();
            for(int j = 0; j < n; j++){
                //We make a copy of the last element, and then insert nums[i] to the copy
                //This is due to the nature of subsets.
                res.push_back(res[j]);
                res.back().push_back(nums[i]);
            }
        }
        return res;
    }
};


 /*
    90. Subsets II
    https://leetcode.com/problems/subsets-ii/
    Given an integer array nums that may contain duplicates, return all possible 
    subsets (the power set).

    The solution set must not contain duplicate subsets. Return the solution in any order.


    Example 1:
    Input: nums = [1,2,2]
    Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

    Example 2:
    Input: nums = [0]
    Output: [[],[0]]
    

    Constraints:
    1 <= nums.length <= 10
    -10 <= nums[i] <= 10
 */
class Solution {
private:
    void backtracking(vector<int>& nums, vector<vector<int>>& res, vector<int>& set, int next) {
        res.push_back(set);
        if (next >= nums.size()) return;

        for (int i = next; i < nums.size(); ++i) {
            if (i > next && nums[i] == nums[i-1]) continue;
            set.push_back(nums[i]);
            backtracking(nums, res, set, i + 1);
            set.pop_back();
        }
    }
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        vector<int> set;
        backtracking(nums, res, set, 0);
        return res;
    }
};

// Simulation
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> res(1, vector<int>());
        int len = nums.size();
        int n = 0;
        sort(nums.begin(), nums.end());
        for(int i = 0; i < len; i++){
            int start = 0;
            //If we need to get rid of the duplicates, we only need to insert the duplicated elements to the last inserted elements. Instead of the whole elements in res.
            //Still a little bite tricky...
            if(i >= 1 && nums[i] == nums[i-1]) start = n;
            else start = 0;
            //Since we only care about the elements inserted in last round, we need to keep track of starting index of last round.
            n = res.size();
            for(int j = start; j < n; j++){
                res.push_back(res[j]);
                res.back().push_back(nums[i]);
            }
        }
        return res;
    }
};

 /*
    491. Non-decreasing Subsequences
    https://leetcode.com/problems/non-decreasing-subsequences/
    Given an integer array nums, return all the different possible non-decreasing subsequences 
    of the given array with at least two elements. You may return the answer in any order.


    Example 1:
    Input: nums = [4,6,7,7]
    Output: [[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]

    Example 2:
    Input: nums = [4,4,3,2,1]
    Output: [[4,4]]
    

    Constraints:
    1 <= nums.length <= 15
    -100 <= nums[i] <= 100
 */
class Solution {
private:
    void backtracking(const vector<int>& nums, vector<vector<int>>& res, vector<int>& set, int next) {
        if (set.size() >= 2) {
            res.push_back(set);
        }
        if (next >= nums.size()) return;

        // Note for this problem, we cannot sort the original nums. Thus we need an extra
        // array / set to record whether we have used this element in the traversal tree level,
        // and ignore the duplicates in this way.
        int used[201] = {0};
        for (int i = next; i < nums.size(); ++i) {
            if ((!set.empty() && nums[i] < set.back()) || used[nums[i] + 100] == 1) continue;
            used[nums[i] + 100] = 1;
            set.push_back(nums[i]);
            backtracking(nums, res, set, i + 1);
            set.pop_back();

        }
    }
public:
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> set;
        backtracking(nums, res, set, 0);
        return res;
    }
};

 /*
    46. Permutations
    https://leetcode.com/problems/permutations/
    Given an array nums of distinct integers, return all the possible permutations.
    You can return the answer in any order.

    

    Example 1:
    Input: nums = [1,2,3]
    Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

    Example 2:
    Input: nums = [0,1]
    Output: [[0,1],[1,0]]

    Example 3:
    Input: nums = [1]
    Output: [[1]]
    

    Constraints:
    1 <= nums.length <= 6
    -10 <= nums[i] <= 10
    All the integers of nums are unique.
 */
class Solution {
private:
    void backtracking(vector<int>& nums, vector<vector<int>>& res, vector<int>& perm, vector<int>& used) {
        if (perm.size() == nums.size()) {
            res.push_back(perm);
            return;
        }
        
        for(int i = 0; i < nums.size(); ++i) {
            if (used[nums[i] + 10] == 1) continue;
            used[nums[i] + 10] = 1;
            perm.push_back(nums[i]);
            backtracking(nums, res, perm, used);
            perm.pop_back();
            used[nums[i] + 10] = 0;
        }
    }
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> perm;
        vector<int> used(21, 0);
        backtracking(nums, res, perm, used);
        return res;
    }
};

 /*
    47. Permutations II
    https://leetcode.com/problems/permutations-ii/
    Given a collection of numbers, nums, that might contain duplicates, return all
    possible unique permutations in any order.


    Example 1:
    Input: nums = [1,1,2]
    Output:
    [[1,1,2],
    [1,2,1],
    [2,1,1]]

    Example 2:
    Input: nums = [1,2,3]
    Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    

    Constraints:
    1 <= nums.length <= 8
    -10 <= nums[i] <= 10
 */
class Solution {
private:
    void backtracking(vector<int>& nums, vector<vector<int>>& res, vector<int>& perm, vector<int>& used) {
        if (perm.size() == nums.size()) {
            res.push_back(perm);
            return;
        }
        
        for(int i = 0; i < nums.size(); ++i) {
            // This is the tricky part!
            // used[i-1] == true, which means the previous element is set in the traversal tree branch;
            // used[i-1] == false, which means the previous element is set in the traversal tree level;
            // We need to remove duplicates in the tree level, not the branch.
            // note the i-1 here, not i.
            if (i > 0 && nums[i] == nums[i-1] && used[i-1] == 0) continue;
            
            // Only updates the logic if the current value is not set!
            if (used[i] == 0) {
                used[i] = 1;
                perm.push_back(nums[i]);
                backtracking(nums, res, perm, used);
                perm.pop_back();
                used[i] = 0;
            }
        }
    }
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> perm;
        // Unlike problem 46, here we need to allocate the used array to store the state for each
        // element in the nums.
        vector<int> used(nums.size(), 0);
        sort(nums.begin(), nums.end());
        backtracking(nums, res, perm, used);
        return res;
    }
};

 /*
    332. Reconstruct Itinerary
    https://leetcode.com/problems/reconstruct-itinerary/
    You are given a list of airline tickets where tickets[i] = [fromi, toi] represent the
     departure and the arrival airports of one flight. Reconstruct the itinerary in order 
     and return it.

    All of the tickets belong to a man who departs from "JFK", thus, the itinerary must begin 
    with "JFK". If there are multiple valid itineraries, you should return the itinerary that 
    has the smallest lexical order when read as a single string.

    For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
    You may assume all tickets form at least one valid itinerary. You must use all the tickets 
    once and only once.


    Example 1:
    Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
    Output: ["JFK","MUC","LHR","SFO","SJC"]

    Example 2:
    Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
    Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
    Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"] 
    but it is larger in lexical order.
    

    Constraints:
    1 <= tickets.length <= 300
    tickets[i].length == 2
    fromi.length == 3
    toi.length == 3
    fromi and toi consist of uppercase English letters.
    fromi != toi
 */
class Solution {
private:
    // Using an ordered map to ensure the lexical order.
    unordered_map<string, map<string, int>> uGraph; 
    bool backtracking(vector<string>& res, int numOfTikets) {
        // The itinerary is equal to the tickets num + 1 
        if (res.size() == numOfTikets + 1) {
            return true;
        }

        // We search for the previous stop
        for (pair<const string, int>& p : uGraph[res.back()]) {
            // We can still fly from this airport
            if (p.second > 0) {
                p.second --;
                res.push_back(p.first);
                if (backtracking(res, numOfTikets)) return true;
                p.second ++;
                res.pop_back();
            }
        }
        return false;
    }

public:
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        vector<string> res;
        res.push_back("JFK");
        // Build graph
        for (auto& e : tickets) {
            uGraph[e[0]][e[1]] ++;
        }
        backtracking(res, tickets.size());
        return res;
    }
};

 /*
    51. N-Queens
    https://leetcode.com/problems/n-queens/
    The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that 
    no two queens attack each other.

    Given an integer n, return all distinct solutions to the n-queens puzzle. You may return 
    the answer in any order.

    Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' 
    and '.' both indicate a queen and an empty space, respectively.

    
    Example 1:
    Input: n = 4
    Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
    Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above

    Example 2:
    Input: n = 1
    Output: [["Q"]]
    

    Constraints:
    1 <= n <= 9
 */
class Solution {
private:
    bool isValid(int row, int col, vector<string>& chessboard) {
        // Optimization.
        // we only need to check row, given we evaluate col by col. At this point, there should be no
        // 'Q' from the previous col. See line 32.
        for (int i = 0; i < row; i++) { 
            if (chessboard[i][col] == 'Q') {
                return false;
            }
        }
        // We only need to check the 45 degrees (above)
        for (int i = row - 1, j = col - 1; i >=0 && j >= 0; i--, j--) {
            if (chessboard[i][j] == 'Q') {
                return false;
            }
        }
        // We only need to check the 135 degrees (above)
        for(int i = row - 1, j = col + 1; i >= 0 && j < chessboard.size(); i--, j++) {
            if (chessboard[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }
    void backtracking(vector<vector<string>>& res, vector<string>& layout, int row, int n) {
        if (row >= n) {
            res.push_back(layout);
            return;
        }

        for (int i = 0; i < layout[row].size(); ++i) {
            layout[row][i] = 'Q';
            if (isValid(row, i, layout)) backtracking(res, layout, row + 1, n);
            layout[row][i] = '.';
        }
    }
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        vector<string> layout(n, string(n, '.'));
        backtracking(res, layout, 0, n);
        return res;
    }
};
