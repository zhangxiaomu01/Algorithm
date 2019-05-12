#include<windows.h>
#include<vector>
#include<string>


//282. Expression Add Operators
//https://leetcode.com/problems/expression-add-operators/
//Hard problem. The concept is not hard, basically, we calculate each possible segementation, and then insert each operators in between and check the final results.
//We can improve the efficiency if we calculate the intermediate results on the fly.
//The implementation is not that easy.
class Solution {
private:
    void dfs(string s, const string& num, const long d, long pos, long cv, long pv, vector<string>& res, char op){
        if(pos == num.size() && cv == d)
            res.push_back(s);
        else{
            for(int j = pos+1; j <= num.size(); j++){
                string t = num.substr(pos, j - pos);
                long v = stol(t);
                
                if(to_string(v).size() != t.size()) break;
                dfs(s + '+' + t, num, d, j, cv + v, v, res, '+');
                dfs(s + '-' + t, num, d, j, cv - v, v, res, '-');
                //Note we need to pass op to dfs function below, since we already calculate the multiplication, and we only care about the sum and deduction now.
                if(op == '+')
                    dfs(s + '*' + t, num, d, j, cv - pv + pv*v, pv*v, res, op);
                else if(op == '-')
                    dfs(s + '*' + t, num, d, j, cv + pv - pv*v, pv*v, res, op);
                //We are actually handling the case that we have several * at the very beginning
                else if(op == '*')
                    dfs(s + '*' + t, num, d, j, cv * v, pv*v, res, op);
            }
        }
        
    }
public:
    vector<string> addOperators(string num, int target) {
        vector<string> res;
        int len = num.size();
        if(len == 0) return res;
        for(int i = 1; i <= len; i++){
            string s = num.substr(0, i);
            long val = stol(s);
            //We can safely break here because "05", "00" is considered as invalid number, or we have already covered in the previous iteration like "05 = 0 + 5" 
            if(to_string(val).size() != s.size()){
                break;
            }
            dfs(s, num, target, i, val, val, res, '*');
        }
        return res;
    }
};

//78. Subsets
//https://leetcode.com/problems/subsets/
/*
Classic backtracking problem: Be sure to get familiar with this technique...
The natural approach is to try all possible combination under limitations.
*/
class Solution {
private:
    void backtrack(vector<int>& n, vector<vector<int>>& res, int pos, vector<int>& temp){
        res.push_back(temp);
        //Note we will try all combinations
        for(int i = pos; i < n.size(); i++){
            temp.push_back(n[i]);
            backtrack(n, res, i+1, temp);
            temp.pop_back();
        }
    }
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        int len = nums.size();
        vector<vector<int>> res;
        vector<int> temp;
        backtrack(nums, res, 0, temp);
        return res;
    }
};

/*
This approach is more straightforward. We mimic the process of how to build a subsets. 
A very good approach. Neat.
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

//90. Subsets II
//https://leetcode.com/problems/subsets-ii/
//The same thing like subset problem. We need to handle duplicates now
class Solution {
private:
    void backtrack(vector<int>& n, vector<vector<int>>& res, int pos, vector<int>& temp){
        res.push_back(temp);
        for(int i = pos; i < n.size(); i++){
            //The only duplicate we need to handle is when we do the backtrack, and encounter the same element again. 
            //We need to get rid of the duplicate sets like this. 
            if(i != pos && n[i] == n[i-1]) continue;
            temp.push_back(n[i]);
            backtrack(n, res, i+1, temp);
            temp.pop_back();
        }
    }
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> temp;
        //Sorting is required.
        sort(nums.begin(), nums.end());
        backtrack(nums, res, 0, temp);
        return res;
    }
};
//A little bit tricky....
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

//77. Combinations
//https://leetcode.com/problems/combinations/
/*
Basic backtracking problem. The structure is similar to subset problem.
*/
//DFS + Backtracking
class Solution {
private:
    void backtracking(vector<vector<int>>& res, vector<int>& temp, int start, int n, int k){
        if(temp.size() == k){
            res.push_back(temp);
            return;
        }
        for(int i = start; i < n; i++){
            temp.push_back(i+1);
            backtracking(res, temp, i+1, n, k);
            temp.pop_back();
        }
    }
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> temp;
        backtracking(res, temp, 0, n, k);
        return res;
    }
};

//This iterative version is elegant, however, hard to get the right insight during interview
class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        //Allocating an array of size k, which represents one possible combination
        vector<int> container(k, 0);
        int i = 0;
        //The whole process is similar to dfs, it might be easier if we implement using stack.
        while(i >= 0){
            container[i]++;
            //If the value exceed the maximum value, we need to increase the previous value 
            if(container[i]>n) i--;
            else if(i == k-1) res.push_back(container);
            else{
                i++;
                container[i] = container[i-1];
            }
        }
        return res;
    }
};

class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> st;
        int i = 0;
        st.push_back(i);
        //Using stack... However,it's slower than other solutions
        //Push and pop operations may contribute to resizing array.
        while(!st.empty()){
            st.back()++;
            if(st.back() > n) st.pop_back();
            else if(st.size() == k) res.push_back(st);
            else{
                int val = st.back();
                st.push_back(val);
            }
        }
        return res;
    }
};

//39. Combination Sum
//https://leetcode.com/problems/combination-sum/
/*
Backtracking problem: Note that in order to reuse elements, we need to start from the same position
*/
//Unoptimized version, actually we can terminate earlier.
class Solution {
private:
    void dfs(vector<vector<int>>& res, vector<int>& temp, int pos, int t, vector<int> nums){
        if(t < 0) return;
        if(t == 0){
            res.push_back(temp);
            return;
        }
        for(int i = pos; i < nums.size(); i++){
            temp.push_back(nums[i]);
            //Since no duplicates in candidates array, and we can reuse elements, make pos == i
            dfs(res, temp, i, t - nums[i], nums);
            temp.pop_back();
        }
    }
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> temp;
        dfs(res, temp, 0, target, candidates);
        return res;
    }
};

//Optimized version
class Solution {
private:
    void dfs(vector<vector<int>>& res, vector<int>& temp, int pos, int t, vector<int> nums){
        if(t == 0){
            res.push_back(temp);
            return;
        }
        //We add t-nums[i]>= 0 here to terminate earlier... 
        for(int i = pos; i < nums.size() && t - nums[i] >= 0; i++){
            temp.push_back(nums[i]);
            //In order to reuse the same value, pos should be the same as i, instead of i+1
            dfs(res, temp, i, t - nums[i], nums);
            temp.pop_back();
        }
    }
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> temp;
        //Sorting is required, since we add t - nums[i] >= 0, we need to always substract the smaller elements first
        sort(candidates.begin(), candidates.end());
        dfs(res, temp, 0, target, candidates);
        return res;
    }
};

//40. Combination Sum II
//https://leetcode.com/problems/combination-sum-ii/
/*
In general, the code is exactly the same as before. 
The tricky part is how we handle duplicate elements...
*/
class Solution {
private:
    void dfs(vector<vector<int>>& res, vector<int>& temp, int pos, int t, vector<int>& nums){
        if(t == 0){
            res.push_back(temp);
            return;
        }
        for(int i = pos; i < nums.size() && t - nums[i] >= 0; i++){
            /*Here we can handle duplicates... 
            The reason behind it is when the first time we encounter a duplicate element, we include it in our solution. Since we already sort the array, then we know any duplicate set has already been covered by the first time we push nums[i] to our set. The only thing we need to check if we found any duplicate elements later(not the first time), we need to get rid of these sets. Note we cannot break here, or we will miss several sets.
            */
            if(i != pos && i > 0 && nums[i] == nums[i-1]) continue;
            temp.push_back(nums[i]);
            dfs(res, temp, i+1, t-nums[i], nums);
            temp.pop_back();
        }
    }
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> temp;
        sort(candidates.begin(), candidates.end());
        dfs(res, temp, 0, target, candidates);
        return res;
    }
};

//216. Combination Sum III
//https://leetcode.com/problems/combination-sum-iii/
//Exactly the same as Combination Sum problem
class Solution {
private:
    void dfs(int k, int n, int pos, vector<vector<int>>& res, vector<int>& temp){
        if(temp.size() == k && n == 0){
            res.push_back(temp);
            return;
        } 
        for(int i = pos; i <= 9 && n - i >= 0; i++){
            temp.push_back(i);
            dfs(k, n-i, i+1, res, temp);
            temp.pop_back();
        }
        
    }
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> res;
        vector<int> temp;
        dfs(k, n, 1, res, temp);
        return res;
    }
};

//377. Combination Sum IV
//https://leetcode.com/problems/combination-sum-iv/
/*
The general idea is totally different than Combination sum I/II/III. Instead of using backtracking, we are just using 
dfs to count the maximum possible combinations. Since we only need how many possible valid combinations, we can easily
convert the solution to iterative version. (It's actually DP) 
*/
//The recursive version is more intuitive, although it's not that efficient
//We use Recuirsion + memorization to improve the efficiency
//The reason why we can use memorization is because if we already know how many combinations can lead to some target value,
//We do not need to do the same calculation again.
class Solution {
private:

    int dfs(vector<int>& nums, int target, vector<int>& memo){
        //Any invalid combination sum we return 0 to indicate it's invalid
        //Else we find a new combination sum, we return 1
        if(nums.empty() || target < 0) return 0;
        if(target == 0) return 1;
        if(memo[target]!= -1) return memo[target];
        unsigned int count = 0;
        for(int i = 0; i < nums.size(); i++){
            //We are actually checking each possible combination, first we reduce nums[0], then nums[1]... and so on
            count += dfs(nums, target - nums[i], memo);
        }
        return memo[target] = count;
    }
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<int> memo(target+1, -1);
        return dfs(nums, target, memo);
    }
};

//Iterative DP solution
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<unsigned int> dp(target+1, 0);
        dp[0] = 1;
        sort(nums.begin(), nums.end());
        for(int i = 1; i <= target; i++){
            for(int n : nums){
                //We can break here is because we sort the array
                if(i < n) break;
                //dp[i] means how many combination sets which sum to value i
                dp[i] += dp[i - n];
            }
        }
        return dp.back();
    }
};

//46. Permutation
//https://leetcode.com/problems/permutations/
/*
Similar to set problem. However, we swap elements in the original array...
*/
class Solution {
private:
    void dfs(vector<vector<int>>& res, int pos, vector<int>& nums){
        if(pos >= nums.size()){
            res.push_back(nums);
            return;
        }
        for(int i = pos; i < nums.size(); i++){
            //We swap two elements temporarily
            swap(nums[pos], nums[i]);
            //Note we need to update pos, instead of i+1;
            //Remember that permutation, we fix one position and swap elements accordinglily
            dfs(res, pos+1, nums);
            swap(nums[pos], nums[i]);
        }
        
    }
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        dfs(res, 0, nums);
        return res;
    }
};


//47. Permutations II
//https://leetcode.com/problems/permutations-ii/
/*
Permutations with duplicates...
*/
class Solution {
private:
    void dfs(vector<vector<int>>& res, vector<int>& nums, int pos){
        if(pos >= nums.size()){
            res.push_back(nums);
            return;
        }
        //Note how we handle duplicates, we can only put set here because for each round, you need to keep track of which repetitive element has already been swapped
        //For [1,1,2], when we meet second 1, we already calculate the permutation when we handle first 1
        unordered_set<int> Dset;
        for(int i = pos; i < nums.size(); i++){
            if(Dset.find(nums[i]) == Dset.end()){
                Dset.insert(nums[i]);
                swap(nums[pos], nums[i]);
                dfs(res, nums, pos+1);
                swap(nums[pos], nums[i]);
            }
        }
    }
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        dfs(res, nums, 0);
        return res;
    }
};

//31. Next Permutation
//https://leetcode.com/problems/next-permutation/
/*
The problem is easy to understand, however, the code is hard to implement.
*/
//The problem is not hard, however, the code is hard to implement
//Tried several times, still failed
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int len = nums.size();
        if(len <= 1) return;
        int i = len-2;
        //We find the first digit that is smaller than the digit following it
        while(i >=0 && nums[i] >= nums[i+1])
            i--;
        //If we find one, note it can be the first element
        if(i >= 0){
            int j = len -1;
            //Find the first element from i+1 to len-1, that is smaller than nums[i]
            //Note here we need to have <=, instead of <
            while(j >= i+1 && nums[j] <= nums[i])
                j--;
            swap(nums[i], nums[j]);
        }
        //If all the element are in descending order, than i = -1, we actually reverse the whole array
        reverse(nums.begin() + 1 + i, nums.end());
    }
};

