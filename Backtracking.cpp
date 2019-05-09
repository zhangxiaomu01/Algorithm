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


