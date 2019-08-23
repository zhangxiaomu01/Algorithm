//8. String to Integer (atoi)
//https://leetcode.com/problems/string-to-integer-atoi/
/* Not that hard, be careful about the test case */
//Note how we handle the overflow
class Solution {
public:
    int myAtoi(string str) {
        int i = 0,  sign = 1;
        unsigned int ans = 0;
        int len = str.size();
        while(i < len && str[i] == ' ')
            i++;
        
        if(i < len && str[i] == '-'){
            sign = -1;
            i++;
        }else if (i < len && str[i] == '+')
            i++;
        
        while(i < len && (str[i] >= '0' && str[i] <= '9')){
            int backA = ans;
            ans = ans * 10 + (str[i] - '0');
            if(ans < 0 || backA != (ans - (str[i] - '0'))/10 || ans > INT_MAX )
                return sign == 1 ? INT_MAX : INT_MIN;
            i++;
        }
        return ans * sign;
        
    }
};


//3. Longest Substring Without Repeating Characters
//https://leetcode.com/problems/longest-substring-without-repeating-characters/
//The general idea is sliding window. We use hash map to store
//the visited element. Each time when we encounter the duplicate 
//element, we slide the left pointer and until we reach the new 
//duplicate element. Each loop we update the max length and save the 
//visited element to the map
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int len = s.size();
        unordered_map<char, int> map;
        int maxLen = 0;
        int start = 0;
        for(int i = 0; i < len; ++i){
            if(map.find(s[i])!= map.end()){
                //Slide the lowest boundry
                while(map.count(s[i]) > 0) map.erase(s[start++]);
            }
            //Note we shift the start 1 character right
            maxLen = max(maxLen, i - start + 1);
            map[s[i]] = i;
        }
        return maxLen;
    }
};

//The same idea. However, we use int[256] to map the all possible
//characters to potential entry, and can quiaky retrieve the starting
// index
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        vector<int> dict(256, -1);
        int len = s.size();
        int start = -1, maxLen = 0;
        for(int i = 0; i < len; ++i){
            //We find a duplicate element which has already
            //saved in the dict
            if(dict[s[i]] > start){
                start = dict[s[i]];
            }
            maxLen = max(maxLen, i - start);
            dict[s[i]] = i;
        }
        return maxLen;
    }
};


//387. First Unique Character in a String
//https://leetcode.com/problems/first-unique-character-in-a-string/
/* Test it case by case, nothing special */
class Solution {
public:
    int firstUniqChar(string s) {
        int dict[256];
        fill_n(dict, 256, 0);
        int len = s.size();
        int j = 0;
        for(int i = 0; i < len; ++i){
            int id = s[i];
            dict[id] += 1;
            
            if(i!= j && s[i] == s[j]){
                j++;
                while(j < len && dict[s[j]] > 1)
                    j++;
            }
        }
        
        return j == len ? -1 : j;
    }
};


//392. Is Subsequence
//https://leetcode.com/problems/is-subsequence/
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int len = t.size();
        int lenS = s.size();
        int j = 0;
        for(int i = 0; i < len && j < lenS; ++i){
            j += static_cast<int>(s[j] == t[i]);
            if(j == lenS) return true;
        }
        //case: s == ""
        return j==lenS;
    }
};

/* Binary search */
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int lenT = t.size();
        //Define a dictionary to save all the corresponding
        //characters in t
        vector<int>* dict[256];
        fill_n(dict, 256, nullptr);
        
        for(int i = 0; i < lenT; ++i){
            if(dict[t[i]] == nullptr)
                dict[t[i]] = new vector<int>();
            dict[t[i]]->push_back(i);
        }
        int lenS = s.size();
        //Record the last index we have been searched
        int preIndex = 0;
        for(int i = 0; i < lenS; ++i){
            if(dict[s[i]] == nullptr) return false;
            auto it = lower_bound(dict[s[i]]->begin(), dict[s[i]]->end(), preIndex);
            if(it == dict[s[i]]->end() || *it < preIndex) return false;
            //Note we need to add the index by 1 in order to prevent the repetitive
            //search for the same element (handle duplicate characters in the string)
            preIndex = *it + 1;
        }
        
        for(auto& i: dict){
            delete i;
        }
        return true;
        
    }
};


//540. Single Element in a Sorted Array
//https://leetcode.com/problems/single-element-in-a-sorted-array/
/* Bit manipulation */
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int res = 0;
        int len = nums.size();
        for(int n : nums){
            res ^= n;
        }
        return res;
    }
};

/* Binary search */
/* The binary search solution hard to get right. Note we will
have two pairs of conditions, the first one is the mid pointer
(even / odd), the second one is whether nums[mid] == nums[mid-1]
or nums[mid] == nums[mid+1]
The problem is when we need to use nums[mid] == nums[mid-1] or
nums[mid] == nums[mid+1] given that we already know that mid pointer
is even or odd.
Note by further observation, that if mid is located in an even index 
entry, which means the lower half has odd number of elements. We need
to compare with the nums[mid+1], we know if nums[mid] == nums[mid+1],
the unique number must be in the left since the left half contains 
odd number of elements. Else, we know unique number must be in the 
right half. The same reason for the odd mid pointer.*/
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int l = 0, r = nums.size()-1;
        int len = nums.size();
        if(len == 1) return nums[0];
        while(l < r){
            int mid = l + (r - l)/2;
            if(mid % 2 == 0){
                if(nums[mid] == nums[mid+1])
                    l = mid+1;
                else
                    r = mid;
            }else{
                if(nums[mid] == nums[mid-1])
                    l = mid+1;
                else
                    r = mid;
            }
        }
        return nums[r];
    }
};


//383. Ransom Note
//https://leetcode.com/problems/ransom-note/
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        vector<int> dict(256, 0);
        for(char c: magazine){
            dict[c]++;
        }
        for(char c : ransomNote){
            dict[c]--;
            if(dict[c] < 0) return false;
        }
        return true;
    }
};


//344. Reverse String
//https://leetcode.com/problems/reverse-string/
class Solution {
public:
    void reverseString(vector<char>& s) {
        int len = s.size();
        int i = 0, j = len - 1;
        while(i < j){
            swap(s[i++], s[j--]);
        }
    }
};



//345. Reverse Vowels of a String
//https://leetcode.com/problems/reverse-vowels-of-a-string/
class Solution {
private:
    bool isVowel(char c){
        return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'
               || c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
    }
public:
    string reverseVowels(string s) {
        int len = s.size();
        int i = 0, j = len-1;
        while(i < j){
            if(isVowel(s[i]) && isVowel(s[j])){
                swap(s[i], s[j]);
                i++;
                j--;
            }
            
            if(!isVowel(s[i])) i++;
            if(!isVowel(s[j])) j--;
        }
        return s;
    }
};



//290. Word Pattern
//https://leetcode.com/problems/word-pattern/
/* Naive check solution */
class Solution {
public:
    bool wordPattern(string pattern, string str) {
        stringstream ss(str);
        unordered_map<char, string> dictC;
        unordered_map<string, char> dictS;
        int i = 0;
        while(ss){
            string t;
            ss >> t;
            if((i == pattern.size() && t != "") || (i < pattern.size() && t == "")) return false;
            //reach the end of the str
            if(t == "")
                break;
            if(dictC.count(pattern[i]) == 0 && dictS.count(t) == 0){
                dictC[pattern[i]] = t;
                dictS[t] = pattern[i];
            }else if(dictC.count(pattern[i]) > 0 && dictS.count(t) > 0){
                if(dictC[pattern[i]] != t || dictS[t] != pattern[i])
                    return false;
            }else return false;
            
            i++;
        }
        return true;
    }
};


//20. Valid Parentheses
//https://leetcode.com/problems/valid-parentheses/
class Solution {
public:
    bool isValid(string s) {
        unordered_map<char,char> hash = {{')','('},{'}','{'},{']','['}};
        vector<char> pStack;
        int len = s.size();
        for(int i = 0; i< len; i++)
        {
            if(s[i] == '(' || s[i] == '{'||s[i] == '[')
                pStack.push_back(s[i]);
            if(s[i] == ')' || s[i] == '}'||s[i] == ']'){
                if(pStack.empty() || pStack.back() != hash[s[i]])
                    return false;
                else
                    pStack.pop_back();
            }                     
        }
        return pStack.empty();
    }
};


//22. Generate Parentheses
//https://leetcode.com/problems/generate-parentheses/
/* Recursion, the key insight is that when we want to insert r parenthesis,
we need to make sure that num of l < num of r */
class Solution {
private:
    void helper(int l, int r, int n, string s, vector<string>& res){
        if(l > n || r > n) return;
        if(l == n && r == n) res.push_back(s);
        
        if(l == 0 || l < n)
            helper(l+1, r, n, s + '(', res);
        
        if(l > r && r < n)
            helper(l, r+1, n, s + ')', res);
    }
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        helper(0, 0, n, "", res);
        return res;
    }
};

//Interesting implementation! The tricky part is how to define the string
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        if(n == 0)
            res.push_back("");
        else{
            for(int left = 0; left < n; left++){
                vector<string> tempL = generateParenthesis(left);
                for(int i = 0; i< tempL.size(); i++){
                     string l = tempL[i];
                    vector<string> tempR = generateParenthesis(n - left -1);
                    for(int j = 0; j < tempR.size(); j++){
                        string r = tempR[j];
                        //Here is how we formulate the final string
                        res.push_back("(" + l + ")" + r);
                    }   
                }
            }
        }
        return res;
    }
};



//241. Different Ways to Add Parentheses
//https://leetcode.com/problems/different-ways-to-add-parentheses/
/* Recursive Solution: 
The general idea for this approach is to recursively calculate each 
possible calculation result from [0, i] and [i+1, n]. Then we do the
final operation based on current operator. 
Not easy to form the intuition.
*/
class Solution {
public:
    vector<int> diffWaysToCompute(string input) {
        //compute the final result
        vector<int> res;
        int len = input.size();
        for(int i = 0; i < len; ++i){
            char c = input[i];
            //if not 0, 1, ..., 9
            if(!isdigit(c)){
                vector<int> resL, resR;
                resL = diffWaysToCompute(input.substr(0, i));
                resR = diffWaysToCompute(input.substr(i+1));
                
                for(int n1 : resL){
                    for(int n2 : resR){
                        switch(c){
                            case '+':
                                res.push_back(n1 + n2);
                                break;
                            case '-':
                                res.push_back(n1 - n2);
                                break;
                            case '*':
                                res.push_back(n1 * n2);
                                break;
                        }
                    }
                }
            }
        }
        //We do not push any value in res, which means res
        //contains only digits. We push back the pure integer
        //to res
        if(res.empty()){
            res.push_back(stoi(input));
        }
        return res;
    }
};

/* Optimized DP solution */
/* Recursive solution + memorization : Actually, become even slower */
class Solution {
    vector<int> helper(string& input, unordered_map<string, vector<int>>& memo){
        int len = input.size();
        vector<int> res;
        for(int i = 1; i < len; ++i){
            char c = input[i];
            if(!isdigit(c)){
                vector<int> resL, resR;
                string sLeft = input.substr(0, i);
                //cout << sLeft << endl;
                if(memo.find(sLeft)!= memo.end()){
                    resL = memo[sLeft];
                }else
                    resL = move(helper(sLeft, memo));
                
                string sRight = input.substr(i+1);
                if(memo.find(sRight) != memo.end())
                    resR = memo[sRight];
                else
                    resR = move(helper(sRight, memo));
                
                for(int j = 0; j < resL.size(); ++j){
                    for(int k = 0; k < resR.size(); ++k){
                        if(c == '+')
                            res.push_back(resL[j] + resR[k]);
                        else if (c == '-')
                            res.push_back(resL[j] - resR[k]);
                        else if (c == '*')
                            res.push_back(resL[j] * resR[k]);
                    }
                }
            }
        }
        if(res.empty())
            res.push_back(stoi(input));
        
        memo[input] = move(res);
        return memo[input];
    }
public:
    vector<int> diffWaysToCompute(string input) {
        unordered_map<string, vector<int>> memo;
        return helper(input, memo);
    }
};

//32. Longest Valid Parentheses
//https://leetcode.com/problems/longest-valid-parentheses/
/* DP solution! */
class Solution {
public:
    int longestValidParentheses(string s) {
        int len = s.size();
        if(len == 0) return 0;
        
        int dp[len];
        //third parameter - number of bytes
        memset(dp, 0, sizeof(dp));
        int maxLen = 0;
        //count how many '(' we have
        int count = 0;
        for(int i = 0; i < s.size(); ++i){
            if(s[i] == '(')
                count++;
            if(count > 0 && s[i] == ')'){
                //if first element is ')', count
                //is equal to 0, no need to worry
                //about the boundry
                //dp[i-1] is 0 if s[i-1] is '('
                //dp[i-1] is the local max number of
                //valid parentheses if s[i-1] is ')'
                dp[i] = 2 + dp[i-1];
                if(i - dp[i] >= 0)
                    dp[i] += dp[i-dp[i]];
                //Count will never be less than 0
                //do not put it out side the else
                count --;
            }
            maxLen = max(maxLen, dp[i]);
        }
        return maxLen;
    }
};


//301. Remove Invalid Parentheses
//https://leetcode.com/problems/remove-invalid-parentheses/
/* BFS version, we need a set to do early prune, or TLE*/
class Solution {
private:
    bool isValid(string &s){
        int len = s.size();
        int countL = 0;
        for(int i = 0; i < len; ++i){
            if(s[i] == '(')
                countL++;
            else if(s[i] == ')'){
                if(countL > 0)
                    countL--;
                else return false;
            }
        }
        return !countL;
    }
public:
    vector<string> removeInvalidParentheses(string s) {
        vector<string> res;
        unordered_set<string>set;
        queue<string> Q;
        int count = 0;
        Q.push(s);
        
        while(!Q.empty()){
            int lenQ = Q.size();
            count++;
            for(int i = 0; i < lenQ; i++){
                string tempS = Q.front();
                Q.pop();
                if(set.count(tempS) != 0) continue;
                set.insert(tempS);
                if(isValid(tempS)) {
                    res.push_back(tempS);
                    continue;
                }
                if(!isValid(tempS) && !res.empty()) continue;
                
                for(int j = 0; j < tempS.size(); ++j){
                    if(tempS[j] == '(' || tempS[j] == ')'){
                        Q.push(tempS.substr(0, j) + tempS.substr(j+1));
                    }    
                }
            }
            if(!res.empty()) break;
        }
        return res;
    }
};


//28. Implement strStr()
//https://leetcode.com/problems/implement-strstr/
//The general idea is to use two pointers, one for the starting
//position, the other is for the length of the needle
class Solution {
public:
    int strStr(string haystack, string needle) {
        int len_n = needle.size();
        int len_h = haystack.size();
        if(len_n == 0)
            return 0;
        
        int j = 0;
        for(int i = 0; i< len_h; i++){
            if(haystack[i] == needle[j]){
                if(j == len_n - 1)
                    return i - j;
                j++;
            }
            else{
                i = i - j;
                j = 0;
            }
                
        }
        return -1;
    }
};


//443. String Compression
//https://leetcode.com/problems/string-compression/
/* Not optimized version */
class Solution {
public:
    int compress(vector<char>& chars) {
        vector<char> compress;
        int count = 1;
        chars.push_back('0');
        for(int i = 0; i < chars.size(); ++i){
            if(i == 0) {
                compress.push_back(chars[i]);
                continue;
            }
                
            if(chars[i] == chars[i-1]){
                count++;
            }else if(chars[i]!= chars[i-1]){
                if(count != 1){
                    stringstream ss;
                    ss << count;
                    string tS(ss.str());
                    for(char digit : tS){
                        //cout << digit << endl;
                        compress.push_back(digit);
                    }                   
                }
                count = 1;
                compress.push_back(chars[i]);
            }
        }
        compress.pop_back();
        int len = compress.size();
        swap(compress, chars);
        return len;
    }
};

//Optimized solution: two pointers
class Solution {
public:
    int compress(vector<char>& chars) {
        int cnt = 0;
        int curPtr = 0;
        int len = chars.size();
        //handles the corner case
        chars.push_back('0');
        for(int i = 0; i < len; ++i){
            cnt++;
            if(chars[i] != chars[i+1] || i == len){
                chars[curPtr++] = chars[i];
                if(cnt > 1){
                    string s = to_string(cnt);
                    for(int j = 0; j < s.size(); ++j){
                        chars[curPtr++] = s[j];
                    }
                }
                cnt = 0;
            }
        }
        chars.pop_back();
        return curPtr;
    }
};


//14. Longest Common Prefix
//https://leetcode.com/problems/longest-common-prefix/
/* Brute force solution, compare prefix char by char. Can terminate earlier! */
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int len = strs.size();
        if(len == 0) return "";
        if(len == 1) return strs[0];
        int prefixLen = 0;
        for(int i = 0; i < strs[0].size(); ++i){
            for(int j = 1; j < strs.size(); ++j){
                if(i >= strs[j].size() || strs[j][i] != strs[j-1][i]){
                    return strs[j].substr(0, prefixLen);
                }
            }
            prefixLen++;
        }
        return strs[0].substr(0, prefixLen);
    }
};


//Sorting to decrease the number of comparisons
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int len = strs.size();
        int prefixLen = 0;
        if(len == 0) return "";
        sort(strs.begin(), strs.end());
        int n = strs[0].size();
        for(int i = 0; i<n; i++)
        {
            if(strs[0][i] == strs[len-1][i]){
                prefixLen ++;
            }  
            else
                break;
        }
        return strs[0].substr(0, prefixLen);
    }
};

//58. Length of Last Word
//https://leetcode.com/problems/length-of-last-word/
/* pretty standard solution */
class Solution {
public:
    int lengthOfLastWord(string s) {
        int len_s = s.size() - 1;
        int len_w = 0;
        while(len_s >= 0 && s[len_s] == ' ') len_s--;
        while(len_s >= 0 && s[len_s] != ' '){
            len_s--;
            len_w++;
        }
        return len_w;
    }
};


//151. Reverse Words in a String
//https://leetcode.com/problems/reverse-words-in-a-string/
/* string stream solution. Not memory efficient! */
class Solution {
public:
    string reverseWords(string s) {
        istringstream ss(s);
        string word;
        string res;
        while(ss >> word){
            word.push_back(' ');
            reverse(word.begin(), word.end());
            res.append(word);
        }
        reverse(res.begin(), res.end());
        res.pop_back();
        return res;
    }
};

/* Optimized version */
//In-place solution is much trickier! Should be able to get it right.
class Solution {
public:
    string reverseWords(string s) {
        int len = s.size();
        if(len == 0) return s;
        int l = 0, i = 0, j = 0;
        int wordCount = 0;
        while(true){
            while(i < len && s[i] == ' ') i++;
            if(i == len) break;
            if(wordCount) s[j++] = ' ';
            l = j;
            while(i < len && s[i] != ' '){
                s[j] = s[i];
                i++;
                j++;
            }
            reverse(s.begin() + l, s.begin() + j);
            wordCount++;
        }
        s.resize(j);
        reverse(s.begin(), s.end());
        return s;
    }
};


//409. Longest Palindrome
//https://leetcode.com/problems/longest-palindrome/
//Utilize a dictionary, the problem is not hard, but has too many corner cases
class Solution {
public:
    int longestPalindrome(string s) {
        int dict[256];
        memset(dict, 0, sizeof(dict));
        for(char c : s){
            dict[c] += 1; 
        }
        int res = 0;
        int maxOdd = 0;
        for(int i = 0; i < 256; ++i){
            if(dict[i] != 0){
                if(dict[i] & 1 && dict[i] > maxOdd){
                    res += maxOdd ? maxOdd-1 : 0;
                    maxOdd = max(maxOdd, dict[i]);
                }   
                else if(dict[i] & 1)
                    res += dict[i] - 1;
                else
                    res += dict[i];
            }
        }
        res += maxOdd;
        return res;
    }
};

