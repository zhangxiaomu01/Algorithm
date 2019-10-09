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

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.empty()) return 0;
        int uDict[256] = {0};
        int len = s.size();
        int i = 0;
        int maxLen = 0;
        for(int j = 0; j < len; ++j){
            uDict[s[j]] ++;
            if(uDict[s[j]] <= 1){
                maxLen = max(maxLen, j-i+1);
            }else{
                while(i < j && uDict[s[j]] > 1){
                    uDict[s[i]]--;
                    i++;
                }
            }
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

//Recursive version!
class Solution {
private:
    void reverseS(vector<char>& str, int s, int e){
        if(s >= e) return;
        swap(str[s], str[e]);
        reverseS(str, s+1, e-1);
    }
public:
    void reverseString(vector<char>& s) {
        reverseS(s, 0, s.size()-1);
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


//415. Add Strings
//https://leetcode.com/problems/add-strings/
class Solution {
public:
    string addStrings(string num1, string num2) {
        int carry = 0;
        int len1 = num1.size(), len2 = num2.size();
        if(len1 > len2) return addStrings(num2, num1);
        
        int j = len2 - 1;
        string res;
        for(int i = len1 - 1; i >= 0; --i){
            int digit1 = num1[i] - '0';
            int digit2 = num2[j] - '0';
            int sum = digit1 + digit2 + carry;
            res.push_back(sum % 10 + '0');
            carry = sum / 10;
            j--;
        }
        for(; j >= 0; --j){
            if(carry == 0){
                res.push_back(num2[j]);
            }else{
                int digit = num2[j] - '0';
                digit = digit + carry;
                
                res.push_back(digit % 10 + '0');
                carry = digit / 10;
            }
        }
        if(carry == 1) res.push_back('1');
        reverse(res.begin(), res.end());
        return res;
    }
};


//205. Isomorphic Strings
//https://leetcode.com/problems/isomorphic-strings/
//You can also use two arrays char[256] to record this
//Since we may only have 256 characters
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        int len = s.size();
        unordered_map<char, char> dict;
        unordered_map<char, char> rdict;
        for(int i = 0; i < len; i++){
            if(dict.find(s[i])!= dict.end()){
                if(dict[s[i]] != t[i])
                    return false;
            }
            if(rdict.find(t[i])!=rdict.end()){
                if(rdict[t[i]] != s[i])
                    return false;
            }
            
            dict[s[i]] = t[i];
            rdict[t[i]] = s[i];
        }
        return true;
    }
};



//49. Group Anagrams
//https://leetcode.com/problems/group-anagrams/
//Using hash map to map the key!
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> hashMap;
        vector<string> temp = strs;
        for(int i = 0; i < strs.size(); ++i){
            sort(temp[i].begin(), temp[i].end());
            hashMap[temp[i]].push_back(strs[i]);
        }
        int len = hashMap.size();
        vector<vector<string>> res(len, vector<string>());
        int j = 0;
        for(auto it = hashMap.begin(); it != hashMap.end(); ++it){
            swap((*it).second, res[j++]);
        }
        return res;
    }
};


//395. Longest Substring with At Least K Repeating Characters
//https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/
//Brute Force Solution. Even this solution is not easy to get!
class Solution {
public:
    int longestSubstring(string s, int k) {
        int len = s.size();
        //When k is less than or equal to 1, we can safely return
        if(k <= 1) return len;
        int maxLen = 0;
        for(int i = 0; i < len; ++i){
            //No need to continue search!
            if(maxLen > len - i) return maxLen;
            int cMap[26];
            memset(cMap, 0, sizeof(cMap));
            int check = 0, count = 0;
            int end = i;
            for(int j = i; j < len; ++j){
                if(cMap[s[j] - 'a'] == 0) 
                    count++;
                if(cMap[s[j] - 'a'] < k)
                    check++;
                cMap[s[j] - 'a'] ++;
                //This is the most tricky part, we record the minimum
                //characters satisfying the condition at least k
                //and move the second pointer to it
                if(check >= count * k){
                    end = j;
                }
            }
            int newLen = (end == i ? 0 : end - i + 1);
            maxLen = max(maxLen, newLen);
        }
        return maxLen;
    }
};

//OPtimized recursive solution!!
//Optimized recursive solution! 
class Solution {
private:
    int helper(string& s, int k, int l, int r){
        if(r - l < k) return 0;
        int dict[26];
        memset(dict, 0, sizeof(dict));
        int len = s.size();
        for(int i = l; i < r; ++i){
            dict[s[i] - 'a']++;
        }
        int maxLen = 0;
        for(int j = l; j < r;){
            while(j < r && dict[s[j] - 'a'] < k) j++;
            if(j == r) break;
            int end = j;
            while(end < r && dict[s[end] - 'a'] >= k) end++;
            //The whole substring meets our need, no need to do further
            //calculation!
            if(j == l && end == r) return r - l;
            maxLen = max(maxLen, helper(s, k, j, end));
            //start with the next element!
            j = end;
        }
        
        return maxLen;
    }
public:
    int longestSubstring(string s, int k) {
        int len = s.size();
        if(k <= 1) return len;
        return helper(s, k, 0, len);
    }
};


//402. Remove K Digits
//https://leetcode.com/problems/remove-k-digits/
class Solution {
public:
    string removeKdigits(string num, int k) {
        string res = "";
        for(char c : num){
            while(!res.empty() && k > 0 && res.back() > c){
                res.pop_back();
                k--;
            }
            //get rid of leading 0, if res is empty, we need to have 
            //c != '0'
            if(!res.empty() || c != '0') res.push_back(c);
        }
        //If we still need to pop more numbers
        while(!res.empty() && k > 0){
            res.pop_back();
            k--;
        }
        return res.empty() ? "0" : res;
    }
};


//929. Unique Email Addresses
//https://leetcode.com/problems/unique-email-addresses/
//The idea how to solve the problem is not hard!
//However, how to utilize the standard template library is key to 
//success!
class Solution {
public:
    int numUniqueEmails(vector<string>& emails) {
        unordered_set<string> uSet;
        string name, domain;
        unsigned int len = emails.size();
        for(int i = 0; i < len; ++i){
            //return size_t
            //The position of the first character that matches.
            auto pivot = emails[i].find_first_of('@');
            name = emails[i].substr(0, pivot);
            domain = emails[i].substr(pivot);
            //https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
            name.erase(remove(name.begin(), name.end(), '.'), name.end());
            auto pos = name.find_first_of('+');
            pos == string::npos ? uSet.insert(name + domain) : uSet.insert(name.erase(pos) + domain);
        }
        return static_cast<int>(uSet.size());
    }
};

//482. License Key Formatting
//https://leetcode.com/problems/license-key-formatting/
//Not hard!
class Solution {
public:
    string licenseKeyFormatting(string S, int K) {
        string res;
        int lenS = S.size();
        int count = 0;
        for(int i = lenS - 1; i >= 0; --i){
            if(S[i] != '-'){
                res.push_back(isalpha(S[i]) ? toupper(S[i]) : S[i]);
                count++;
                if(count == K){
                    res.push_back('-');
                    count = 0;
                }  
            }  
        }
        if(res.back() == '-') res.pop_back();
        reverse(res.begin(), res.end());
        return res;
    }
};


//87. Scramble String
//https://leetcode.com/problems/scramble-string/
//A hard problem, almost impossible to get the solution during the 
//interview. In general, when we come to the recursive solution,
//we always need to come with the base case, then validation, then
//recursive formula.
class Solution {
public:
    bool isScramble(string s1, string s2) {
        //base case
        if(s1 == s2) return true;
        if(s1.size() != s2.size()) return false;
        int dict[26] = {0};
        
        //check validation
        for(int i = 0; i < s1.size(); ++i){
            dict[s1[i] - 'a'] ++;
        }
        for(int i = 0; i < s2.size(); ++i){
            dict[s2[i] - 'a'] --;
            //s2 has different number of characters than s1
            if(dict[s2[i] - 'a'] < 0)
                return false;
        }
        
        //recursive check for substring
        for(int i = 1; i < s1.size(); ++i){
            if(isScramble(s1.substr(0, i), s2.substr(0, i)) && 
              isScramble(s1.substr(i), s2.substr(i)))
                return true;
            if(isScramble(s1.substr(0, i), s2.substr(s2.size() - i)) && 
              isScramble(s1.substr(i), s2.substr(0, s2.size() - i)))
                return true;
        }
        return false;
    }
};


//Actually, I do not think this uDict can prevent too many repetitive 
//calculation! It's slower
class Solution {
private:
    bool checkScramble(string s1, string s2, unordered_map<string, bool>& uDict){
        if(s1 == s2) return true;
        if(s1.size() != s2.size()) return false;
        string tempS = s1 + s2;
        if(uDict.count(tempS)) return uDict[tempS];
        int cDict[26] = {0};
        
        for(int i = 0; i < s1.size(); ++i){
            cDict[s1[i] - 'a']++;
        }
        for(int i = 0; i < s2.size(); ++i){
            cDict[s2[i] - 'a'] --;
            if(cDict[s2[i] - 'a'] < 0)
                return uDict[tempS] = false;
        }
        
        for(int i = 1; i < s1.size(); ++i){
            if(checkScramble(s1.substr(0, i), s2.substr(0, i), uDict) && 
               checkScramble(s1.substr(i), s2.substr(i), uDict))
                return uDict[tempS] = true;
            if(checkScramble(s1.substr(0, i), s2.substr(s2.size() - i), uDict) 
               && checkScramble(s1.substr(i), s2.substr(0, s2.size() - i), uDict))
                return uDict[tempS] = true;
        }
        return uDict[tempS] = false;
    }
public:
    bool isScramble(string s1, string s2) {
        unordered_map<string, bool> uDict;
        return checkScramble(s1, s2, uDict);
    }
};


//844. Backspace String Compare
//https://leetcode.com/problems/backspace-string-compare/
//Space: O(n)
class Solution {
public:
    bool backspaceCompare(string S, string T) {
        stack<char> sSt, tSt;
        int lenS = S.size();
        int lenT = T.size();
        int i = 0, j = 0;
        while(i < lenS){
            if(S[i] != '#')
                sSt.push(S[i]);
            else if(!sSt.empty()){
                sSt.pop();
            }
            i++;
        }
        while(j < lenT){
            if(T[j] != '#')
                tSt.push(T[j]);
            else if(!tSt.empty())
                tSt.pop();
            j++;
        }
        //std::cout << sSt.size() << std::endl;
        if(sSt.size() != tSt.size()) return false;
        while(!sSt.empty()){
            if(sSt.top() != tSt.top()) return false;
            sSt.pop();
            tSt.pop();
        }
        return true;
    }
};

//The idea is not hard, however get it well structured and figure out all the 
//corner cases is not easy!
class Solution {
public:
    bool backspaceCompare(string S, string T) {
        int lenS = S.size(), lenT = T.size();
        int i = lenS-1, j = lenT-1;
       
        while(i >= 0 || j >= 0){
            int count = 0;
             
            while(i >= 0 && (count || S[i] == '#')){
                count += (S[i] == '#' ? 1 : -1);
                i--;
            }
            count = 0;
            while(j >= 0 && (count || T[j] == '#')){
                count += (T[j] == '#' ? 1 : -1);
                j--;
            }
            //reset count!
            count = 0;
            if(i >= 0 && j >= 0 && S[i] != T[j]) return false;
            //This part is hard to get, we need to return both i == -1 &&
            //j == -1
            if(i < 0 || j < 0 ){
                return (i==-1) && (j == -1);
            } 
            //Decrement i and j here. If S[i] == T[j], and both != '#'
            --i; --j;
        }
        return true;
    }
};


//777. Swap Adjacent in LR String
//https://leetcode.com/problems/swap-adjacent-in-lr-string/
//The key insight is that start[i] == 'L', then its index i should be 
//>= j, or we can never make the move. For start[i] == 'R', then i 
//should be <= j
class Solution {
public:
    bool canTransform(string start, string end) {
        if(start == end) return true;
        if(start.size() != end.size()) return false;
        string s1, s2;
        int lenS = start.size();
        for(int i = 0; i < lenS; ++i){
            if(start[i] != 'X') s1.push_back(start[i]);
            if(end[i] != 'X') s2.push_back(end[i]);
        }
        //We do not have equal number of 'L' or 'R'
        //or they have wrong sequence
        if(s1 != s2) return false;
        
        //Two pointers!
        for(int i = 0, j = 0; i < lenS && j < lenS;){
            while(i < lenS && start[i] == 'X') i++;
            while(j < lenS && end[j] == 'X') j++;
            if(start[i] != end[j]) return false;
            if((start[i] == 'L' && i < j) || start[i] == 'R' && i > j)
                return false;
            else
                i++; j++;
        }
        return true;
    }
};


//38. Count and Say
//https://leetcode.com/problems/count-and-say/
//Just run simulation!
class Solution {
public:
    string countAndSay(int n) {
        if(n == 0) return "";
        if(n == 1) return "1";
        string res = "1";
        while(n-- > 1){
            string tempRes;
            for(int i = 0; i < res.size(); ++i){
                int count = 1;
                while( i+1 < res.size() && res[i+1] == res[i]) {
                    count++;
                    i++;
                }
                tempRes.append(to_string(count));
                tempRes.push_back(res[i]);
            }
            swap(res, tempRes);
        }
        
        return res;
    }
};


//316. Remove Duplicate Letters
//https://leetcode.com/problems/remove-duplicate-letters/
//I came up with the stack and using dict to store elements, but failed to
//know whether the character is in the res
//The key insight is stack + isInRes, and update them properly!
class Solution {
public:
    string removeDuplicateLetters(string s) {
        string res;
        if(s.empty()) return res;
        //record how many characters we have now
        int dict[26] = {0};
        //Record whether we have the character in our res now
        int isInRes[26] = {0};
        for(char c : s) dict[c - 'a'] ++;
        for(int i = 0; i < s.size(); ++i){
            dict[s[i] - 'a'] --;
            if(res.empty()){
                res.push_back(s[i]);
                isInRes[s[i] - 'a'] = 1;
                continue;
            }
            //skip those who is already in the res
            if(isInRes[s[i] - 'a']) continue; 
            
            while(!res.empty() && s[i] < res.back() && dict[res.back() - 'a'] > 0){
                isInRes[res.back() - 'a'] = 0;
                res.pop_back();
            }
            res.push_back(s[i]);
            isInRes[s[i] - 'a'] = 1;
        }
        
        return res;
    }
};

//168. Excel Sheet Column Title
//https://leetcode.com/problems/excel-sheet-column-title/
//Make sure you have totally understand the process before writing code
//Mimic the recursive call. Spent approximately 30 minutes to finish!
class Solution {
public:
    string convertToTitle(int n) {
        string res;
        while(n > 0){
            int index = n % 26;
            //Remove the amount we have
            n = n - (index == 0 ? 26 : index);
            index = ((index == 0) ? 26 : index) - 1;
            //cout << index << endl;
            res.push_back('A' + index);
            n /= 26;
        }
        reverse(res.begin(), res.end());
        return res;
    }
};


//A more elegant version
// Note hard, need some intuition...
class Solution {
public:
    string convertToTitle(int n) {
        string res;
        while(n){
            n--;
            res.push_back(static_cast<char>(n%26 + 'A'));
            n = n/26;
        }
        reverse(res.begin(), res.end());
        return res;
    }
};

//171. Excel Sheet Column Number
//https://leetcode.com/problems/excel-sheet-column-number/
//Not hard, just to be careful!
class Solution {
public:
    int titleToNumber(string s) {
        int res = 0;
        if(s.empty()) return res;
        int i = s.size() - 1;
        int count = 0;
        for(; i >= 0; --i){
            int coord = s[i] - 'A' + 1;
            res = res + coord * pow(26, count);
            count++;
        }
        return res;
    }
};

//Be careful about this kind of simple problem
class Solution {
public:
    int titleToNumber(string s) {
        int len = s.size();
        if(len == 0) return 0;
        int res = 0;
        for(int i = 0; i < len; i++){
            res = res*26 + static_cast<int>(s[i] - 'A') +1;
        }
        return res;
    }
};


//13. Roman to Integer
//https://leetcode.com/problems/roman-to-integer/
//The key insight is to discover when s[i] > s[j], we need to substract
//res -= uMap[s[i]] first, then add uMap[s[j]] - uMap[s[i]]
class Solution {  
public:
    int romanToInt(string s) {
        unordered_map<char, int> uMap({{'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000}});
        int len = s.size();
        int res = 0;
        stack<char> tempSt;
        for(int i = 0; i < len; ++i){
            if(i >= 1 && uMap[s[i]] > uMap[s[i-1]]){
                int val = uMap[s[i-1]];
                res -= val;
                res += uMap[s[i]] - val;
            }else {
                res += uMap[s[i]];
            }
        }
        return res;
    }
};

//The same idea, different implementation! find(string) is not efficient!
class Solution {
public:
    int romanToInt(string s) {
        int finalSum = 0;        
        if(s.find("IV")!=-1) finalSum -= 2;
        if(s.find("IX")!=-1) finalSum -= 2;
        if(s.find("XL")!=-1) finalSum -= 20;
        if(s.find("XC")!=-1) finalSum -= 20;
        if(s.find("CD")!=-1) finalSum -= 200;
        if(s.find("CM")!=-1) finalSum -= 200;
        
        for(int i = 0; i< s.size(); i++)
        {
            if(s[i] == 'M') finalSum += 1000;
            if(s[i] == 'D') finalSum += 500;
            if(s[i] == 'C') finalSum += 100;
            if(s[i] == 'L') finalSum += 50;
            if(s[i] == 'X') finalSum += 10;
            if(s[i] == 'V') finalSum += 5;
            if(s[i] == 'I') finalSum += 1;
        }
        
        return finalSum;
    }
};


//12. Integer to Roman
//https://leetcode.com/problems/integer-to-roman/
//Very clever approach, hard to get without practice!
class Solution {
public:
    string intToRoman(int num) {
        //Add one empty here in order to append null string, then make
        //code simpler!
        string digits[10] = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        string tens[10] = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        string hundrends[10] = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        string thousands[4] = {"", "M", "MM", "MMM"};
        
        string res;
        res.append(thousands[num/1000]);
        res.append(hundrends[(num%1000)/100]);
        res.append(tens[(num%100)/10]);
        res.append(digits[(num%10)]);
        
        return res;
    }
};


//Another interesting idea. Hard to get!
//Not as efficient as the solution 1
class Solution {
public:
    string intToRoman(int num) {
        int numbers[13] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        string comb[13] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        string res;
        for(int i = 0; i < 13; ++i){
            while(num >= numbers[i]){
                res.append(comb[i]);
                num = num - numbers[i];
            }
        }
        return res;
    }
};


//273. Integer to English Words
//https://leetcode.com/problems/integer-to-english-words/
//Be careful, you spent too much time on this! Not hard, just a little bit annoying
class Solution {
private:
    int range[4] = {1000000000, 1000000, 1000, 100};
    string dict[4] = {"Billion", "Million", "Thousand", "Hundred"};
    string digits[20] = {"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    string tens[9] = {"Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    
    //Covert integer less than 1000. Since thousand - million or million - 
    //billion, we cannot exceed 1000
    string convertItoS(int x){
        string res;
        if(x >= 100){
            int i = x/100;
            res.append(digits[i]);
            res.push_back(' ');
            res.append("Hundred ");
        }
        x = x % 100;
        if(x >= 20){
            int i = x/10;
            res.append(tens[i-1]);
            res.push_back(' ');
        }
        if(x >= 20)
            x = x % 10;
        if(x != 0){
            res.append(digits[x]);
            res.push_back(' ');
        }
        return res;
    }
public:
    string numberToWords(int num) {
        string res;
        if(num == 0) return "Zero";
        
        if(num <= 999){
            res = convertItoS(num);
            res.pop_back();
            return res;
        }
        
        for(int i = 0; i < 4; ++i){
            if(num <= 999){
                res.append(convertItoS(num));
                res.pop_back();
                return res;
            }
            int count = num / range[i];
            if(count == 0) continue;
            else{
                string temp = convertItoS(count);
                res.append(temp);
                res.append(dict[i]);
                res.push_back(' ');
            }
            num = num % range[i];
        }
        return res;
    }
};


//65. Valid Number
//https://leetcode.com/problems/valid-number/
//A hard problem! Not an easy task to get all the potential situations!
//'+/-': only valid when in first pos or immediately after 'e'
//'e': only valid when follows a digit and there cannot be another 'e'
//'.': only valid if not after 'e' and cannot have duplicate '.'
//'0-9': always valid, set number detection flag to be true, and numberAfterE
//to be true
//Other cases: false
//Return isSeenNum and numberAfterE
//Good summary:
//https://leetcode.wang/leetCode-65-Valid-Number.html
class Solution {
public:
    bool isNumber(string s) {
        int i = s.size() - 1;
        while(s[i] == ' ') {
            s.pop_back();
            i--;
        }
            
        i = 0;
        while(s[i] == ' ')
            i++;
        
        //Our startpos is not necessarily to be 0
        int startPos = i;
        
        bool isSeenDigits = false;
        //'.'
        bool isSeenDot = false;
        //we do not need to keep track of '+' or '-' because the only 
        //possible pos for '+/-' is at the very beginning or immediate
        //after 'e'
        //bool isSeenPM = false;
        //'e'
        bool isSeenE = false;
        bool isDigitsAfterE = true;
        
        for(; i < s.size(); ++i){
            if(s[i] >= '0' && s[i] <= '9'){
                isSeenDigits = true;
                isDigitsAfterE = true;
            }else if(s[i] == '+' || s[i] == '-'){
                if(i != startPos && s[i-1] != 'e') return false;
            }else if(s[i] == '.'){
                if(isSeenDot || isSeenE) return false;
                isSeenDot = true;
            }
            else if(s[i] == 'e'){
                if(isSeenE || !isSeenDigits) return false;
                isSeenE = true;
                isDigitsAfterE = false;
            }else{
                return false;
            }
        }
        
        return isSeenDigits && isDigitsAfterE;
    }
};


//Clever state machine solution! You need to draw the state machine diagram first!
//https://leetcode.com/problems/valid-number/discuss/23725/C%2B%2B-My-thought-with-DFA
//State picture: Implemented by myself
/*
[s0 - s1 : +/-], [s0 - s2 : number], [s0 - s3 : '.'],
[s1 - s2 : number], [s1 - s3 : '.'],
[s2 - s2 : number], [s2 - s3 : '.'], [s2 - s4 : 'e'],
[s3 - s4 : 'e'], [s3 - s3 : number],
[s4 - s5 : +/-], [s4 - s6 : number], 
[s5 - s6 : number],
[s6 - s6 : number]
The valid final state: s2, isSeenDigit && s3, s6
*/
class Solution {
public:
    bool isNumber(string s) {
        if(s.empty()) return false;
        int i = s.size() - 1;
        while(s[i] == ' '){
            s.pop_back(); i--;
        }
        
        i = 0;
        while(i < s.size() && s[i] == ' '){
            i++;
        }
        s.erase(0, i);
        if(s.empty()) return false;
        
        int state = 0;
        bool isSeenDigit = false;
        for(int i = 0; i < s.size(); ++i){
            if(s[i] >= '0' && s[i] <= '9'){
                isSeenDigit = true;
                if(state <= 2) state = 2;
                else if(state >= 4) state = 6;
            }else if(s[i] == '+' || s[i] == '-'){
                if(state == 0 || state == 4) state++;
                else return false;
            }else if(s[i] == '.'){
                if(state < 3) state = 3;
                else return false;
            }else if(s[i] == 'e'){
                if(state == 2 || (isSeenDigit && state == 3))
                    state = 4;
                else return false;
            }
            else 
                return false;
        }
        
        return (state == 2) || (isSeenDigit && state == 3) || (state == 6);
    }
};


//68. Text Justification
//https://leetcode.com/problems/text-justification/
//Just run simulation, check each word line by line.
class Solution {
public:
    vector<string> fullJustify(vector<string>& words, int maxWidth) {
        queue<string> Q;
        int len = words.size(); 
        vector<string> res;
        int localLengthW = 0;
        for(int i = 0; i < len;){
            //We need to consider white space
            if(localLengthW + words[i].size() <= maxWidth){
                auto& w = words[i];
                localLengthW += w.size() + 1;
                if(localLengthW > maxWidth) {
                    localLengthW --;
                    Q.push(w);
                }else{
                    w.push_back(' ');
                    Q.push(w);
                }
                ++i;
            }else{
                string tempRes;
                int extraGap = maxWidth - localLengthW;
                //cout << extraGap << endl;
                if(extraGap == 0 && Q.back().back() != ' '){
                    while(!Q.empty()){
                        tempRes.append(Q.front());
                        Q.pop();
                    }
                }else if(Q.size() == 1){
                    tempRes.append(Q.front());
                    tempRes.append(extraGap, ' ');
                    Q.pop();
                }else{
                    int lenQ = Q.size();
                    Q.back().pop_back();
                    //Since extraGap != 0, remember last element we add a
                    //' ' behind, we need to remove that
                    extraGap++;
                    //How many ' ' we need to add for each gap
                    int addOn = extraGap / (lenQ - 1);
                    int posAdd = extraGap % (lenQ - 1);
                    
                    int j = 0;
                    while(!Q.empty()){
                        auto& w = Q.front();
                        if(Q.size() > 1)
                            w.append((j < posAdd ? 1 : 0) + addOn, ' ');
                        tempRes.append(w);
                        Q.pop();
                        j++;
                    }
                }
                res.push_back(tempRes);
                localLengthW = 0;
            }
        }
        string tempRes;
        //Last line
        while(!Q.empty()){
            tempRes.append(Q.front());
            Q.pop();
        }
        int extraSpace = maxWidth - tempRes.size();
        tempRes.append(extraSpace, ' ');
        res.push_back(tempRes);
        return res;
    }
};



//680. Valid Palindrome II
//https://leetcode.com/problems/valid-palindrome-ii/
//Not hard, check skip which character!
class Solution {
public:
    bool validPalindrome(string s) {
        int len = s.size();
        int i = 0, j = len - 1;
        while(i < j){
            if(s[i] == s[j]){
                i++; j--;
            }else
                break;
        }
        if(i >= j) return true;
        
        if(i < j){
            int curI = i, curJ = j;
            i++;
            while(i < j){
                if(s[i] == s[j]){
                    i++; 
                    j--;
                }  
                else 
                    break;
            }
            if(i >= j) return true;
            i = curI;
            j = curJ;
            j--;
            while(i < j){
                if(s[i] == s[j]){
                    i++; j--;
                } 
                else 
                    break;
            }
            if(i >= j) return true;
        }
        return false;
    }
};

//More concise version
bool validPalindrome(string s) {
    for (int i = 0, j = s.size() - 1; i < j; i++, j--)
        if (s[i] != s[j]) {
            int i1 = i, j1 = j - 1, i2 = i + 1, j2 = j;
            while (i1 < j1 && s[i1] == s[j1]) {i1++; j1--;};
            while (i2 < j2 && s[i2] == s[j2]) {i2++; j2--;};
            return i1 >= j1 || i2 >= j2;
        }
    return true;
}



//937. Reorder Data in Log Files
//https://leetcode.com/problems/reorder-data-in-log-files/
//stable_sort + custom comparator!
//Not a hard problem! however, we need to consider the retain stable
//during the sorting process. It's still good to know that there is 
//some way in STL can help us to do so!
//A very weird problem, hard to solve.
class Solution {
public:
    vector<string> reorderLogFiles(vector<string>& logs) {
        auto myComp = [](string a, string b){
                int i = a.find(' ');
                int j = b.find(' ');
                if(!isdigit(a[i+1]) && !isdigit(b[j+1])){
                    if(a.substr(i+1) == b.substr(j+1))
                        //handle ties of two strings
                        return a.substr(0, i+1) < b.substr(0, j+1);
                    else if(a.substr(i+1) < b.substr(j+1))
                        return true;
                }else{
                    //make sure characters go first
                    if(!isdigit(a[i+1]))
                        return true;
                    else
                        return false;
                }
            return false;
        };
        stable_sort(logs.begin(), logs.end(), myComp);
        return logs;
    }
};


//43. Multiply Strings
//https://leetcode.com/problems/multiply-strings/
class Solution {
public:
    string multiply(string num1, string num2) {
        int len1 = num1.size(), len2 = num2.size();
        if(num1 == "0" || num2 == "0")
            return "0";
        //Result will not exceed len1 + len2
        string res(len1 + len2, '0');
        int carry = 0;
        for(int i = len1 - 1; i >= 0; --i){
            carry = 0;
            for(int j = len2-1; j >= 0; --j){
                //here we add all the result together! This is a critical
                //part!
                int tempRes = (num1[i] - '0') * (num2[j] - '0') + carry + (res[i+j+1] - '0');
                res[i+j+1] = (tempRes%10 + '0');
                carry = tempRes / 10;
            }
            res[i] = (carry + '0');
        }
        size_t pos = res.find_first_not_of('0');
        return res.substr(pos);
    }
};



//76. Minimum Window Substring
//https://leetcode.com/problems/minimum-window-substring/
//The implementation is tricky! Logic is not hard
//Better to do 2 passes during the interview!
class Solution {
public:
    string minWindow(string s, string t) {
        if(s.empty() || t.empty() || (t.size() > s.size())) return "";
        int tDict[128] = {0};
        //when required == 0, which means we have find a window which
        //contains all the characters from t
        int required = t.size();
        for(char c : t){
            tDict[c] ++;
        }
        int minLen = INT_MAX;
        //start will be the beginning index of the minimum sub string
        int start = 0, r = 0, l = 0;
        while(r < s.size() && l < s.size()){
            //if required not equal to 0, we need to search right
            if(required){
                tDict[s[r]]--;
                //Note >= here, means we find 1 character from t
                if(tDict[s[r]] >= 0) required--;
                r++;
            }else{
                //when we know our window has all characters, update
                //minLen. Should be r - l here, not r-l+1
                if(minLen > r - l){
                    minLen = r - l;
                    start = l;
                }
                tDict[s[l]]++;
                //we still need some character from s
                if(tDict[s[l]] > 0) required++;
                l++;
            }
        }
        //cout << required << endl;
        while(l < s.size() && required == 0){
            if(minLen > r - l){
                minLen = r - l;
                start = l;
            }
            tDict[s[l]]++;
            //we still need some character from s
            if(tDict[s[l]] > 0) required++;
            l++;
        }
        
        return minLen == INT_MAX ? "" : s.substr(start, minLen);
    }
};



//30. Substring with Concatenation of All Words
//https://leetcode.com/problems/substring-with-concatenation-of-all-words/
//Just do a direct search for each entry i in s. We start from i and check
//each potential substring.
class Solution {
public:
    vector<int> findSubstring(string s, vector<string>& words) {
        vector<int> res;
        if(s.empty() || words.empty()) return res;
        
        int len_s = s.size(), numW = words.size(), len_w = words[0].size();
        if(len_s < len_w * numW) return res;
        
        unordered_map<string, int> wordDict;
        for(auto& word : words){
            wordDict[word]++;
        }
        
        for(int i = 0; i < len_s - numW * len_w + 1; ++i){
            unordered_map<string, int> detect;
            int j = 0;
            for(; j < numW; ++j){
                string tempWord = s.substr(i + j*len_w, len_w);
                if(wordDict.count(tempWord) > 0){
                    detect[tempWord]++;
                    if(detect[tempWord] > wordDict[tempWord])
                        break;
                }else{
                    break;
                }
            }
            if(j == numW) res.push_back(i);
        }
        return res;
    }
};



//434. Number of Segments in a String
//https://leetcode.com/problems/number-of-segments-in-a-string/
class Solution {
public:
    int countSegments(string s) {
        //handle the last case
        s.push_back(' ');
        int len = s.size();
        int i = 0;
        int res = 0;
        while(i < len && s[i] == ' ') i++;
        for(; i < len; ++i){
            if(s[i] == ' '){
                res++;
                while(i < len && s[i] == ' ') i++;
            }
        }
        return res;
    }
};


//438. Find All Anagrams in a String
//https://leetcode.com/problems/find-all-anagrams-in-a-string/
/*
Not very hard! Implemented by myself, get it work at the first try!
*/
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
         vector<int> res;
        if(p.empty() || s.empty()) return res;
        //count characters in p
        int uDict[26] = {0};
        for(char c : p){
            uDict[c - 'a']++;
        }
        int uDetect[26] = {0};
       
        int i = 0;
        for(int j = 0; j < s.size(); ++j){
            char curC = s[j];
            if(uDict[curC- 'a'] == 0){
                i = j+1;
                memset(uDetect, 0, sizeof(uDetect));
            }else{
                if(uDetect[curC-'a'] < uDict[curC-'a']){
                    uDetect[curC-'a']++;
                    
                }else{
                    uDetect[curC-'a']++;
                    while(i < j && uDetect[curC-'a'] > uDict[curC-'a']){
                        uDetect[s[i]-'a']--;
                        i++;
                    }
                }
                if(j - i + 1 == p.size()) {
                    res.push_back(i);
                    uDetect[s[i]-'a'] --;
                    i++;
                }
            }
        }
        return res;
        
    }
};


//833. Find And Replace in String
//https://leetcode.com/problems/find-and-replace-in-string/
//The idea is we sort the index from start to the end, so we can start our
//replacement from right to left. (no need to keep track of where to insert
//for string S). The idea is not complex, need sometime to get it!
class Solution {
public:
    string findReplaceString(string S, vector<int>& indexes, vector<string>& sources, vector<string>& targets) {
        string res;
        vector<pair<int, int>> vMap;
        int len = indexes.size();
        for(int i = 0; i < len; ++i){
            //build the index[i] and ith elements map
            vMap.push_back({indexes[i], i});
        }
        //sort in descending order
        sort(vMap.rbegin(), vMap.rend());
        
        for(int i = 0; i < len; ++i){
            //indexS represents the starting index i in S
            int indexS = vMap[i].first;
            string& src = sources[vMap[i].second];
            string& tar = targets[vMap[i].second];
            //We need to replace now
            if(S.substr(indexS, src.size()) == src){
                string tempS = S.substr(0, indexS);
                tempS.append(tar);
                int k = 0;
                while(k < S.size() - (indexS + src.size())){
                    tempS.push_back(S[k+indexS + src.size()]);
                    k++;
                }
                swap(tempS, S);
            }
        }
        return S;
        
    }
};

class Solution {
public:
    string findReplaceString(string S, vector<int>& indexes, vector<string>& sources, vector<string>& targets) {
        int lenS = S.size();
        //using an array to keep track of the starting index and corresponding sources and
        //target. Very clever!
        int replace[lenS];
        //only works for -1 or 0, no other values!
        memset(replace, -1, sizeof(replace));
        
        for(int i = 0; i < indexes.size(); ++i){
            //A map to map the starting index from S with sources and targets
            replace[indexes[i]] = i;
        }
        
        string res;
        for(int i = 0; i < lenS; ++i){
            if(replace[i] == -1){
                res.push_back(S[i]);
            }else{
                string& src = sources[replace[i]];
                if(S.substr(i, src.size()) == src){
                    //append target to current res
                    res.append(targets[replace[i]]);
                    //since we increment our i for each round, we need to subtract 1 here
                    i += src.size() - 1;
                }else{
                    res.push_back(S[i]);
                }
            }
        }
        
        return res;
        
    }
};



//809. Expressive Words
//https://leetcode.com/problems/expressive-words/
//In general, the problem is not hard. You have to find a way to organize 
//your code well.
//This 4 pointers solution is elegant. Hard to get it right!
class Solution {
private:
    bool checkWords(const string& s, const string& w){
        int len_s = s.size(), len_w = w.size();
        int i = 0, j = 0;
        for(int sI = 0, wJ = 0; i < len_s && j <len_w; i = sI, j = wJ){
            if(s[i] != w[j]) return false;
            while(sI < len_s && s[sI] == s[i]) sI++;
            while(wJ < len_w && w[wJ] == w[j]) wJ++;
            //A little bit tricky here, note when sI - i == wJ - j
            //this condition will be false!
            if(sI - i != wJ - j && sI - i < max(3, wJ-j)) 
                return false;
        }
        //critical here!
        return i == len_s && j == len_w;
    }
public:
    int expressiveWords(string S, vector<string>& words) {
        int res = 0;
        if(S.empty() || words.empty()) return 0;
        for(auto& w : words){
            if(checkWords(S, w)) res++;
        }
        return res;
    }
};



//966. Vowel Spellchecker
//https://leetcode.com/problems/vowel-spellchecker/
//Handle the 3 cases separately!
/*
1. Check whether there is an exactly match
2. Check whether there is upper case 
3. Check vowel (replace all vowels with '#', and create the key!)
*/
class Solution {
public:
    vector<string> spellchecker(vector<string>& wordlist, vector<string>& queries) {
        unordered_set<string> uSet(wordlist.begin(), wordlist.end());
        unordered_map<string, vector<string>> uMap;
        unordered_map<string, vector<string>> uMapVowel;
        
        auto toLowerC = [](char& c){ c = tolower(c); };
        
        for(int i = 0; i < wordlist.size(); ++i){
            //make a copy!
            string s = wordlist[i];
            for_each(s.begin(), s.end(), toLowerC);
            uMap[s].push_back(wordlist[i]);
            
            string sNoVowel;
            for(char c : s){
                if(c != 'a' && c != 'e' && c != 'i' && c != 'o' &&
                   c != 'u'){
                    sNoVowel.push_back(c);
                }else
                    //we need to push the holder here or we cannot tell 
                    //where is the vowel
                    sNoVowel.push_back('#');
            }
            uMapVowel[sNoVowel].push_back(wordlist[i]);
        }
        vector<string> res;
        for(auto& s : queries){
            if(uSet.count(s) > 0) {
                res.push_back(s);
                continue;
            }
            for_each(s.begin(), s.end(), toLowerC);
            
            if(uMap.count(s) > 0) {
                res.push_back(uMap[s][0]);
                continue;
            }
            string sNoVowel;
            //s is lower case now!
            for(char c : s){
                if(c != 'a' && c != 'e' && c != 'i' && c != 'o' &&
                   c != 'u'){
                    sNoVowel.push_back(c);
                }else
                    sNoVowel.push_back('#');
            }
            
            if(uMapVowel.count(sNoVowel) > 0) {
                res.push_back(uMapVowel[sNoVowel][0]);
            }
            else res.push_back("");
        }
        return res;
    }
};


//424. Longest Repeating Character Replacement
//https://leetcode.com/problems/longest-repeating-character-replacement/
/* Sliding window solution
Did not get the idea at first!
https://leetcode.com/problems/longest-repeating-character-replacement/discuss/208284/
Maintain a window with r - l + 1 - maxFreq > k is key to success
*/
class Solution {
public:
    int characterReplacement(string s, int k) {
        int len = s.size();
        //dict to keep track of the frequency of each character
        //within current window
        int dict[26] = {0};
        int maxFreq = 1;
        int l = 0, r = 0;
        int maxLen = 0;
        for(; r < len; ++r){
            dict[s[r] - 'A']++;
            //always update the frequency of R
            maxFreq = max(dict[s[r]-'A'], maxFreq);
            if(r - l + 1 - maxFreq > k){
                maxLen = max(r - l, maxLen);
                dict[s[l]-'A']--;
                l++;
                //Update maxFreq when we decrement dict[l]
                maxFreq = *max_element(dict, dict+26);
            }
        }
        //We need to update maxLen here for the case:
        //"ABAB" 2
        maxLen = max(maxLen, r-l);
        return maxLen;
    }
};


//Optimization! Keep the global maxFreq
class Solution {
public:
    int characterReplacement(string s, int k) {
        int len = s.size();
        //dict to keep track of the frequency of each character
        //within current window
        int dict[26] = {0};
        int maxFreq = 1;
        int l = 0, r = 0;
        int maxLen = 0;
        for(; r < len; ++r){
            dict[s[r] - 'A']++;
            //always update the frequency of R
            maxFreq = max(dict[s[r]-'A'], maxFreq);
            if(r - l + 1 - maxFreq > k){
                maxLen = max(r - l, maxLen);
                dict[s[l]-'A']--;
                l++;
                //No need to update maxFreq here, since if window A has 
                //larger maxFreq than B, then length of A must be longer
                //than B. We only need a global maxFreq, which will be 
                //sufficient since maxFreq only goes up!
                //maxFreq = *max_element(dict, dict+26);
            }
        }
        //We need to update maxLen here for the case:
        //"ABAB" 2
        maxLen = max(maxLen, r-l);
        return maxLen;
    }
};


//467. Unique Substrings in Wraparound String
//https://leetcode.com/problems/unique-substrings-in-wraparound-string/
/* //First try, wrong
class Solution {
private:
    
    int getRes(string& s, vector<int>& cDict){
        int len = s.size();
        int res = (len + 1) * len / 2;
        if(len == 1 && cDict[s[0] - 'a'] == 1) return 0;
        for(char c : s){
            if(cDict[c - 'a'] != 1){
                cDict[c - 'a'] = 1;
            }else{
                res--;
            }
        }
        return res;
    }
public:
    int findSubstringInWraproundString(string p) {
        int len = p.size();
        if(len == 0) return 0;
        int res = 0;
        string tempStr;
        unordered_set<string> uSet;
        vector<int> chaDict(26, 0);
        //p_01 stores p without repeating character
        string p_01;
        for(int i = 0; i < len; ++i){
            if(i>0 && p[i] == p[i-1])
                continue;
            p_01.push_back(p[i]);
        }
        p.swap(p_01);
        len = p.size();
        
        for(int i = 0; i < len; ++i){
            if(i == 0 || p[i-1] == 'z' && p[i] == 'a' || p[i] == (p[i-1]+1)){
                tempStr.push_back(p[i]);
            }else{
                uSet.insert(tempStr);
                //cout << tempStr << endl;
                //reset tempStr here
                tempStr = "";
                tempStr.push_back(p[i]);
            }
        }
        uSet.insert(tempStr);
        //cout << tempStr << endl;
        for(auto it = uSet.begin(); it != uSet.end(); ++it){
            string s = *it;
            //cout << s << endl;
            //int len_s = getLen(s, chaDict);
            res += getRes(s, chaDict);
        }
        return res;
    }
};
*/

/*
Great explanation: 
https://leetcode.com/problems/unique-substrings-in-wraparound-string/discuss/95454/
*/
//O(n^2) with unordered_set 
//cannot pass OJ: TLE
class Solution {
private:
    struct Hash{
        size_t operator()(const pair<char, int>& p) const {
            return hash<string>()(to_string(p.second)+ '_' + p.first);
        }
    };
public:
    int findSubstringInWraproundString(string p) {
        int len = p.size();
        //set to reduce duplicate element!
        unordered_set<pair<char, int>, Hash> cSet;
        for(int i = 0; i < len; ++i){
            for(int j = i; j < len; ++j){
                //p[j-1]-p[j] == 25 --- 'za'
                //Here we must have j > i (instead of j > 0)
                //since for each entry i, we need to at least examine it 
                //once!
                if(j > i && p[j] != p[j-1]+1 && p[j-1] - p[j] != 25)
                    break;
                //note for each character c, if it has diffferent substring
                //length, then we need to insert it to our set
                //duplicate will be handled by set!
                cSet.insert({p[j], j-i+1});
            }
        }
        return cSet.size();
    }
};

//O(n^2), no hashing, passed OJ, slow!
class Solution {
public:
    int findSubstringInWraproundString(string p) {
        int len = p.size();
        //thid dict saves the potential maximum length of the substr start
        //with character cDict[i]; We only need to take of maximum length,
        //since the shorter length has already been covered by longer 
        //length! The length means starting with character cDict[i], how
        //many substrings (with different length) we can get !
        int cDict[26] = {0};
        for(int i = 0; i < len; ++i){
            for(int j = i; j < len; ++j){
                if(j > i && p[j] != p[j-1]+1 && p[j-1]-p[j] != 25)
                    break;
                cDict[p[j]-'a'] = max(cDict[p[j]-'a'], j-i+1);
            }
        }
        
        return accumulate(cDict, cDict+26, 0);
    }
};

//O(n) solution, tricky!!
class Solution {
public:
    int findSubstringInWraproundString(string p) {
        int len = p.size();
        int cDict[26] = {0};
        //start index of i
        int i = 0;
        for(int j = i; j < len; ){
            if(j > i && p[j] != p[j-1]+1 && p[j-1]-p[j] != 25){
                //Here i+26 means we at most need to update 26 characters
                //if the substring length is greater than 26, then it must
                //have included all characters
                for(int k = i; k < min(i+26, j); ++k)
                    cDict[p[k]-'a'] = max(cDict[p[k]-'a'], j-k);
                i = j;
            }else
                ++j;
        }
        //Handle the situation when last substring perfectly match, then
        //we miss one case
        for(int k = i; k < min(i+26, len); ++k){
            cDict[p[k]-'a'] = max(cDict[p[k]-'a'], len-k);
        }

        return accumulate(cDict, cDict+26, 0);
    }
};

//Similar O(n) solution, but very hard to understand at first glance
class Solution {
public:
    int findSubstringInWraproundString(string p) {
        int len = p.size();
        int cDict[26] = {0};
        int localLen = 0;
        int res = 0;
        for(int i = 0; i < len; ++i){
            int cur = p[i] - 'a';
            if(i > 0 && p[i-1] != (cur + 26 - 1)%26 + 'a') localLen = 0;
            if(++localLen > cDict[cur]){
                res += localLen - cDict[cur];
                cDict[cur] = localLen;
            }
        }
        return res;
    }
};



//423. Reconstruct Original Digits from English
//https://leetcode.com/problems/reconstruct-original-digits-from-english/
//Interesting question with not so intuitive solution! Identify the uniqueness
//of each number is key to success
class Solution {
public:
    string originalDigits(string s) {
        const string numDict[] = {"zero", "two", "four", "five", "six", "seven", "eight", "three", "nine", "one"};
        const char id[] = {'z', 'w', 'u', 'f', 'x', 'v', 'g', 't', 'i', 'o'};
        const int numMap[] = {0,2,4,5,6,7,8,3,9,1};
        //This array will count how many nums in our final result
        int numArray[10] = {0};
        int uDict[26] = {0};
        unsigned int len = s.size();
        string res;
        
        for(char c : s) { uDict[c - 'a']++; }
        
        for(int i = 0; i < 10; ++i){
            int countNum = uDict[id[i] - 'a'];
            for(int j = 0; j < numDict[i].size(); ++j){
                uDict[numDict[i][j] - 'a'] -= countNum;
            }
            numArray[numMap[i]] += countNum;
        }
        
        for(int i = 0; i < 10; ++i){
            int localCount = numArray[i];
            while(localCount > 0){
                res.push_back(i + '0');
                localCount--;
            }
        }
        return res;
    }
};


//524. Longest Word in Dictionary through Deleting
//https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/
/* Brute force approach. Not that efficient! 
Sorting the array d and do some early prune might be more efficient!
like start with the longest the string, eliminate string shorter than
local res's length*/
class Solution {
public:
    string findLongestWord(string s, vector<string>& d) {
        int len_s = s.size();
        int len_d = d.size();
        string res = "";
        for(int i = 0; i < len_d; ++i){
            if(d[i].size() < res.size()) continue;
            int p = 0, q = 0;
            while(p < len_s && q < d[i].size()){
                if(s[p] == d[i][q])
                    q++;
                p++;
            }
            //save word as ans if the three conditions are met: 
                //1. k==n: if every letter of word exists in string s (and in order of the word)
                //2. ans.size()<n: word is a longer word than ans 
                //3. (ans.size()==n && ans>word): word is same size as ans but is lexicographically smaller
            //REMEMBER: "return the longest word with the smallest lexicographical order."
            if(q == d[i].size() && ((res.size() == d[i].size() && d[i] < res) || res.size() < d[i].size())){
                res = d[i];
            }
        }
        return res;
    }
};


//125. Valid Palindrome
//https://leetcode.com/problems/valid-palindrome/
class Solution {
public:
    bool isPalindrome(string s) {
        if(s.empty()) return true;
        int i = 0, j = s.size()-1;
        while(i < j){
            while(isalnum(s[i]) == 0 && i < j) i++;
            while(isalnum(s[j]) == 0 && i < j) j--;
            if(toupper(s[i]) != toupper(s[j])) return false;
            i++;
            j--;
        }
        return true;
    }
};


//9. Palindrome Number
//https://leetcode.com/problems/palindrome-number/
//The most intuitive way is to convert it to string! 
class Solution {
public:
    bool isPalindrome(int x) {
        int preX = x;
        int val = 0;
        if(x < 0) return false;
        if(x < 10) return true;
        //All 10, 20 etc. is impossible!
        //We need to add the condition here, or the following code will
        //be wrong when we have test case like 10
        if(x % 10 == 0) return false;
        //rebuild half of the integer
        //121 .. val = 12, prex = 12, x = 1
        //in the end
        while(x > val){
            val = val * 10 + x % 10;
            preX = x;
            x /= 10;
        }
        
        return (preX == val) || (x == val);
        
    }
};

class Solution {
public:
    bool isPalindrome(int x) {
        if(x<0)
            return false;
        else if(x>=0 && x<9)
            return true;
        int temp = x;
        long reverse = 0;
        while(temp !=0){
            reverse = reverse*10 + temp%10;
            temp = temp/10;
        }
        return reverse == x;
    }
};



//459. Repeated Substring Pattern
//https://leetcode.com/problems/repeated-substring-pattern/
/*
//First try, wrong. Made the wrong assumption that a pattern cannot have
//duplicate characters!
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        if(s.size() < 2) return false;
        int uDict[26] = {0};
        int cnt = 0;
        for(char c : s){
            uDict[c - 'a']++;
            cnt = max(cnt, uDict[c-'a']);
        }
        for(int i = 0; i < 26; ++i){
            if(uDict[i] != 0 && uDict[i] != cnt){
                return false;
            }
        }
        int ssLen = s.size() / cnt;
        string pattern = s.substr(0, ssLen);
        string tempStr;
        while(cnt > 0){
            tempStr.append(pattern);
            cnt--;
        }
        //cout << tempStr << endl;
        return tempStr == s;
    }
};
*/

//Solution 1: We shift the string by i position in order to check wheather 
//the remaining will be a match! Not that intuitive for me!
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int len = s.size();
        if(len < 2) return false;
        for(int i = 1; i <= len/2; ++i){
            if(len % i == 0 && s.substr(i) == s.substr(0, len-i))
                return true;
        }
        return false;
    }
};

//Solution 2: KMP algorithm
/*
First, we build the KMP table.

Roughly speaking, dp[i+1] stores the maximum number of characters that the
string is repeating itself up to position i.
Therefore, if a string repeats a length 5 substring 4 times, then the last
entry would be of value 15.
To check if the string is repeating itself, we just need the last entry to
be non-zero and str.size() to divide (str.size()-last entry).
*/
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int len = s.size();
        if(len < 2) return false;
        int i = 1, j = 0;
        int dp[len] = {0};
        while(i < len){
            if(s[i] == s[j]) {
                dp[i] = ++j;
                i++;
            }
            else if(j == 0) ++i;
            else
                j = dp[j-1];
        }
        /*
        for(int i = 0; i <= len; ++i){
            cout << i << " " << dp[i] << endl;
        }*/
        return dp[len-1] && (len % (len - dp[len-1]) == 0);
    }
};


//451. Sort Characters By Frequency
//https://leetcode.com/problems/sort-characters-by-frequency/
//Bucket sort
//Easy to get O(nlogn). O(n) solution should be preferred!
class Solution {
public:
    string frequencySort(string s) {
        int len_s = s.size();
        int Dict[128] = {0};
        for(char c : s){
            Dict[c]++;
        }
        //Create a bucket to store frequency. Note frequency <= len_s
        vector<string> bucket(len_s+1);
        for(int i = 0; i < 127; ++i){
            if(Dict[i] > 0){
                bucket[Dict[i]].append(Dict[i], static_cast<char>(i));
            }
        }
        string res;
        for(int i = len_s; i >= 0; --i){
            res.append(bucket[i]);
        }
        return res;
        
    }
};



//159. Longest Substring with At Most Two Distinct Characters
//https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/
/* With hash map */
class Solution {
public:
    int lengthOfLongestSubstringTwoDistinct(string s) {
        unsigned int len = s.size();
        unordered_map<char, int> uMap;
        int i = 0;
        int maxLen = 0;
        for(int j = 0; j < len; ++j){
            uMap[s[j]]++;
            while(uMap.size() > 2){
                uMap[s[i]]--;
                if(uMap[s[i]] == 0) uMap.erase(s[i]);
                i++;
            }
            maxLen = max(maxLen, j - i + 1);
        }
        return maxLen;
    }
};

//Same idea, with a common array. Faster! Not implemented by me!
class Solution {
    int freq[257];
public:
    int lengthOfLongestSubstringTwoDistinct(string s) {
        int sz = s.size();
        if(sz <= 1)
            return sz;
        for(int i = 0; i < 257; i++)
            freq[i] = 0;
        int st,ed;
        st = 0;
        ed = 1;
        int distinct_count = 1;
        freq[s[st]]++;
        int ans = 1;
        while(ed < sz){
            freq[s[ed]]++;
            if(freq[s[ed]] == 1)
                distinct_count++;
            while(st <= ed && distinct_count > 2){
                freq[s[st]]--;
                if(freq[s[st]] == 0)
                    distinct_count--;
                st++;
            }
            ans = max(ans, ed - st + 1);
            ed++;
        }
        return ans;
    }
};



//681. Next Closest Time
//https://leetcode.com/problems/next-closest-time/
//A tricky problem, not easy to immediately get the solution!
class Solution {
public:
    string nextClosestTime(string time) {
        string res = time;
        set<char> charSet;
        //we will sort on the fly
        for(char c : time){
            if(c == ':') continue;
            charSet.insert(c);
        }
        
        int len = time.size();
        for(int i = len-1; i >= 0; --i){
            char c = time[i];
            if(c == ':') continue;
            auto it = charSet.find(c);
            //if this character is not the maximum one
            if(c != *(charSet.rbegin())){
                ++it;
                res[i] = *it;
                //cout << res[i] << endl;
                if((i >= 3 && stoi(res.substr(i, 2)) <= 59) || (i <= 2 && stoi(res.substr(0, 2)) <= 23))
                    return res;
            }
            //make current character the smallest
            res[i] = *(charSet.begin());
        }
        return res;
        
    }
};



//7. Reverse Integer
//https://leetcode.com/problems/reverse-integer/
class Solution {
private:
    bool checkValid(string& s1, string& s2){
        if(s1.size() < s2.size()) return true;
        if(s1.size() == s2.size()){
            //Be careful about when we need to return true
            for(int i = 0; i < s1.size(); ++i){
                if(s1[i] < s2[i]) return true;
                if(s1[i] > s2[i]) return false;
            }
        }
        return true;
    }
public:
    int reverse(int x) {
        //We need to handle the corner case when x cannot have valid abs value
        if(x == INT_MIN) return 0;
        int temp = abs(x);
        string s = to_string(temp);
        std::reverse(s.begin(), s.end());
        
        
        string upperBound = to_string(2147483647);
        string lowerBound = upperBound;
        lowerBound.back() += 1;
        //cout << lowerBound << endl;
        //cout << s << endl;
        
        if(x >= 0 && checkValid(s, upperBound)){
            return stoi(s);
        }
        
        if(x < 0 && checkValid(s, lowerBound)){
            if(s == lowerBound) return INT_MIN;
            return stoi(s) * (-1);
        }
        return 0;
    }
};

//elegant solution: utilize the property of unsigned int
class Solution {
public:
    int reverse(int x) {
        int s;
        unsigned xu;
        if (x >= 0) {
            s = 1;
            xu = x;
        }
        else {
            s = -1;
            xu = -1 * unsigned(x);
        }

        
        int y = 0;
        while (xu > 0) {
            int r = xu % 10;
            if (y > 214748364) {
                return 0;
            }
            y *= 10;
            y += r;
            xu /= 10;
        }
        
        return s * y;
    }
};


//246. Strobogrammatic Number
//https://leetcode.com/problems/strobogrammatic-number/
//Not hard
class Solution {
public:
    bool isStrobogrammatic(string num) {
        if(num.empty()) return true;
        const char candidates[10] = {'0', '1', '*', '*', '*', '*', '9', '*', '8', '6'};
        int len = num.size();
        
        string copy = num;
        reverse(copy.begin(), copy.end());
        for(int i = 0; i < len; ++i){
            char c = copy[i];
            if(candidates[c - '0'] == '*') return false;
            else copy[i] = candidates[c - '0'];
        }
        
        return copy == num;
    }
};


//With early termination. Other person's code
class Solution {
public:
    bool isStrobogrammatic(string num) {
        int length = num.size();
        for (int i = 0; i < length; i++) {
            switch (num[i]) {
                case '0':                    
                case '1': 
                case '8':
                    if (num[length-i-1] != num[i]) {
                        return false;
                    }
                    break;
                case '6':
                    if (num[length-i-1] != '9') {
                        return false;
                    }
                    break;                    
                case '9' :                    
                    if (num[length-i-1] != '6') {
                        return false;
                    }
                    break;
                default:
                    return false;
            }
        }
        return true;
    }
};


//247. Strobogrammatic Number II
//https://leetcode.com/problems/strobogrammatic-number-ii/
//Back tracking + DFS. My implementation.
class Solution {
private:
    const char candidates[10] = {'0', '1', '*', '*', '*', '*', '9', '*', '8', '6'};
    const char cand[5] = {'0', '1', '6', '8', '9'};
    
    void helper(string tempStr, vector<string>& res, int len, bool isOdd){
        if(len < 0) return;
        if(len == 0){
            if(isOdd){
                tempStr.push_back('0');
                res.push_back(tempStr);
                tempStr.back() = '1';
                res.push_back(tempStr);
                tempStr.back() = '8';
                res.push_back(tempStr);
            }else
                res.push_back(tempStr);
            return;
        }
        
        for(int i = 0; i < 5; ++i){
            if(cand[i] == '0' && tempStr.empty()) continue;
            tempStr.push_back(cand[i]);
            helper(tempStr, res, len-1, isOdd);
            tempStr.pop_back();
        }
    }   
    
public:
    vector<string> findStrobogrammatic(int n) {
        vector<string> res;
        if(n < 0) return res;
        if(n == 0) {
            res.push_back("");
            return res;
        }
        bool isOdd = n % 2 == 1;
        int len = n / 2;
        string tempStr;
        helper(tempStr, res, len, isOdd);
        
        for(auto& s : res){
            
            for(int i = len-1; i >= 0; --i){
                s.push_back(candidates[s[i]-'0']);
            }
            //cout << s << endl;
        }
        return res;
    }
};

/* Other's code. Note the optimization part is we resize the tempStr to be
length of n. Then we do not need the outer loop to do the calculation any more.
Another thing is that we will not have array resizing any more! */
class Solution {
public:
    vector<string> findStrobogrammatic(int n) {
        vector<string> ans;
        vector<char> nums = {'0', '1', '6', '9', '8'};
        string cur(n, '0');
        dfs(n, nums, 0, cur, ans);
        return ans;
    }
private:
    void dfs(int n, const vector<char> &nums, int s, string &cur, vector<string>& ans) {
        if (n % 2 == 0 && s > (n / 2 - 1)) {
            ans.push_back(cur);
            return;
        }

        if (n % 2 == 1 && s > (n/2)) {
            ans.push_back(cur);
            return;
        }

        for (int i = 0; i < nums.size(); ++i) {
            if (n > 1 && s == 0 && nums[i] == '0') continue; // MISTAKE: forget about when n == 1, we can use '0'. so need to add n > 1
            if ((n == 1 || (s == n - 1 - s)) && (nums[i] == '6' || nums[i] == '9')) continue; // MISTAKE: forget this case. if n == 1. it can't be 6 or 9

            // fill s and n - 1 -s
            cur[s] = nums[i];
            if (nums[i] == '6')
                cur[n - 1 - s] = '9';
            else if (nums[i] == '9') {
                cur[n - 1 - s] = '6';
            } else {
                cur[n - 1 - s] = cur[s];
            }

            dfs(n, nums, s + 1, cur, ans);
        }
    }
};



//271. Encode and Decode Strings
//https://leetcode.com/problems/encode-and-decode-strings/
//Not that efficient, but easy to understand!
class Codec {
public:

    // Encodes a list of strings to a single string.
    string encode(vector<string>& strs) {
        string res;
        int len = strs.size();
        if(len == 0) return res;
        for(int i = 0; i < len; ++i){
            int lenS = strs[i].size();
            res.append(to_string(lenS));
            res.push_back('#');
            res.append(strs[i]);
        }
        return res;
    }

    // Decodes a single string to a list of strings.
    vector<string> decode(string s) {
        vector<string> res;
        cout << s << endl;
        int len = s.size();
        int i = 0;
        while(i < len){
            string lenS;
            while(isdigit(s[i])) lenS.push_back(s[i++]);
            int localLen = stoi(lenS);
            i++; // skip # 
            string localStr;
            for(int j = 0; j < localLen; ++j){
                localStr.push_back(s[i]);
                i++;
            }
            res.emplace_back(localStr);
        }
        
        return res;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec codec;
// codec.decode(codec.encode(strs));



//5. Longest Palindromic Substring
//https://leetcode.com/problems/longest-palindromic-substring/
//Extend from the center, implemented by me
class Solution {
public:
    string longestPalindrome(string s) {
        int lenS = s.size();
        if(lenS <= 1) return s;
        
        string newStr;
        newStr.push_back('#');
        for(int i = 0; i < lenS; ++i){
            newStr.push_back(s[i]);
            newStr.push_back('#');
        }
        
        lenS = newStr.size();
        int maxLen = 0;
        int start = 0;
        for(int i = 0; i < lenS; ++i){
            int sIndex = i, eIndex = i;
            while(sIndex >= 0 && eIndex < lenS){
                if(newStr[sIndex] == newStr[eIndex]){
                    sIndex--;
                    eIndex++;
                }else
                    break;
            }
            sIndex++;
            if(maxLen < eIndex - sIndex){
                start = sIndex;
                maxLen = eIndex - sIndex;
            }
        }
        string tempStr = newStr.substr(start, maxLen);
        string res;
        for(int i = 0; i < tempStr.size(); ++i){
            if(tempStr[i] != '#')
                res.push_back(tempStr[i]);
        }
        //cout << tempStr << endl;
        return res;
    }
};


//DP, not that efficient
class Solution {
public:
    string longestPalindrome(string s) {
        if(s == "")
            return s;
        
        int len = s.size();
        int maxsLen = 0;
        bool arr[len][len];
        string finalString;
        
        for(int i = 1; i <= len; i++)
        {
            for(int start = 0; start < len; start++)
            {
                int end = start + i - 1;
                if(end >= len)
                    break;
                
                arr[start][end] = (i == 1 || i == 2 || arr[start + 1][end - 1])&&(s[start] == s[end]);
                if(arr[start][end] && i > maxsLen)
                {
                    finalString = s.substr(start, i);
                    maxsLen = i;
                    //cout << maxsLen;
                    //maxsLen = i;
                }
                   
            }
        }
        return finalString;
        
    }
};


//Manacher's algorithm
//Ref: https://leetcode.wang/leetCode-5-Longest-Palindromic-Substring.html
class Solution {
public:
    
    string preProcess(string s)
    {
        int len = s.size();
        string proStr = "^";
        //string numberStr = "#";
        //string dollarStr = "$";
        
        for(int i = 0; i < len; i++)
        {
            proStr.push_back('#');
            proStr.push_back(s[i]);
        }
        proStr.push_back('#');
        proStr.push_back('$');
        
        return proStr;
        
    }
    
    string longestPalindrome(string s) {
        int len = s.size();
        if(len == 0)
            return s;
        string pS = preProcess(s);
        int len_pS = pS.size();
        
        int C = 0, R = 0, i_mirror = 0;
        int arr[len_pS];
        
        for(int i = 0; i < len_pS; i++)
        {
            i_mirror = 2 * C - i;
            
            if(i < R)
            {
                arr[i] = min(R - i, arr[i_mirror]);
            }
            else
            {
                arr[i] = 0;
            }
            
            while(pS[i+1 + arr[i]] == pS[i - 1 - arr[i]])
            {
                arr[i]++;
            }
            
            if(i + arr[i] > R)
            {
                C = i;
                R = i + arr[i];
            }
            
        }
        
        int maxLen = 0, start = 0;
        for(int i = 0; i < len_pS; i++)
        {
            if(arr[i]>maxLen){
                maxLen = arr[i];
                start = i;
            }
                
        }
        start = (start - maxLen)/2;
        
        return s.substr(start, maxLen);
    }
};

