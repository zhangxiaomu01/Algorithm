/**
 * @file Hash.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2023-01-16
 * 
 * A quick second pass with the common 'Hash map / set' related algorithm problems.
 */
 /*
    242. Valid Anagram
    https://leetcode.com/problems/valid-anagram/

    Given two strings s and t, return true if t is an anagram of s, and false otherwise.

    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

    Example 1:
    Input: s = "anagram", t = "nagaram"
    Output: true

    Example 2:
    Input: s = "rat", t = "car"
    Output: false
    
    Constraints:
    1 <= s.length, t.length <= 5 * 104
    s and t consist of lowercase English letters.
    
    Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?
 */
class Solution {
public:
    bool isAnagram(string s, string t) {
        int res = 0;
        if(s.size() != t.size()) return false;
        vector<int> dict(256, 0); // Change it to an unordered_map if inputs have Unicode characters.
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

 /*
    49. Group Anagrams
    https://leetcode.com/problems/group-anagrams/

    Given an array of strings strs, group the anagrams together. You can return the answer in any order.

    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

    Example 1:
    Input: strs = ["eat","tea","tan","ate","nat","bat"]
    Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

    Example 2:
    Input: strs = [""]
    Output: [[""]]

    Example 3:
    Input: strs = ["a"]
    Output: [["a"]]
    

    Constraints:
    1 <= strs.length <= 104
    0 <= strs[i].length <= 100
    strs[i] consists of lowercase English letters.
 */
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> hashMap;
        vector<string> temp(strs); // Copy constructor
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

 /*
    438. Find All Anagrams in a String
    https://leetcode.com/problems/find-all-anagrams-in-a-string/

    Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order.
    An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

    Example 1:
    Input: s = "cbaebabacd", p = "abc"
    Output: [0,6]
    Explanation:
    The substring with start index = 0 is "cba", which is an anagram of "abc".
    The substring with start index = 6 is "bac", which is an anagram of "abc".

    Example 2:
    Input: s = "abab", p = "ab"
    Output: [0,1,2]
    Explanation:
    The substring with start index = 0 is "ab", which is an anagram of "ab".
    The substring with start index = 1 is "ba", which is an anagram of "ab".
    The substring with start index = 2 is "ab", which is an anagram of "ab".

    Constraints:
    1 <= s.length, p.length <= 3 * 104
    s and p consist of lowercase English letters.
 */
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> res;
        if (p.size() > s.size()) return res;
        int uDict[26] = {0};
        for(char c : p) {
            uDict[c - 'a']++;
        }

        int uDetect[26] = {0};
        int i = 0;
        for(int j = 0; j < s.size(); ++j) {
            if (uDict[s[j] - 'a'] == 0) {
                i = j + 1;
                memset(uDetect, 0, sizeof(uDetect));
                continue;
            }

            uDetect[s[j] - 'a'] ++;
            while(i < j && uDetect[s[j] - 'a'] > uDict[s[j] - 'a']) {
                uDetect[s[i] - 'a']--;
                i++;
            }

            if (j - i + 1 == p.size()) res.push_back(i);
        }
        return res;
    }
};

 /*
    1002. Find Common Characters
    https://leetcode.com/problems/find-common-characters/

    Given a string array words, return an array of all characters that show up in all strings within the words (including duplicates). You may return the answer in any order.

    Example 1:
    Input: words = ["bella","label","roller"]
    Output: ["e","l","l"]

    Example 2:
    Input: words = ["cool","lock","cook"]
    Output: ["c","o"]
    
    Constraints:
    1 <= words.length <= 100
    1 <= words[i].length <= 100
    words[i] consists of lowercase English letters.
 */
class Solution {
public:
    vector<string> commonChars(vector<string>& words) {
        unordered_map<string, int> uMap;
        vector<string> res;
        for (int i = 0; i < words.size(); ++i) {
            int uDict[26] = {0};
            for (int j = 0; j < words[i].size(); ++j) {
                uDict[words[i][j] - 'a'] ++;
                string key = string(1, words[i][j]);
                if (uDict[words[i][j] - 'a'] >= 1) {
                    key += to_string(uDict[words[i][j] - 'a'] - 1);
                }
                uMap[key]++;
            }
        }
        for (auto it = uMap.begin(); it != uMap.end(); ++it) {
            if (it->second == words.size()) {
                if (it->first.size() == 1) res.push_back(it->first);
                else res.push_back(it->first.substr(0, 1));
            }
        }
        return res;
    }
};

// A more elegant solution from Leetcode: we do not need to save string in the map, we only care about how many characters
// has appeared in all words.
class Solution {
public:
    vector<string> commonChars(vector<string>& A) {
        vector<string> result;
        if (A.size() == 0) return result;
        int hash[26] = {0}; // 用来统计所有字符串里字符出现的最小频率
        for (int i = 0; i < A[0].size(); i++) { // 用第一个字符串给hash初始化
            hash[A[0][i] - 'a']++;
        }

        int hashOtherStr[26] = {0}; // 统计除第一个字符串外字符的出现频率
        for (int i = 1; i < A.size(); i++) {
            memset(hashOtherStr, 0, 26 * sizeof(int));
            for (int j = 0; j < A[i].size(); j++) {
                hashOtherStr[A[i][j] - 'a']++;
            }
            // 更新hash，保证hash里统计26个字符在所有字符串里出现的最小次数
            for (int k = 0; k < 26; k++) {
                hash[k] = min(hash[k], hashOtherStr[k]);
            }
        }
        // 将hash统计的字符次数，转成输出形式
        for (int i = 0; i < 26; i++) {
            while (hash[i] != 0) { // 注意这里是while，多个重复的字符
                string s(1, i + 'a'); // char -> string
                result.push_back(s);
                hash[i]--;
            }
        }

        return result;
    }
};

 /*
    202. Happy Number
    https://leetcode.com/problems/happy-number/

    Write an algorithm to determine if a number n is happy.

    A happy number is a number defined by the following process:

    Starting with any positive integer, replace the number by the sum of the squares of its digits.
    Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
    Those numbers for which this process ends in 1 are happy.
    Return true if n is a happy number, and false if not.

    

    Example 1:
    Input: n = 19
    Output: true
    Explanation:
    12 + 92 = 82
    82 + 22 = 68
    62 + 82 = 100
    12 + 02 + 02 = 1

    Example 2:
    Input: n = 2
    Output: false
    

    Constraints:
    1 <= n <= 231 - 1
 */
class Solution {
private:
    int getSum(int n) {
        int res = 0;
        while(n) {
            int next = n % 10;
            res += next * next;
            n = n / 10;
        }
        return res;
    }
public:
    bool isHappy(int n) {
        unordered_set<int> uSet;
        int sum = n;
        // If we detect duplicates, then we know it's not a happy number.
        while(sum != 1) {
            sum = getSum(sum);
            if (uSet.find(sum) != uSet.end() && sum != 1) return false;
            uSet.insert(sum);
        }
        return true;
    }
};

// Sligtly optimized version:
class Solution {
private:
    //Be careful, vector<int>(10, 0) can only be used in a method,
    //not here, out of a function, you need to declare the initialization
    //like this
    vector<int> dict{vector<int>(10, 0)};
public:
    int getNext(int n){
        string num = to_string(n);
        int count = 0;
        for(auto c: num){
            count += dict[c - '0'];
        }
        return count;
    }
    bool isHappy(int n) {
        for(int i = 0; i < 10; i++)
            dict[i] = i*i;
        
        int slow = n, fast = n;
        while(slow!=1 && fast != 1){
            slow = getNext(slow);
            fast = getNext(getNext(fast));
            // An important observation, when loop starts, they will keep looping.
            if(slow == fast && slow!=1 && fast!=1)
                return false;
        }
        return true;
    }
};

