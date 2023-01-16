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
