/**
 * @file String.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2023-01-22
 * 
 * A quick second pass with the common 'String' algorithm problems.
 * 
 */

 /*
    344. Reverse String
    https://leetcode.com/problems/reverse-string/
 
    Write a function that reverses a string. The input string is given as an array of characters s.
    You must do this by modifying the input array in-place with O(1) extra memory.

    
    Example 1:
    Input: s = ["h","e","l","l","o"]
    Output: ["o","l","l","e","h"]

    Example 2:
    Input: s = ["H","a","n","n","a","h"]
    Output: ["h","a","n","n","a","H"]
    

    Constraints:
    1 <= s.length <= 10^5
    s[i] is a printable ascii character.
 */
class Solution {
public:
    void reverseString(vector<char>& s) {
        int l = 0, r = s.size() - 1;
        while (l <= r) {
            swap(s[l++], s[r--]);
        }
    }
};

// Recursive way: which result space complexity to be O(n);
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

 /*
    541. Reverse String II
    https://leetcode.com/problems/reverse-string-ii/
 
    Given a string s and an integer k, reverse the first k characters for every 2k characters counting from the start of the string.
    If there are fewer than k characters left, reverse all of them. If there are less than 2k but greater than or equal to k characters, then reverse the first k characters and leave the other as original.
    
    Example 1:
    Input: s = "abcdefg", k = 2
    Output: "bacdfeg"

    Example 2:
    Input: s = "abcd", k = 2
    Output: "bacd"
    

    Constraints:
    1 <= s.length <= 10^4
    s consists of only lowercase English letters.
    1 <= k <= 10^4
 */
class Solution {
private:
    void reverseString(string& s, int start, int end) {
        while(start <= end) {
            swap(s[start++], s[end--]);
        }
    }
public:
    string reverseStr(string s, int k) {
        for(int i = 0; i < s.size(); i += 2*k) {
            if (s.size() - i < k) {
                reverseString(s, i, s.size() - 1);
            } else {
                reverseString(s, i, i + k - 1);
            }
        }
        return s;
    }
};

 /*
    151. Reverse Words in a String
    https://leetcode.com/problems/reverse-words-in-a-string/
 
    Given an input string s, reverse the order of the words.
    A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.
    Return a string of the words in reverse order concatenated by a single space.
    Note that s may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

    

    Example 1:
    Input: s = "the sky is blue"
    Output: "blue is sky the"

    Example 2:
    Input: s = "  hello world  "
    Output: "world hello"
    Explanation: Your reversed string should not contain leading or trailing spaces.

    Example 3:
    Input: s = "a good   example"
    Output: "example good a"
    Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.
    

    Constraints:
    1 <= s.length <= 104
    s contains English letters (upper-case and lower-case), digits, and spaces ' '.
    There is at least one word in s.
    

    Follow-up: If the string data type is mutable in your language, can you solve it in-place with O(1) extra space?
 */
// NO extra space, O(n) solution.
class Solution {
private:
    void removeExtraSpace(string& s) {
        int slow = 0;
        for(int i = 0; i < s.size(); ++i) {
            if (s[i] != ' ') {
                s[slow++] = s[i];
            } else if (i > 0 && s[i] == ' ' && s[i-1] != ' ') {
                s[slow++] = ' ';
            }
        }
        if (slow > 0 && s[slow - 1] == ' ') s.resize(slow - 1);
        else s.resize(slow);
    }
    void reverseString(string& s, int start, int end) {
        while(start < end) {
            swap(s[start ++], s[end --]);
        }
    }
public:
    string reverseWords(string s) {
        removeExtraSpace(s);
        reverseString(s, 0, s.size() - 1);
        int nextStart = 0;
        s.push_back(' ');
        for(int i = 0; i < s.size(); ++i) {
            if (s[i] == ' ') {
                reverseString(s, nextStart, i - 1);
                nextStart = i + 1;
            }
        }
        s.pop_back();
        return s;
    }
};

// O(n) space, with istringstream, fancy.
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

 /*
    796. Rotate String
    https://leetcode.com/problems/rotate-string/
 
    Given two strings s and goal, return true if and only if s can become goal after some number of shifts on s.
    A shift on s consists of moving the leftmost character of s to the rightmost position.
    For example, if s = "abcde", then it will be "bcdea" after one shift.
    

    Example 1:
    Input: s = "abcde", goal = "cdeab"
    Output: true

    Example 2:
    Input: s = "abcde", goal = "abced"
    Output: false
    

    Constraints:
    1 <= s.length, goal.length <= 100
    s and goal consist of lowercase English letters.
 */
class Solution {
public:
    bool rotateString(string s, string goal) {
        string test = s + s;
        return (s.size() == goal.size()) && (test.find(goal) != string::npos);
    }
};

 /*
    28. Find the Index of the First Occurrence in a String
    https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/
 
    Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

    Example 1:
    Input: haystack = "sadbutsad", needle = "sad"
    Output: 0
    Explanation: "sad" occurs at index 0 and 6.
    The first occurrence is at index 0, so we return 0.

    Example 2:
    Input: haystack = "leetcode", needle = "leeto"
    Output: -1
    Explanation: "leeto" did not occur in "leetcode", so we return -1.
    

    Constraints:
    1 <= haystack.length, needle.length <= 104
    haystack and needle consist of only lowercase English characters.
 */
class Solution {
private:
    vector<int> constructPrefixDict(string& s) {
        // next[i] represents the maximum length of the common prefix which equals the suffix.
        vector<int> next(s.size(), 0);
        int prefixLength = 0, i = 1;
        while(i < s.size()) {
            while (prefixLength > 0 && s[prefixLength] != s[i]) {
                prefixLength = next[prefixLength - 1];
            }

            if (s[i] == s[prefixLength]) {
                prefixLength++;
            }
            next[i] = prefixLength;
            i++;
        }
        return next;
    }
public:
    int strStr(string haystack, string needle) {
        vector<int> next = constructPrefixDict(needle);
        int j = 0;
        for(int i = 0; i < haystack.size(); ++i) {
            while(j > 0 && haystack[i] != needle[j]) {
                j = next[j - 1];
            }
            if (haystack[i] == needle[j]) {
                j++;
            }
            if (j == needle.size()) {
                return i - needle.size() + 1;
            }
        }
        return -1;
    }
};
