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
    1 <= s.length <= 105
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
