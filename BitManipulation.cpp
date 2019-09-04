#include<windows.h>
#include<algorithm>
#include<vector>
#include<array>
#include<cmath>
#include<random>
#include<sstream>
#include<unordered_map>
#include<numeric>
#include<iterator>

using namespace std;

//136. Single Number
//https://leetcode.com/problems/single-number/
/*
XOR, nothing fancy
*/
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int len = nums.size();
        int res = 0;
        for(int i = 0; i < len; i++){
            res = res^nums[i];
        }
        return res;
    }
};


//389. Find the Difference
//https://leetcode.com/problems/find-the-difference/
/*
A natural approach could be allocate one array as simple hash table. Then iterate through string
s and t, and calculate the number of occurrence in s and t. We can easily check 1 character only
appears once in the table.
If we want to reduce the space complexity to O(1), then we can use XOR operation.
*/
class Solution {
public:
    char findTheDifference(string s, string t) {
        //Assign res to 0 is important here, since x^0 is x
        char res = 0;
        for(char i : s) res = res^i;
        for(char i : t) res = res^i;
        return res;
    }
};

//318. Maximum Product of Word Lengths
//https://leetcode.com/problems/maximum-product-of-word-lengths/
/*
The general idea is to using a flag to indicate each character in the string s,
then we can compare whether there is duplicate characters in string s and t by
bitMask[s] & bitMask[t]. The time complexity should be O(max(n^2, n*k)), k is
the average length of each string s.
*/
class Solution {
int calBits(string& s){
    int res = 0;
    for(char c : s){
        res = res|(1 << (c - 'a'));
        //The maximum value for res is we have 26 '1' in the binary form
        //0x3FFFFFF represents this
        if(res == 0x3FFFFFF) break;
    }
    return res;
}
public:
    int maxProduct(vector<string>& words) {
        const int len = words.size();
        if(len == 0) return 0;
        //bitMask stores which character is in a string w.
        //For each character, we will set a flag to be 1 if it presents in w.
        int bitMask[len], wordLen[len];
        for(int i = 0; i < len; i++){
            bitMask[i] = calBits(words[i]);
            wordLen[i] = words[i].size();
        }
        int res = 0;
        for(int i = 0; i < len; i++){
            for(int j = i + 1; j < len; j++){
                //The precedence of == is above &, we need parathesis here.
                if((bitMask[i] & bitMask[j]) == 0){
                    res = max(res, wordLen[i]*wordLen[j]);
                }    
            }
        }
        return res;
    }
};


//371. Sum of Two Integers
//https://leetcode.com/problems/sum-of-two-integers/
/* A very good question! have a great overview about bit manipulation */
class Solution {
public:
    int getSum(int a, int b) {
        long carry = 0; // 64 bit integer
        // 64 bit mask with first 32 bits to be 0, second 32 bits to be 1
        long mask = 0xFFFFFFFF; 
        int sum = a;
        while(b != 0){
            sum = a ^ b;
            carry = (a & b);
            //We cannot directly use the following code because for 32 bit
            //integers, when the first bit is 1 (negative value), we will 
            //have bit overflow 
            //b = (a & b) << 1;
            b =  ((carry & mask) << 1);
            a = sum;
        }
        return sum;
    }
};

//401. Binary Watch
//https://leetcode.com/problems/binary-watch/
//Intresting problem! The way how to handle the hours and minutes is intresting.
//Please be more comfortable with bitset!
class Solution {
public:
    vector<string> readBinaryWatch(int num) {
        vector<string> res;
        for(int h = 0; h <= 11; ++h){
            for(int m = 0; m <= 59; ++m){
                //move hours 6 bits and make room for minutes
                //exhaustive search
                if(bitset<10>(h << 6 | m).count() == num)
                    res.push_back(to_string(h) + (m < 10 ? ":0" : ":") + to_string(m));
            }
        }
        return res;
    }
};

