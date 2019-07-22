//479. Largest Palindrome Product
//https://leetcode.com/problems/largest-palindrome-product/
/*
The key insight for this problem is to solve two problems:
1. Arrange the potential palindrome number in descending order;
2. Check whether the palindrome can be the product of two n-digit numbers
We will have two approaches: 1. First build the palindrome number and 
check whether it can be the product of the two n-digit numbers; 2. Start from
two n-digit numbers, and check whether their product can be the palindrome.
A very detailed explanation can be found:
https://leetcode.com/problems/largest-palindrome-product/discuss/96306/Java-solutions-with-two-different-approaches
*/
class Solution {
private:
    long buildPalindrome(int num){
        string s = to_string(num);
        string rev = s;
        reverse(rev.begin(), rev.end());
        s = s + rev;
        return stol(s);
    }
public:
    int largestPalindrome(int n) {
        if(n == 1) return 9;
        //Set the upper and lower boundry of n-digit
        int upperBound = pow(10, n);
        int lowerBound = upperBound/10;
        upperBound -= 1;
        //We always consider the palidrome in descending order
        for(int i = upperBound; i >= lowerBound; --i){
            long pal = buildPalindrome(i);
            //start from the largest number. 
            //Here j*j should be greater or equal to pal. If j*j < pal, then 
            //it will be impossible for us to find the valid n-digit number
            for(long j = upperBound; j >= lowerBound && j*j >= pal; --j){
                if(pal % j == 0){
                    return pal % 1337;
                }
                    
            }
        }
        return 9;
    }
};


