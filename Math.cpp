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


//263. Ugly Number
//https://leetcode.com/problems/ugly-number/
class Solution {
public:
    bool isUgly(int num) {
        if(num <= 0) return false;
        if(num == 1) return true;
        while(num%2 == 0)
            num /= 2;
        while(num%3 == 0)
            num /=3;
        while(num%5 == 0)
            num /= 5;
        
        return num == 1 ? true : false;
    }
};


//264. Ugly Number II
//https://leetcode.com/problems/ugly-number-ii/
/* The general idea is to use the counter to build the list from smallest element 
to the largest */
class Solution {
public:
    int nthUglyNumber(int n) {
        if(n <= 0) return 0;
        if(n == 1) return 1;
        //initialize 3 pointers to be all 0
        int t2 = 0, t3 = 0, t5 = 0;
        //Record the num from 1 ... n
        vector<int> nums(n, 0);        
        nums[0] = 1;
        for(int i = 1; i < n; ++i){
            nums[i] = min(nums[t2]* 2, min(nums[t3]*3, nums[t5]*5));
            if(nums[i] == nums[t2]*2) t2++;
            if(nums[i] == nums[t3]*3) t3++;
            if(nums[i] == nums[t5]*5) t5++;
        }
        return nums[n-1];
    }
};


//258. Add Digits
//https://leetcode.com/problems/add-digits/
//O(n) is trivial, slow
class Solution {
public:
    int addDigits(int num) {
        int digit = 0;
        if(num < 10) return num;
        while(num != 0 || digit >= 10){
            digit += num % 10;
            num /= 10;
            if(num == 0 && digit >= 10){
                num = digit;
                digit = 0;
            }
        }
        return digit;
    }
};

//https://en.wikipedia.org/wiki/Digital_root#Congruence_formula
class Solution {
public:
    int addDigits(int num) {
        return 1 + (num - 1) % 9;
    }
};



