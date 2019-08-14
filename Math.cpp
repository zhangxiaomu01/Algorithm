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


//367. Valid Perfect Square
//https://leetcode.com/problems/valid-perfect-square/
/* O(n) is trivial */
class Solution {
public:
    bool isPerfectSquare(int num) {
        if(num == 1) return true;
        for(long i = 1; i <= num/2; ++i){
            if(i * i == num)
                return true;
        }
        return false;
    }
};

/* Binary search is powerful */
class Solution {
public:
    bool isPerfectSquare(int num) {
        long l = 1, r = num;
        while(l < r){
            long mid = l + (r - l) / 2;
            long res = mid * mid;
            if(res == num) return true;
            else if(res > num) r = mid;
            else l = mid+1;
        }
        return (l*l == num);
    }
};

//Newton's law: https://en.wikipedia.org/wiki/Newton%27s_method
class Solution {
public:
    bool isPerfectSquare(int num) {
        if(num <= 0) return false;
        if(num == 1) return true;
        long t = num / 2;
        while(t * t > num){
            t = (t + num / t)/ 2;
        }
        return t * t == num;
    }
};


//365. Water and Jug Problem
//https://leetcode.com/problems/water-and-jug-problem/
/* Very tricky solution. The intuition is that we add y
liters water to current volume if current volume is samller
than x. Or we remove x volume of waters from it.*/
class Solution {
public:
    bool canMeasureWater(int x, int y, int z) {
        if(x+y == z || x == z || y == z) return true;
        if(z > x + y) return false;
        //Let's always consider x to be the smaller jug
        if(x > y)
            swap(x, y);
        
        int curVolume = 0; //current water we have
        while(1){
            //since current water is smaller than x, then
            //we can safely add y liters water to it
            if(curVolume < x)
                curVolume += y;
            //we can safely remove x liters water from it
            else
                curVolume -= x;
            if(curVolume == z) return true;
            if(curVolume == 0) return false;
        }
        return false;
    }
};


//326. Power of Three
//https://leetcode.com/problems/power-of-three/
/* Loop version is trivial */
class Solution {
public:
    bool isPowerOfThree(int n) {
        if(n == 1) return true;
        long sum = 1;
        for(int i = 1; i <= n/3; ++i){
            sum *= 3;
            if(sum > n) return false;
            if(sum == n) return true;
        }
        return false;
    }
};

/* This solution utilize the upper bound of the integer.
In general, 3^19 is the maximum number which bounded by INT_MAX.
Then we can check whenther n is divisible by 3^19 */
class Solution {
public:
    bool isPowerOfThree(int n) {
        int x = 1162261467; // 3^19
        return n > 0 && x % n == 0;
    }
};

//313. Super Ugly Number
//https://leetcode.com/problems/super-ugly-number/
/*Every ugly number is constructed from multiply a previous ugly number by one of the 
primes in the list. If current ugly number is ugly[i] , Index[j] is the index of the 
smallest of all ugly numbers that we already constructed , such that 
prime[j]*ugly[index[j]] is not found yet.
Very tricky Problem!!*/
class Solution {
public:
    int nthSuperUglyNumber(int n, vector<int>& primes) {
        int k = primes.size();
        //ugly records all the potential ugly numbers
        //index record the index for the minimum previous ugly number which can
        //be used to construct ugly[j]
        vector<int> index(k, 0), ugly(n, INT_MAX);
        ugly[0] = 1;
        for(int i = 1; i < n; ++i){
            for(int j = 0; j < k; ++j){
                ugly[i] = min(ugly[i], ugly[index[j]] * primes[j]);
            }
            for(int j = 0; j < k; ++j){
                index[j] += (ugly[i] == ugly[index[j]] * primes[j]);
            }
        }
        return ugly[n-1];
    }
};


