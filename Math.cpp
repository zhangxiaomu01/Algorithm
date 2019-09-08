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



//400. Nth Digit
//https://leetcode.com/problems/nth-digit/
/* Not that hard, however, tricky to implement it */
class Solution {
private:
    //Use long to handle integer overflow
    long dict[10] = {1, 9, 90, 900, 9000, 90000, 900000, 9000000, 90000000, 900000000};
    long dictNum[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
public:
    int findNthDigit(int n) {
        if(n < 10) return n;
        long count = n;
        int i = 1;
        while(count > 0){
            if(count >= dict[i] * dictNum[i]){
                count -= dict[i] * dictNum[i];
                i++;
            }
            else
                break;
        }
        int nDigits = dictNum[i-1];
        int startNum = 1;
        while(nDigits > 0){
            startNum *= 10;
            nDigits--;
        }
        count --;
        //cout << i << endl;
        int increNum = count / dictNum[i];
        int index = count % dictNum[i];
        int res = 0;
        startNum += increNum;
        //cout << startNum << endl;
        string temp = to_string(startNum);
        res = temp[index] - '0';
        
        return res;
    }
};


//390. Elimination Game
//https://leetcode.com/problems/elimination-game/
//A very good explanation!!
//https://leetcode.com/problems/elimination-game/discuss/87119/JAVA%3A-Easiest-solution-O(logN)-with-explanation
/*
My idea is to update and record head in each turn. when the total number becomes 1, head is the only number left.

When will head be updated?

if we move from left
if we move from right and the total remaining number % 2 == 1
like 2 4 6 8 10, we move from 10, we will take out 10, 6 and 2, head is deleted and move to 4
like 2 4 6 8 10 12, we move from 12, we will take out 12, 8, 4, head is still remaining 2
then we find a rule to update our head.

example:
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

Let us start with head = 1, left = true, step = 1 (times 2 each turn), remaining = n(24)

we first move from left, we definitely need to move head to next position. (head = head + step)
So after first loop we will have:
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 - > 2 4 6 8 10 12 14 16 18 20 22 24
head = 2, left = false, step = 1 * 2 = 2, remaining = remaining / 2 = 12

second loop, we move from right, in what situation we need to move head?
only if the remaining % 2 == 1, in this case we have 12 % 2 == 0, we don't touch head.
so after this second loop we will have:
2 4 6 8 10 12 14 16 18 20 22 24 - > 2 6 10 14 18 22
head = 2, left = true, step = 2 * 2 = 4, remaining = remaining / 2 = 6

third loop, we move from left, move head to next position
after third loop we will have:
2 6 10 14 18 22 - > 6 14 22
head = 6, left = false, step = 4 * 2 = 8, remaining = remaining / 2 = 3

fourth loop, we move from right, NOTICE HERE:
we have remaining(3) % 2 == 1, so we know we need to move head to next position
after this loop, we will have
6 14 22 - > 14
head = 14, left = true, step = 8 * 2 = 16, remaining = remaining / 2 = 1

while loop end, return head
*/
class Solution {
public:
    int lastRemaining(int n) {
        if(n == 1) return 1;
        int remaining = n;
        //Record wheather we traverse from left or from right
        bool fromLeft = true;
        int step = 1;
        int res = 1;
        while(remaining > 1){
            if(fromLeft || remaining % 2 == 1)
                res += step;
            remaining /= 2;
            //step should multiply by 2 since we take off one element
            //for each round, and cause the gap larger and larger!
            step *= 2;
            fromLeft = !fromLeft;
        }
        
        return res;
    }
};


//441. Arranging Coins
//https://leetcode.com/problems/arranging-coins/
/*
1+2+3+...+x = n
-> (1+x)x/2 = n
-> x^2+x = 2n
-> x^2+x+1/4 = 2n +1/4
-> (x+1/2)^2 = 2n +1/4
-> (x+0.5) = sqrt(2n+0.25)
-> x = -0.5 + sqrt(2n+0.25)
*/
class Solution {
public:
    int arrangeCoins(int n) {
        return floor(-0.5 + sqrt(2 * static_cast<double> (n) + 0.25));
    }
};


//843. Guess the Word
//https://leetcode.com/problems/guess-the-word/
//A nice explanation from:
//https://leetcode.com/problems/guess-the-word/discuss/133862/Random-Guess-and-Minimax-Guess-with-Comparison
class Solution {
private:
    int match(string& s1, string& s2){
        int res = 0;
        for(int i = 0; i < s1.size(); ++i){
            res += s1[i] == s2[i] ? 1 : 0;
        }
        return res;
    }
public:
    void findSecretWord(vector<string>& wordlist, Master& master) {
        //if x == 6, we guess the correct word
        for(int j = 0, x = 0; j < 10 && x < 6; ++j){
            string pick = wordlist[rand() % wordlist.size()];
            x = master.guess(pick);
            //it's like a filter, each time we filter the matched word
            //to a new list. Then keep filtering until we find the matched 
            //word or we run out of times
            vector<string> wordlist2;
            for(int i = 0; i < wordlist.size(); ++i){
                if(x == match(wordlist[i], pick))
                    wordlist2.push_back(wordlist[i]);
            }
            wordlist.swap(wordlist2);      
        }
    }
};


//For the random solution, since we arbitrarily select an word
//it could potentially get a word with 0 matches. If we want to 
//improve the result, we need to minimize the possibilities that
//we select such word. We can do a preprocessing, and try to find
//a word with minimum 0 matched words. Then start with this word.
//The code is slower, but has higher chance to get it right!
class Solution {
private:
    int match(string s1, string s2){
        int res = 0;
        for(int i = 0; i < s1.size(); ++i)
            res += (s1[i] == s2[i]) ? 1 : 0;
        return res;
    }
public:
    void findSecretWord(vector<string>& wordlist, Master& master) {
        for(int i = 0, x = 0; i < 10 && x < 6; ++i){
            unordered_map<string, int> uMap;
            for(string& w1 : wordlist)
                for(string& w2 : wordlist)
                    if(match(w1, w2) == 0) 
                        uMap[w1]++;
            
            string pick = wordlist[0];
            int count = INT_MAX;
            /*
            //should not check uMap, because it could potentially be
            //empty
            for(auto it = uMap.begin(); it != uMap.end(); ++it){
                if(it->second < x){
                    pick = it->first;
                    x = it->second;
                }
            }*/
            //The purpose for this loop is just find pick, count is
            //an auxiliary varibale
            for(string& w : wordlist){
                if(uMap[w] < count){
                    pick = w; 
                    count = uMap[w];
                } 
            }
            //check guess here!
            x = master.guess(pick);
            vector<string> wordlist2;
            for(string& w : wordlist){
                if(match(w, pick) == x)
                    wordlist2.push_back(w);
            }
            wordlist.swap(wordlist2);
        }
    }
};



