#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<unordered_set>
#include<unordered_map>
#include<queue>
#include<stack>
using namespace std;

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


//50. Pow(x, n)
//https://leetcode.com/problems/powx-n/
//Note how we handle n, n could be potentially large, and we need to divide
//it by 2, instead of -1.
class Solution {
private:
    double helper(double x, long long n){
        if(n == 0) return 1.000;
        if(n == 1) return x;
        //Note we need to divide by 2, or we will have 
        //stack overflow
        double res = helper(x, n/2);
        res *= res;
        if(n % 2 == 1)
            res = x * res;
        return res;
    }
public:
    double myPow(double x, int n) {
        //if(x == 0.000000000 && n < 0) 
        if(x == 1.000000) return x;
        long long nl = abs(static_cast<long long>(n));
        if(n > 0) return helper(x, nl);
        else return 1.0 / helper(x, nl);
            
    }
};

//Iterative version
class Solution {
public:
    double myPow(double x, int n) {
        double ans = 1;
        long nl = abs(static_cast<long>(n));
        while(nl > 0){
            if((nl&1) == 1) ans = ans * x;
            nl >>= 1;
            x *= x;
        }
        return n<0 ? 1.0/ans : ans;
    }

};


//779. K-th Symbol in Grammar
//https://leetcode.com/problems/k-th-symbol-in-grammar/
//An interesting idea
//https://leetcode.com/problems/k-th-symbol-in-grammar/discuss/121544/
/*
if K % 2 == 1, it is the first number in '01' or '10',
if Kth number is 0, K+1 th is 1.
if Kth number is 1, K+1 th is 0.
so it will be different from K + 1.

If K % 2 == 0, it is the second number in '01' or '10', generated from K/2 th number.
If Kth number is 0, it is generated from 1.
If Kth number is 1, it is generated from 0.
*/
//I cannot get it in the interview. Not intuitive solution!
class Solution {
public:
    int kthGrammar(int N, int K) {
        int res = 0;
        //Track back from beginning to end
        while(K > 1){
            K = (K % 2 == 1) ? K+1 : K / 2;
            res ^= 1;
        }
        return res;
    }
};

//Actually, we can consider the seqeunce as binary search tree. 
//If current node is 1, then its two sub tree is L:1, R: 0
//If current node is 0, then its two sub tree is L:0, R: 1
//If K is even, then its parent should be K/2. And we know if its parent
//is 1, it should be 0. If the parent is 0, then it should be 1.
//If K is odd, then its parent should be (K+1)/2. It should be the same as
//its parent!
//Please draw a binary tree graph to see this!
//It's not that easy to get this problems
class Solution {
public:
    int kthGrammar(int N, int K) {
        if(N == 1) return 0;
        //Odd should be the same as its parent
        if(K % 2 == 1) return (kthGrammar(N-1, (K+1)/2) == 0) ? 0 : 1;
        else return (kthGrammar(N-1, K/2) == 0) ? 1 : 0;
    }
};



//1041. Robot Bounded In Circle
//https://leetcode.com/problems/robot-bounded-in-circle/
//The critical observation is that after one iteration, if the robot goes
//back to origin or does not face towards north, then it can form a cycle.
class Solution {
public:
    bool isRobotBounded(string instructions) {
        //define the four directions.
        const vector<vector<int>> dir = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        //indicate the currect direction is dir[0] == {0, 1}
        int curDir = 0;
        int pos[2] = {0, 0};
        for(int i = 0; i < instructions.size(); ++i){
            if(instructions[i] == 'R')
                curDir = (curDir + 1) % 4;
            else if(instructions[i] == 'L')
                curDir = (curDir + 3) % 4;
            else{
                pos[0] += dir[curDir][0];
                pos[1] += dir[curDir][1];
            }
                
        }
        //If the robot return to the origin and it does not face to the north
        return (pos[0] == 0 && pos[1] == 0) || curDir != 0;
    }
};


//789. Escape The Ghosts
//https://leetcode.com/problems/escape-the-ghosts/
//A tricky problem which is not easy to get the intuition.
/*
Your distance to target is abs(t[0]) + abs(t[1]).
For every ghost g, distance to target is abs(t[0] - g[0]) + abs(t[1] - g[1]).
You need to be closer to target than any ghost to escape.

A short proof why this works:
1. If the ghost's distance to target is shorter, it can move to target and 
stay there.
2. If the ghost's distance to target is longer, the ghost should never reach 
the player.
Let's say the player simply moves towards the target, and the ghost also move 
towards the target (obviously the ghost can never reach the player if it moves 
away from target), both the ghosts and player's distance to target should 
reduce 1 per every step, and if the ghost reaches the player, means their 
distances to the target become equal, which is not possible in this case.

*/
class Solution {
public:
    bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target) {
        int playerDist = abs(target[0]) + abs(target[1]);
        for(auto& g : ghosts){
            int ghostDist = abs(g[0] - target[0]) + abs(g[1] - target[1]);
            if(ghostDist <= playerDist) return false;
        }
        return true;
    }
};



//1354. Construct Target Array With Multiple Sums
//https://leetcode.com/problems/construct-target-array-with-multiple-sums/
//A very hard problem, tricky and not easy to deal with!
//This is essentially a math problem! It's not easy to identify the rules
//Let's say the total sum of array is sum, and the max element is eMax,
//Then eMax must be the value from last iteration, and the value before
//we replace eMax to is eMax - (sum - eMax). Note sum - eMax is the previous
//sum of all the other elements aside from the element located at eMax. 
//In order to improve the overall efficiency, we need to replace 
//eMax - (sum - eMax) to eMax % (sum - eMax). A simple proof is:
/*
Let's say [mx, a1,a2,..,an]
other=a1+a2+...+an
sm=other+mx
if mx-other>other, there must be multiple(mx//other) times operations, and previous value of mx is mx%other
for instance, [10,3], mx=10, other=3, prev=1(10%3)
[1,3] => [4,3] => [7,3] => [10,3]
you keep adding other to the element whose index is mx's, 3(10//3) times, and then 1 becomes 10
*/
//Since the value will be large, we promote int to long to solve this issue.
//A good explanation:
//https://leetcode.com/problems/construct-target-array-with-multiple-sums/discuss/510214/JavaC%2B%2B-O(n)-Solution-(Reaching-Points)
class Solution {
public:
    bool isPossible(vector<int>& target) {
        long sum = 0;
        long curMax = 0;
        //We maintain a max heap to get the largest element efficiently
        priority_queue<long> pq;
        for(int e : target){
            pq.push(e);
            sum += e;
        }
        
        while(pq.top() != 1 && sum != 1){
            curMax = pq.top();
            pq.pop();
            
            //[n, 1] => [n-1, 1] => ... => [2, 1] => [1, 1]
            if(sum - curMax == 1) return true;
            
            if(curMax <= sum / 2) return false;
            
            if(sum - curMax == 0) return false;
            
            long preMax = curMax % (sum - curMax);
            pq.push(preMax);
            //curMax - preMax >= 0
            sum = sum - (curMax - preMax);
            
        }
        
        return sum == 1 || sum == target.size();
    }
};


//A much more intuitive way to solve it. However, much slower because we get rid of the 
//module trick!
//A much more intuitive way to solve it. However, much slower because we get rid of the 
//module trick!
class Solution {
public:
    bool isPossible(vector<int>& target) {
        long sum = 0;
        long curMax = 0;
        //We maintain a max heap to get the largest element efficiently
        priority_queue<long> pq;
        for(int e : target){
            pq.push(e);
            sum += e;
        }
        
        while(pq.top() != 1 && sum != 1){
            curMax = pq.top();
            pq.pop();
            
            //[n, 1] => [n-1, 1] => ... => [2, 1] => [1, 1]
            if(sum - curMax == 1) return true;
            
            if(curMax <= sum / 2 || sum - curMax == 0) return false;
            //Much slower here, and cannot pass OJ
            long preMax = curMax - (sum - curMax);
            pq.push(preMax);
            //curMax - preMax >= 0
            sum = sum - (curMax - preMax);
            
        }
        
        return sum == 1 || sum == target.size();
    }
};


//1359. Count All Valid Pickup and Delivery Options
//1359. Count All Valid Pickup and Delivery Options
//A pure math problem. A great explanation from lee
//https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options/discuss/516968/JavaC%2B%2BPython-Easy-and-Concise
//O(n), space O(1)
class Solution {
public:
    int countOrders(int n) {
        long res = 1;
        int mod = 1e9 + 7;
        
        for(int i = 1; i <= n; ++i){
            //Note when we insert i th pair, we will have 
            //(2 * i - 1) * 2 * i / 2 choices! For the rest of 
            //i - 1 pairs, we already have res number of permutations.
            res = res * (2 * i - 1) * i % mod;
        }
        
        return int(res);
    }
};


//1360. Number of Days Between Two Dates
//https://leetcode.com/problems/number-of-days-between-two-dates/
//A tricky problem! Note how we check the leap year!
class Solution {
public:
    int daysBetweenDates(string date1, string date2) {
        if(date1 == date2) return 0;
        
        int dict[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        int y1 = 0, y2 = 0, m1 = 0, m2 = 0, d1 = 0, d2 = 0;
        y1 = stoi(date1.substr(0, 4));
        m1 = stoi(date1.substr(5, 2));
        d1 = stoi(date1.substr(8, 2));
        
        y2 = stoi(date2.substr(0, 4));
        m2 = stoi(date2.substr(5, 2));
        d2 = stoi(date2.substr(8, 2));
        
        //cout << y1 << " " << m1 << " " << d1 << endl;
        
        vector<int> fD = {y1, m1, d1}, sD = {y2, m2, d2};
        
        //Make sure smaller dates goes before large date
        if(fD[0] > sD[0])
            swap(fD, sD);
        else if(fD[0] == sD[0] && fD[1] > sD[1])
            swap(fD, sD);
        else if(fD[0] == sD[0] && fD[1] == sD[1] && fD[2] > sD[2])
            swap(fD, sD);
        
        int res = 0;
        
        //Calculate the number of days from the epoch. From the first day of the smallest year to
        //current date of both years. Then we substract the larger with the smaller.
        int nDay1 = 0, nDay2 = 0;
        int diffMonth = fD[1] - 1; 
        for(int i = 0; i < diffMonth; ++i){
            nDay1 += dict[i];
        }
        //Handles the leap year
        if((((fD[0] % 4 == 0) && (fD[0] % 100 != 0)) || (fD[0] % 400 == 0)) && fD[1] > 2) nDay1 += 1;
        nDay1 += fD[2];
        //cout << nDay1 << endl;

        for(int i = fD[0]; i < sD[0]; ++i){
            if(i % 4 == 0) nDay2 += 366;
            else
                nDay2 += 365;
        }
        for(int i = 0; i < sD[1] - 1; ++i){
            nDay2 += dict[i];
        }
        //Handles the leap year
        if((((sD[0] % 4 == 0) && (sD[0] % 100 != 0)) || (sD[0] % 400 == 0)) && sD[1] > 2) nDay2 += 1;
        nDay2 += sD[2];
        
        res = nDay2 - nDay1;
        return res;
    }
};


//1363. Largest Multiple of Three
//https://leetcode.com/problems/largest-multiple-of-three/
//This is my trial, unfortunately, it does not work! It's pretty close to the final answer
/*
class Solution {
public:
    string largestMultipleOfThree(vector<int>& digits) {
        int dict[10] = {0};
        for(int c : digits){
            dict[c] ++;
        }
        
        string res = "";
        
        for(int i = 9; i >= 0; --i){
            while (i == 9 && dict[i] > 0) {
                res.push_back('9');
                dict[i] --;
            }
            
            while(i == 8 && dict[i] > 0){
                if(dict[7] > 0){
                    res.push_back('8');
                    res.push_back('7');
                    dict[i] --;
                    dict[7] --;
                }else if(dict[4] > 0){
                    res.push_back('8');
                    res.push_back('4');
                    dict[i] --;
                    dict[4] --;
                }else if(dict[1] > 0){
                    res.push_back('8');
                    res.push_back('1');
                    dict[i] --;
                    dict[1] --;
                }else
                    break;
            }
            
            while(i == 7 && dict[i] > 0){
                if(dict[5] > 0){
                    res.push_back('7');
                    res.push_back('5');
                    dict[i] --;
                    dict[5] --;
                }else if(dict[3] > 0){
                    res.push_back('7');
                    res.push_back('2');
                    dict[i] --;
                    dict[2] --;
                }else
                    break;
            }
            
            while(i == 6 && dict[i] > 0){
                res.push_back('6');
                dict[i] --;
            }
            
            while(i == 5 && dict[i] > 0){
                if(dict[4] > 0){
                    res.push_back('5');
                    res.push_back('4');
                    dict[i] --;
                    dict[4] --;
                }else if(dict[1] > 0){
                    res.push_back('5');
                    res.push_back('1');
                    dict[i] --;
                    dict[1] --;
                }else
                    break;
            }
            
            while(i == 4 && dict[i] > 0){
                if(dict[2] > 0){
                    res.push_back('4');
                    res.push_back('2');
                    dict[i] --;
                    dict[2] --;
                }else
                    break;
            }
            
            while(i == 3 && dict[i] > 0){
                res.push_back('3');
                dict[i] --;
            }
            
            while(i == 2 && dict[i] > 0){
                if(dict[1] > 0){
                    res.push_back('2');
                    res.push_back('1');
                    dict[i] --;
                    dict[1] --;
                }else
                    break;
            }
            
            while(i == 1 && dict[i] >= 3){
                res.push_back('1');
                res.push_back('1');
                res.push_back('1');
                dict[i] -= 3;
            }
            
            while(i == 0 && dict[i] > 0){
                res.push_back('0');
                dict[i] --;
            }
            
        }
        
        int k = 0;
        while(k < res.size() - 1 && res[k] == '0'){
            k++;
        }
        //cout << k << endl;
        res =res.substr(k, res.size() - k);
        return res;
    }
};

*/

//Actually, this is a pure math problem!
/*
Basic Math
999....999 % 3 == 0
1000....000 % 3 == 1
a000....000 % 3 == a % 3
abcdefghijk % 3 == (a+b+c+..+i+j+k) % 3


Explanation
Calculate the sum of digits total = sum(A)
If total % 3 == 0, we got it directly
If total % 3 == 1 and we have one of 1,4,7 in A:
we try to remove one digit of 1,4,7
If total % 3 == 2 and we have one of 2,5,8 in A:
we try to remove one digit of 2,5,8
If total % 3 == 2:
we try to remove two digits of 1,4,7
If total % 3 == 1:
we try to remove two digits of 2,5,8
Submit

Complexity
We can apply counting sort, so it will be O(n)
Space O(sort)

This excellent solution is from lee:
https://leetcode.com/problems/largest-multiple-of-three/discuss/517628/Python-Basic-Math
*/
class Solution {
    string toStr(vector<int>& A){
        string res;
        for(int i = 9; i >= 0; --i){
            while(A[i] > 0){
                res.push_back(i + '0');
                A[i] --;
            }
        }
        return res;
    }
    
    bool getRightAns(vector<int>& A, int t, int& Sum){
        if(A[t] > 0) {
            A[t] --;
            Sum -= t;
        }
        if(Sum % 3 == 0) return true;
        return false;
    }
    
public:
    string largestMultipleOfThree(vector<int>& digits) {
        int sum = 0;
        vector<int> Arry(10, 0);
        for(int e : digits){
            Arry[e] ++;
            sum += e;
        }
        
        string res = "";
        
        if(sum % 3 == 0)
            res = toStr(Arry);
        else if(sum % 3 == 1 && (getRightAns(Arry, 1, sum) || getRightAns(Arry, 4, sum) || getRightAns(Arry, 7, sum))){
            res = toStr(Arry);
        }
        else if(sum% 3 == 2 && (getRightAns(Arry, 2, sum) || getRightAns(Arry, 5, sum) || getRightAns(Arry, 8, sum))){
            res = toStr(Arry);
        }
        else if(sum% 3 == 2 && (getRightAns(Arry, 1, sum) || getRightAns(Arry, 1, sum)|| getRightAns(Arry, 4, sum) || getRightAns(Arry, 4, sum) || getRightAns(Arry, 7, sum) || getRightAns(Arry, 7, sum)))
            res = toStr(Arry);
        else if(sum% 3 == 1 && (getRightAns(Arry, 2, sum) || getRightAns(Arry, 2, sum)|| getRightAns(Arry, 5, sum) || getRightAns(Arry, 5, sum) || getRightAns(Arry, 8, sum) || getRightAns(Arry, 8, sum)))
            res = toStr(Arry);
        
        if(res == "") return "";
        
        return res[0] == '0' ? "0" : res;
    }
};


// 1401. Circle and Rectangle Overlapping
// https://leetcode.com/problems/circle-and-rectangle-overlapping/
// A very clever idea from 
//https://leetcode.com/problems/circle-and-rectangle-overlapping/discuss/563463/C%2B%2B-with-simple-explanation
//Just move the center of the circle to the Origin (0, 0), so the problem becomes:
//"is there a point (x, y) (x1 <= x <= x2, y1 <= y <= y2) satisfying x^2 + y^2 <= r^2".
//Then we can compute the radius^2, and the minimum distance from the each verties. Since a rectangle is convex and 
//each vertex is an extreme point.
class Solution {
public:
    bool checkOverlap(int radius, int x_center, int y_center, int x1, int y1, int x2, int y2) {
        // Recenter the circle to (0, 0)
        x1 -= x_center;
        y1 -= y_center;
        x2 -= x_center;
        y2 -= y_center;
        
        //if the triangle lies across axis, we know for this axis, the triangle will be intersect with circle
        //Image it as some sort of 1-d projection
        int minX = x1 * x2 > 0 ? min(x1 * x1, x2 * x2) : 0;
        int minY = y1 * y2 > 0 ? min(y1 * y1, y2 * y2) : 0;
        
        return minX + minY <= radius * radius;
    }
};


// 1453. Maximum Number of Darts Inside of a Circular Dartboard
// https://leetcode.com/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/
// Interesting idea:
// https://leetcode.com/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/discuss/636372/JavaC%2B%2BPython-POJ-1981
// Not a good interview problem!
class Solution {
public:
    int numPoints(vector<vector<int>>& points, int r) {
        int len = points.size();
        // We can at least cover 1 point since points is not null
        int res = 1;
        for(int i = 0; i < len; ++i){
            for(int j = i + 1; j < len; ++j){
                int x1 = points[i][0], y1 = points[i][1];
                int x2 = points[j][0], y2 = points[j][1];
                double dist = sqrt((x2 - x1) * (x2 - x1) + (y2- y1) * (y2 - y1));
                // Impossible to form a circle with radius r!
                if(dist > r*r) continue;
                // Typically we need to consider circle center from both left and right side of two points
                // However if we iterate all two pairs, and always make the circle stay on the same side
                // We will capture all the possible solutions.
                // Calculate the circle center from which goes through the two points
                // This equation is a little bit tricky! Draw a picture!
                double x0 = (x2 + x1) / 2.0 + (y2 - y1) * sqrt(r * r - dist * dist / 4.0) / dist;
                double y0 = (y2 + y1) / 2.0 - (x2 - x1) * sqrt(r * r - dist * dist / 4.0) / dist;
                int cnt = 0;
                for(auto& p : points){
                    int x = p[0], y = p[1];
                    if((x - x0) * (x - x0) + (y - y0) * (y - y0) <= r * r + 0.0000001){
                        cnt ++;
                    }
                }
                res = max(res, cnt);
            }
        }
        
        return res;
    }
};


// 1467. Probability of a Two Boxes Having The Same Number of Distinct Balls
// https://leetcode.com/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/
/*
Math problem! This great solution is from:
https://leetcode.com/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/discuss/662154/C%2B%2B-24ms%3A-Permutations-of-Combination
DFS approach. It's very hard to get it work! It's so brilliant!
*/
class Solution {
private:
    //since 1 <= ball[i] <= 6
    const int fact[7] = {1, 1, 2, 6, 24, 120, 720};
    double factorial(int n){
        return n < 3 ? n : n * factorial(n-1);
    }
    double match = 0, all = 0;
    // p indicates color
    void dfs(vector<int>& balls, int p, int col1, int col2, int cnt1, int cnt2, double prm1, double prm2){
        // both boxes have n balls
        if(cnt1 == 0 && cnt2 == 0){
            all += prm1 * prm2;
            match += (col1 == col2) ? prm1 * prm2 : 0;
        }else if(cnt1 >= 0 && cnt2 >= 0){ // Note here we must use >= because it's possible to have one box filled with n / 2 balls first
            // Here we do not need to worry about p >= balls.size(), because if that happens, cnt1 and cnt2
            // must be <=0
            for(int b1 = 0, b2 = balls[p]; b2 >= 0; --b2, ++b1)
                // if we add b number of balls of same color, the new prm should be prmOri / fact[b]
                dfs(balls, p+1, col1 + (b1 > 0), col2 + (b2 > 0), cnt1 - b1, 
                    cnt2 - b2, prm1 / fact[b1], prm2 / fact[b2]);
        }
    }
    
    
public:
    double getProbability(vector<int>& balls) {
        int total = accumulate(balls.begin(), balls.end(), 0);
        double totalPermutationForEachBag = factorial(total / 2);
        dfs(balls, 0, 0, 0, total / 2, total / 2, totalPermutationForEachBag, totalPermutationForEachBag);
        return match / all;
    }
};


// 1515. Best Position for a Service Centre
// https://leetcode.com/problems/best-position-for-a-service-centre/
// A great solution from:
// https://leetcode.com/problems/best-position-for-a-service-centre/discuss/731606/C%2B%2B-Simulated-Annealing
// Not very efficient though. The idea is interesting and worth investing sometime to get it.
class Solution {
private:
    double calNorm(vector<int>& p1, vector<double>& p2){
        return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]));
    }
    
    double sumDist(vector<vector<int>>& pos, vector<double> C){
        double res = 0.0;
        for(int i = 0; i < pos.size(); ++i){
            res += calNorm(pos[i], C);
        }
        return res;
    }
public:
    double getMinDistSum(vector<vector<int>>& pos) {
        int len = pos.size();
        const int dir[5] = {-1, 0, 1, 0, -1};
        
        // We start from the origin
        vector<double> C = {0, 0};
        //pos[i][0] && pos[i][1] is between [0, 100]
        double step = 100.0;
        double res = sumDist(pos, C);
        // double preRes = 0.0;
        double gap = 1e-6;
        // At first, I tried abs(res - preRes) here, but I will get TLE error
        // The issue is because we may not update the preRes timely, so when we
        // happen get a curSum pretty close to res, then we might end up with never 
        // getting achance to go to the if(curSum < res) statement and cause an infinite
        // loop. We only need to make sure our step is smaller than gap. So we know each
        // time, our points update will be sufficiently smaller to capture the final res
        // while(abs(res - preRes) > gap){
        while(step > gap){
            bool isSmaller = false;
            for(int i = 0; i < 4; ++i){
                vector<double> newC = {C[0] + dir[i] * step, C[1] + dir[i+1] * step};
                double curSum = sumDist(pos, newC);
                // Once we find a smaller sum, we can update our 
                if(curSum < res){
                    // preRes = res; 
                    res = curSum;
                    isSmaller = true;
                    C= newC;
                    break;
                }
            }
            // This is the tricky part, if we cannot find a smaller sum during for all 
            // 4 directions, we need to shrink step
            if(!isSmaller) step /= 2.0;
        }
        return res;
    }
};


// 1819. Number of Different Subsequences GCDs
// https://leetcode.com/problems/number-of-different-subsequences-gcds/
// An interesting problem with an excellent solution.
/*
The general idea is that we need to iterate from 1 to maximum possible numbers 200000, and 
figure out whether this number can be a common GCD of the subsequence, whenever we find one,
we include it to the final count.
*/
class Solution {
public:
    int countDifferentSubsequenceGCDs(vector<int>& nums) {
        bool set[200001] = {false};
        int max_value = 0;
        for (int i = 0; i < nums.size(); ++i) {
            set[nums[i]] = true;
            max_value = max(max_value, nums[i]);
        }
        int cnt = 0;
        // i is the possible divisor, we start from the smallest 1, once it's included, we 
        // increase it by 1 and keep iterating.
        for (int i = 1; i <= 200000; ++i) {
            int commonDivisor = 0;
            // if x is from nums, then i must be a common divisor of some number in nums.
            for (int x = i; x <= max_value && commonDivisor != i; x += i) {
                if (set[x]) commonDivisor = gcd(commonDivisor, x);
            }
            // If we break the earlier loop with commonDivisor == i, then it's the GCD of
            // some subsequence.
            if (commonDivisor == i) cnt ++;
        }
        return cnt;
    }
};


// 1835. Find XOR Sum of All Pairs Bitwise AND
// https://leetcode.com/problems/find-xor-sum-of-all-pairs-bitwise-and/
/*
We all know the distributive property that (a1+a2)*(b1+b2) = a1*b1 + a1*b2 + a2*b1 + a2*b2

Now focus on each bit,
for example the last bit of A[i] and B[j],
and think how it works and affect the result.


Explanation
Distributive property is similar for AND and XOR here.
(a1^a2) & (b1^b2) = (a1&b1) ^ (a1&b2) ^ (a2&b1) ^ (a2&b2)
(I wasn't aware of this at first either)


Complexity
Time O(A + B)
Space O(1)

I think if I spent a little bit more time on this one, I can get it.
*/

class Solution {
public:
    int getXORSum(vector<int>& arr1, vector<int>& arr2) {
        int XorA = 0, XorB = 0;
        for (int a: arr1) XorA ^= a;
        for (int b: arr2) XorB ^= b;
        return XorA & XorB;
    }
};

