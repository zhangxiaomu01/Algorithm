#include<iostream>
#include<algorithm>
#include<vector>
#include<string>
#include<queue>


using namespace std;

// A good collection of DP problems:
// https://leetcode.com/discuss/general-discussion/669996/greedy-for-beginners-problems-sample-solutions


//1326. Minimum Number of Taps to Open to Water a Garden
//https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/
//Greedy solution, you got it wrong when you did the contest. This approach is elegant.
class Solution {
public:
    int minTaps(int n, vector<int>& ranges) {
        int len = ranges.size();
        vector<pair<int, int>> processed;
        auto myComp = [](pair<int, int> p1, pair<int, int> p2){
            if(p1.second == p2.second) return p1.first < p2.first;
            return p1.second > p2.second;
        };
        
        for(int i = 0; i < ranges.size(); ++i){
            int startPos = max(0, i - ranges[i]);
            int endPos = min(i + ranges[i], n);
            processed.push_back(pair<int, int>(startPos, endPos));
        }
        
        sort(processed.begin(), processed.end(), myComp);
        
        int j = 0;
        
        int count = 0;
        //Indicate the final destination we want to achieve
        int dest = n;
        
        for(int i = 0; i <= n; ){
            int curPos = dest;
            while(i <= n && processed[i].second >= dest){
                curPos = min(curPos, processed[i].first);
                ++i;
                if(curPos == 0) return count+1;
            }
            
            if(curPos == dest && i <= n) return -1;
            
            dest = curPos;
            ++count;
              
        }
        
        return count;
    }
};

//DP solution. Note that dp[j] represents the minimum number of taps I should turn on
//when I want to fully water the garden.
//We expand from left to right, and check the minimum number we need  in order to 
//water the previous area and update dp[j] accordingly.
class Solution {
public:
   int minTaps(int n, vector<int>& A) {
        vector<int> dp(n + 1, n + 2);
        dp[0] = 0;
        for (int i = 0; i <= n; ++i)
            for (int j = max(i - A[i] + 1, 0); j <= min(i + A[i], n); ++j)
                dp[j] = min(dp[j], dp[max(0, i - A[i])] + 1);
        return dp[n]  < n + 2 ? dp[n] : -1;
    }
};



//1353. Maximum Number of Events That Can Be Attended
//https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/
//You did not make it right during the contest!
//Greedy solution
//The key insight is that at day d, we need to attend the event which includes day d and 
//ends earliest. We maintain a min heap to keep track of the end days.Once it exceeds the 
//current day d, we remove it from the heap.
//Note we need to sort the event in order to make sure all events happen earlier to come
//first
class Solution {
public:
    int maxEvents(vector<vector<int>>& events) {
        int len = events.size();
        sort(events.begin(), events.end());
        //min heap to keep track of the end day of each event
        priority_queue<int, vector<int>, greater<int>> pq;
        
        int cnt = 0;
        int i = 0; //count events[i]
        //At most, we have 100000 days
        for(int d = 1; d <= 100000; ++d){
            //pop those events who expires on day d
            while(pq.size() && pq.top() < d)
                pq.pop();
            
            while(i < len && events[i][0] == d)
                pq.push(events[i++][1]);
            
            if(pq.size()){
                pq.pop();
                cnt++;
            }
            
        }
        return cnt;
        
    }
};

//Greedy solution
//Optimized version! It's really clever
class Solution {
public:
    int maxEvents(vector<vector<int>>& events) {
        int len = events.size();
        sort(events.begin(), events.end());
        //min heap to keep track of the end day of each event
        priority_queue<int, vector<int>, greater<int>> pq;
        
        int cnt = 0;
        int i = 0; //count events[i]
        int d = 0;
        
        while(pq.size() > 0 || i < len){
            //if we are at the very beginning, or we have a gap
            if(pq.size() == 0)
                d = events[i][0];
            
            while(i < len && events[i][0] <= d)
                pq.push(events[i++][1]);
            
            pq.pop();
            cnt++;
            d++;
            
            while(pq.size() && pq.top() < d)
                pq.pop();
        }
        return cnt;
        
    }
};



//1383. Maximum Performance of a Team
//https://leetcode.com/problems/maximum-performance-of-a-team/
// You almost got it right during the contest!
// Greedy
class Solution {
private:
    const int mV = 1000000007;
    
public:
    int maxPerformance(int n, vector<int>& speed, vector<int>& efficiency, int k) {
        vector<vector<int>> Eng(n, vector<int>(2, 0));
        for(int i = 0; i < n; ++i){
            Eng[i][1] = speed[i];
            Eng[i][0] = efficiency[i];
        }
        
        //min heap
        priority_queue<int, vector<int>, greater<int>> pq;
        
        auto myComp1 = [](vector<int>& v1, vector<int>& v2){
            if(v1[0] == v2[0]) return v1[1] < v2[1];
            else return v1[0] < v2[0];
        };
        
        sort(Eng.begin(), Eng.end(), myComp1);
        
        long long curMax = 0;
        long long res = 0;
        
        for(int i = n-1; i >= 0; --i){
            pq.push(Eng[i][1]);
            curMax += Eng[i][1];
            
            if(i < n-k){
                //If it exceeds k elements, we pop smallest out of pq
                curMax -= pq.top();
                pq.pop();
            }
            res = max(res, curMax * Eng[i][0]);
        }
        return int(res % mV);  
    }
};



// 1405. Longest Happy String
// https://leetcode.com/problems/longest-happy-string/
// This greedy approach is so clever, we just always put two characters together from the longest
// pile, if we cannot do this, then we put 1 such chracter from the longest pile. Then we select 1 or
// 0 character from middle pile. We also need to swap the piles in order to make sure that longest pile
// always comes first, then middle pile, then the last pile.
// Here is the post:
// https://leetcode.com/problems/longest-happy-string/discuss/564277/C%2B%2BJava-a-greater-b-greater-c
// This solution is marvelous! 
class Solution {
public:
    string longestDiverseString(int a, int b, int c, char aa = 'a', char bb = 'b', char cc = 'c') {
        // if all the shorter chracters has been successfully inserted
        // We also make sure that the longest chracter always goes first
        if(b < c) 
            return longestDiverseString(a, c, b, aa, cc, bb);
        else if (a < b)
            return longestDiverseString(b, a, c, bb, aa, cc);
        // base case: string(min(a, 2), aa);
        if(b == 0)
            return string(min(a, 2), aa);
        
        int consumeA = min(a, 2);
        // If we already consumes the A with consumeA times, then the rest is still greater than b, 
        int consumeB = a - consumeA >= b ? 1 : 0; 
        
        string temp = string(consumeA, aa) + string(consumeB, bb);
        return temp + longestDiverseString(a - consumeA, b - consumeB, c, aa, bb, cc);
        
    }
};


// 1402. Reducing Dishes
// https://leetcode.com/problems/reducing-dishes/
// Greedy problem! Can be solved using dp + memorization
// Two key observations here are we always need to make the most satisfied dishes
// to the end; the second one is that 
// total accumulate level (from largest to smallest) >= satisfaction[i], then we 
// can safely include satisfaction[i] to our final result
class Solution {
public:
    int maxSatisfaction(vector<int>& satisfaction) {
        int len = satisfaction.size();
        sort(satisfaction.begin(), satisfaction.end());
        int accumulate = 0;
        int res = 0;
        for(int i = len-1; i >= 0 && satisfaction[i] > -accumulate; --i){
            accumulate += satisfaction[i];
            res += accumulate;
        }
        
        return res;
    }
};


