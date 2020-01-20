#include<iostream>
#include<algorithm>
#include<vector>
#include<string>

using namespace std;


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