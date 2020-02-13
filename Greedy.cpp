#include<iostream>
#include<algorithm>
#include<vector>
#include<string>
#include<queue>


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
