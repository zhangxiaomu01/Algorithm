#include<windows.h>
#include<iostream>
#include<algorithm>
#include<vector>
#include<array>
#include<cmath>
#include<random>
#include<sstream>
#include<unordered_map>
#include<numeric>
#include<iterator>
#include<unordered_set>
#include<queue>
#include<set>
#include<map>

using namespace std;

//374. Guess Number Higher or Lower
//https://leetcode.com/problems/guess-number-higher-or-lower/
//Normal binary search
// Forward declaration of guess API.
// @param num, your guess
// @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
int guess(int num);

class Solution {
public:
    int guessNumber(int n) {
        if(n <= 0) return 0;
        int l = 1, r = n;
        while(l <= r){
            int mid = l + (r - l) / 2;
            int response = guess(mid);
            if(response == 0)
                return mid;
            else if (response > 0){
                l = mid + 1;
            }
            else
                r = mid - 1;
        }
        return 0;
    }
};


//278. First Bad Version
//https://leetcode.com/problems/first-bad-version/
/* Standard binary search */
// Forward declaration of isBadVersion API.
bool isBadVersion(int version);

class Solution {
public:
    int firstBadVersion(int n) {
        int l = 1, r = n;
        while(l < r){
            int mid = l + (r - l) / 2;
            if(isBadVersion(mid))
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }
};


//1341. The K Weakest Rows in a Matrix
//https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/
//Not hard, but a good problem!
class Solution {
    //Binary search!
    int calFirstZeroIndex(vector<int>& v){
        int len = v.size();
        int lo = 0, hi = len - 1;
        if(v[hi] == 1) return hi;
        //I did not get this right during the contest!
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            
            if (v[mid] == 1)
                lo = mid + 1;
            else
                hi = mid;
        }
        
        return hi - 1;
        
    }
public:
    vector<int> kWeakestRows(vector<vector<int>>& mat, int k) {
        int m = mat.size();
        int n = m ? mat[0].size() : 0;
        //Maintain the max Heap
        auto myComp = [](pair<int, int> p1, pair<int, int> p2){
            //We have a trick here, typically we need to make sure the smaller index 
            //goes before larger index. Since this is a max pq, so we need to reverse
            //the order
            if(p1.second == p2.second) return p1.first < p2.first; 
            else return p1.second < p2.second;
        };
        
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(myComp)> pq(myComp);
        

        for(int i = 0; i < m; ++i){
            int firstZero = calFirstZeroIndex(mat[i]);
            //cout << firstZero << endl;
            pq.push({i, firstZero});
            if(pq.size() > k) pq.pop();
        }
        
        vector<int> res(k);
        int cnt = k-1;
        while(!pq.empty()){
            //res.push_back(pq.top().first);
            res[cnt--] = pq.top().first;
            pq.pop();
        }
        return res;
        
    }
};
