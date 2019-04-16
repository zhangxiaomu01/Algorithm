#include<windows.h>
#include<algorithm>
#include<vector>
#include<limits>
#include<cmath>
#include<queue>
#include<utility>
using namespace std;

//48. Rotate Matrix
//https://leetcode.com/problems/rotate-image/
//Note the rotate the 2D matrix (n*n) is equivalent to 
//First reverse the order of rows, and swap each
//pair of diagonal elements swap(M[i][j], M[j][i])
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = m ? matrix[0].size() : 0;
        reverse(matrix.begin(), matrix.end());
        for(int i = 0; i < m; i++){
            for(int j = i+1; j < n; j++){
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
};

/******************************************************************/

//54. Spiral Matrix
//https://leetcode.com/problems/spiral-matrix/
/*
Note how to organize the code is critical...
*/
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = m ? matrix[0].size():0;
        vector<int> res(m*n);
        int l = 0, r = n-1, u = 0, d = m-1, k=0;
        //We use l, r, u, d to set the boundry, and when we hit the boundry
        //We change iterate direction respectively.
        while(true){
            for(int col = l; col <= r; col++) res[k++] = matrix[u][col];
            if(++u > d) break;
            
            for(int row = u;row <= d; row++) res[k++] = matrix[row][r];
            if(--r < l) break;
            
            for(int col = r; col >= l; col--) res[k++] = matrix[d][col];
            if(--d < u) break;
            
            for(int row = d; row >= u; row--) res[k++] = matrix[row][l];
            if(++l > r) break;

        }
        return res;
    }
};

/******************************************************************/

//59. Spiral Matrix II
//https://leetcode.com/problems/spiral-matrix-ii/
//The same idea as Spiral Matrix I
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> finalRes (n, vector<int>(n));
        int res = 1;
        int l = 0, r = n-1, u = 0, d = n-1; 
        while(true){
            for(int col = l; col <= r; col++ ) {
                finalRes[l][col] = res;
                res++;
            }
            if(++u > d) break;;
            for(int row = u; row <= d; row++) {
                finalRes[row][r] = res;
                res++;
            }
            if(--r < l) break;
            for(int col = r; col >= l; col --){
                finalRes[d][col] = res;
                res++;
            }
            if(--d < u) break;
            for(int row = d; row>= u; row --){
                finalRes[row][l] = res;
                res++;
            }
            if(++l > r) break;
        }
        
        return finalRes;
    }
};


/******************************************************************/
//73. Set Matrix Zeroes
//https://leetcode.com/problems/set-matrix-zeroes/
/*
The general idea is that we use the first row and column as a table, 
in the first pass, we check if M[i][j] == 0, we know that corresponding 
M[0][j] and M[i][0] must be 0. so we set the value respectively.
Two flags are set during the first pass to determine whether we have a 0
in first row or column, if there is 0(s), in the end we set first row and 
column to be both 0.
*/
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int row = matrix.size();
        int col = row ? matrix[0].size() : 0;
        //Set two flags to record whether we need to set the first 
        //row or column to be all 0s.
        bool fRow = false, fCol = false;
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                if(matrix[i][j] == 0){
                    if(i == 0) fRow = true;
                    if(j == 0) fCol = true;
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        
        for(int i = 1; i < row; i++){
            for(int j = 1; j < col; j++){
                if(matrix[0][j] == 0 || matrix[i][0] == 0)
                    matrix[i][j] = 0;
            }
        }
        
        if(fRow){
            for(int j = 0; j < col; j++)
                matrix[0][j] = 0;
        }
        if(fCol){
            for(int i = 0; i < row; i++)
                matrix[i][0] = 0;
        }
    }
};


//329. Longest Increasing Path in a Matrix
//https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
/*
This is not a hard problem, however, I got stuck with how to convert the 
pure recursive solution to recursion + memorization. Note the structure of code is
important. Some structures make the problem easier.
*/
//Pure recursive solution, will get time limit exeed error
class Solution {
public:
    int rec(vector<vector<int>>& M, int len, int i, int j){
        int m = M.size();
        int n = m? M[0].size() : 0;
        if(i < 0 || j<0 || i > m || j > n)
            return len;
        int l, r, u, d;
        l = r = u = d = len;
        if(i+1 < m && M[i+1][j] > M[i][j]){
            d++;
            d = rec(M, d, i+1, j);
        }
        if(i-1 >= 0 && M[i-1][j] > M[i][j]){
            u++;
            u = rec(M, u, i-1, j);
        }
        if(j+1 < n && M[i][j+1] > M[i][j]){
            r++;
            r = rec(M, r, i, j+1);
        }
        if(j-1 >= 0 && M[i][j-1] > M[i][j]){
            l++;
            l = rec(M, l, i, j-1);
        }
        
        len = max(len, max(max(u,d), max(l, r)));    
        return len;
    }
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = m ? matrix[0].size() : 0;
        if(m == 0) return 0;
        int res = 0;
        
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                res = max(res, rec(matrix,1, i, j));
            }
        }
        return res;
    }
};
//Recursion + Memorization
class Solution {
public:
    int rec(vector<vector<int>>& M, vector<vector<int>>& memo, int len, int i, int j){
        int m = M.size();
        int n = m? M[0].size() : 0;
        if(i < 0 || j<0 || i > m || j > n)
            return len;
        
        if(memo[i][j]!= 0) return memo[i][j];
        
        int l, r, u, d;
        l = r = u = d = len;
        if(i+1 < m && M[i+1][j] > M[i][j]){
            d = rec(M,memo, d, i+1, j);
        }
        if(i-1 >= 0 && M[i-1][j] > M[i][j]){
            u = rec(M,memo, u, i-1, j);
        }
        if(j+1 < n && M[i][j+1] > M[i][j]){
            r = rec(M,memo, r, i, j+1);
        }
        if(j-1 >= 0 && M[i][j-1] > M[i][j]){
            l = rec(M,memo, l, i, j-1);
        }
        len = max(max(u,d), max(l, r)) + 1;   
        memo[i][j] = len;
        return memo[i][j];
    }
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = m ? matrix[0].size() : 0;
        if(m == 0) return 0;
        int res = 0;
        vector<vector<int>> memo(m, vector<int>(n,0));
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                //if(memo[i][j] == 0)
                res = max(res, rec(matrix,memo, 0, i, j));
            }
        }
        return res;
    }
};

//378. Kth Smallest Element in a Sorted Matrix
//https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
/*
Using heap to keep track of the smallest element in the matrix, each time we pop 1.
In order to save space complexity, we can only store the first column as initial state.
O(klog n)
*/
class Solution {
private:
    struct comp{
       bool operator()(const pair<int, pair<int, int>>& a, const pair<int, pair<int, int>>& b){
           return a.first > b.first;
       } 
    };
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int row = matrix.size();
        int col = row ? matrix[0].size() : 0;
        //We keep track of the smallest element in the heap and its corresponding index
        priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, comp> pQ;
        
        for(int i = 0; i < row; i++)
            pQ.push(make_pair(matrix[i][0], make_pair(i, 0)));
        
        int count = k;
        int res = 0;
        //We iterate each element one by one
        while(count-- > 0){
            res = pQ.top().first;
            int i = pQ.top().second.first;
            int j = pQ.top().second.second;
            pQ.pop();
            if(j < col - 1)
                pQ.push(make_pair(matrix[i][j+1], make_pair(i, j + 1)));
        }
        return res;
    }
};
//Another solution, with time complexity O(nlog(max - min))
class Solution
{
public:
	int kthSmallest(vector<vector<int>>& matrix, int k)
	{
		int n = matrix.size();
		int le = matrix[0][0], ri = matrix[n - 1][n - 1];
		int mid = 0;
		while (le < ri)
		{
			mid = le + (ri-le)/2;
			int num = 0;
			for (int i = 0; i < n; i++)
			{
				int pos = upper_bound(matrix[i].begin(), matrix[i].end(), mid) - matrix[i].begin();
				num += pos;
			}
			if (num < k)
			{
				le = mid + 1;
			}
			else
			{
				ri = mid;
			}
		}
		return le;
	}
};

//74. Search a 2D Matrix
//https://leetcode.com/problems/search-a-2d-matrix/
/*
Basically we can do binary search, note this method we need to convert the 2D
array to a 1D array, and map the index back to 2D when do the real comparison.
This conversion should be careful...
*/
//Binary search
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = matrix.size();
        int col = row ? matrix[0].size() : 0;
        if(row == 0 || col == 0) return false;
        int lo = 0, hi = row*col - 1;
        while(lo < hi){
            int mid = lo + (hi - lo)/2;
            //Note how we convert the 1D index back to 2D index
            int i = mid/col, j = mid%col;
            if(matrix[i][j] == target)
                return true;
            //Note we need shift lo and hi by 1, since we already cover the equal situation
            else if(matrix[i][j] < target)
                lo = mid + 1;
            else
                hi = mid - 1;
        }
        
        return matrix[lo/col][lo%col] == target;
    }
};
/*
Another natural solution is to use two passes, first pass we determine which row our target will be,
then we do binary search for the selected row. 
Binary search, we need to treat coner case carefully...
*/
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if(matrix.empty()||matrix[0].empty()) return false;
        int row = matrix.size(), col = matrix[0].size();
        int head = 0, tail = row - 1;
        if(target < matrix[0][0] || target > matrix[row - 1][col - 1]) return false;
        
        //The loop condition is tricky, we need to shift mid to the right, in order to prevent tail to be negative. 
        while(head<tail && matrix[tail][0] > target){
            int mid = (head + tail + 1)/2;
            if(matrix[mid][0] > target) tail = mid - 1;
            else if(matrix[mid][0] < target) head = mid;
            else return true;
        }
        
        int k = tail;
        tail = col - 1;
        head = 0;
        //Be careful about When will we exit the loop...
        while(head<=tail){
            int mid = (head + tail) /2;
            if(matrix[k][mid] > target) tail = mid - 1;
            else if(matrix[k][mid] < target) head = mid + 1;
            else return true;
        }
        return false;
        
    }
};


//240. Search a 2D Matrix II
/*
A natura approach is to iterate through each row, and do the binary search, the 
time complexity should be O(m log n), we can potentially have some optimization 
like a preprocessing, to get rid of impossible rows.
Another neatural approach is to use priority queue, we keep adding and deleting
elements and compare each of them with target, until we find the one. The time 
complexity shoul be the same, however the space complexity should be O(n).
The following algorithm is great, it runs in O(m+n), we guide the search direction
at each iteration and make the search more efficient!
*/
//Search from lower left to top right
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = matrix.size();
        int col = row ? matrix[0].size() : 0;
        int c = 0, r = row - 1;
        while (r >= 0 && c < col){
            if(matrix[r][c] == target)
                return true;
            else
                matrix[r][c] > target ? r-- : c++;
        }
        return false;
    }
};