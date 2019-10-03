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
//https://leetcode.com/problems/search-a-2d-matrix-ii/
//The most natural approach! binary search for each row
class Solution {
    bool searchVector(const vector<int>& v, int target){
        int len = v.size();
        int l = 0, r = len-1;
        while(l <= r){
            int mid = l + (r - l) / 2;
            if(v[mid] == target) return true;
            else if (v[mid] < target) l = mid + 1;
            else r = mid - 1;
        }
        return false;
    }
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        int n = m ? matrix[0].size() : 0;
        if(!m || !n) return false;
        for(int i = 0; i < m; ++i){
            if(target == matrix[i].front() || target == matrix[i].back())
                return true;
            if(target < matrix[i].front())
                return false;
            if(target > matrix[i].back())
                continue;
            if(searchVector(matrix[i], target)) return true;
        }
        return false;
    }
};

//Divide and conquer! Dammn, it's hard to get it right!!!
//Do not try to implement it during the interview. Using smart method!
class Solution {
    bool binarySearch(vector<vector<int>>& M, int target, int up, int left, int bottom, int right){
        if(left > right || up > bottom) return false;
        if(target < M[up][left] || target > M[bottom][right])
            return false;
        int l = left, r = right, u = up, b = bottom;
        while(l <= r && u <= b){
            int midX = l + (r - l) / 2;
            int midY = u + (b - u) / 2;
            if(M[midY][midX] == target) return true;
            else if(M[midY][midX] < target){
                u = midY < M.size() - 1 ? midY + 1 : M.size() - 1;
                l = midX < M[0].size() - 1 ? midX + 1: M[0].size() - 1;
               
            }else{
                b = midY > 0 ? midY - 1 : 0;
                r = midX > 0 ? midX - 1 : 0;
            }
        }
        //check left bottom and right top
        //Tricky part!
        return binarySearch(M, target, u, left, bottom, l-1) || 
            binarySearch(M, target, up, l, u-1, right);
        
    }
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        int n = m ? matrix[0].size() : 0;
        if(!m || !n || target < matrix[0][0] || target > matrix[m-1][n-1]) 
            return false;
        return binarySearch(matrix, target, 0, 0, m-1, n-1);
    }
};


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

//79. Word Search
//https://leetcode.com/problems/word-search/
/*
A natural approach is to do BFS for each characters and
using backtracking.
*/
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        int row = board.size();
        int col = row ? board[0].size() : 0;
        int index = 0;
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                if(search(board, i, j, word, index)) return true;
            }
        }
        return false;
    }
    bool search(vector<vector<char>>& B, int i, int j, string& W, int index){
        if(index == W.size())
            return true;
        //Set the false condition...
        if(i < 0 || i > B.size()-1 || j < 0 || j > B[0].size()-1 || B[i][j] != W[index])
            return false;
        bool res;
        char tempC = B[i][j];
        B[i][j] = '-';
        res = search(B, i+1, j, W, index+1)||search(B, i-1, j, W, index+1)||
              search(B, i, j+1, W, index+1)||search(B, i, j-1, W, index+1);
        B[i][j] = tempC;
        return res;
    }
};

//212. Word Search II
//https://leetcode.com/problems/word-search-ii/
/*
A natural approach will be the same as word search I, we keep searching each word from
the vector, and for each string, we will do an exhaustive search, the time complexity is
huge. For each string, the time complexity should be O(mn * mn), if we have k string, then
in total, it will be O(k*(mn)^2)
A better way is to do a preprocessing first, we first preprocessing all the strings, and 
save them to a trie, so we can compare all of them together. This is a hard problem, not 
only the logic, but also involves too much coding.
*/
struct TrieNode {
    bool isWord;
    TrieNode* nextC[26];
    TrieNode():isWord(false){
        for(int i = 0; i < 26; i++){
            nextC[i] = nullptr;
        }
    }
};
//Implementing trie data structure here
//Without deconstructor, will have memory leak
class Trie{
private:
    TrieNode* root = new TrieNode();
    bool findWord(string& s, TrieNode*& p){
        for(char c: s){
            int index = c - 'a';
            if(!p->nextC[index])
                return false;
            p = p->nextC[index];
        }
        return true;
    }
public:
    Trie(){};
    void insertWord(string& s){
        TrieNode* p = root;
        for(char c:s){
            int index = c - 'a';
            if(!p->nextC[index]){
                p->nextC[index] = new TrieNode();
            }
            p = p->nextC[index];
        }
        p->isWord = true;
    }
    bool searchWord(string& s){
        TrieNode* p = root;
        if(!findWord(s, p)){
            return false;
        }
        bool res = p->isWord;
        //Whenever we find a valid word from words list,
        //We eliminate it from our trie.
        p->isWord = false;
        return res;
    }
    
    bool searchPrefix(string& s){
        TrieNode* p = root;
        if(!findWord(s, p)){
            return false;
        }
        return true;
    }
    
};
class Solution {
private:
    //BFS for all the potential word
    void matching(Trie& trie, vector<vector<char>>& b, vector<string>& res, string& temp, int i, int j){
        int m = b.size();
        int n = b[0].size();
        if(i < 0 || j < 0|| i>=m || j >= n || b[i][j] == '0'){
            return;
        }
        //Backtracking temp
        temp.push_back(b[i][j]);
        if(!trie.searchPrefix(temp)){
            temp.pop_back();
            return;
        }
        if(trie.searchWord(temp)){
            res.push_back(temp);
        }
        char tempC = b[i][j];
        b[i][j] = '0';
        matching(trie, b, res, temp, i-1, j);
        matching(trie, b, res, temp, i+1, j);
        matching(trie, b, res, temp, i, j-1);
        matching(trie, b, res, temp, i, j+1);
        b[i][j] = tempC;
        temp.pop_back();
    }
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        int m = board.size();
        int n = m ? board[0].size() : 0;
        vector<string> res;
        if(m == 0 || words.size() == 0) return res;
        Trie trie;
        for(auto& s: words){
            trie.insertWord(s);
        }
        string word("");
        //We have to start with potentially every grid
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                matching(trie, board, res, word, i, j);
                if(res.size() >= words.size())
                    return res;
            }
        }
        return res;
    }  
};


//36. Valid Sudoku
//https://leetcode.com/problems/valid-sudoku/
/*
The problem is not hard, we only need to check whether each row/column/black contains valid
1-9 numbers. A natural approach is we check each row, column and block (by calculating the 
entry of each block), and check whether we have duplicate numbers by saving existing number 
to a 1D array. When we find a conflict exists we know it's invalid.
*/
class Solution {
private:
    bool hasDuplicate(const vector<vector<char>>& B, int iS, int iE, int jS, int jE){
        int* dict = new int[9];
        for(int i = 0; i < 9; i++)
            dict[i] = 0;
        for(int i = iS; i < iE; i++){
            for(int j = jS; j < jE; j++){
                if(B[i][j] != '.'){
                    int index = B[i][j] - '0' - 1;
                    if(dict[index] != 0)
                        return true;
                    dict[index] = 1;
                }
            }   
        }
        delete []dict;
        return false;
    }
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        //We first check each row
        for(int i = 0; i < 9; i++){
            if(hasDuplicate(board, i, i+1, 0, 9))
                return false;
        }
        //Then we check each column
        for(int i = 0; i < 9; i++){
            if(hasDuplicate(board, 0, 9, i, i+1)){
                return false;
            }
        }
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                if(hasDuplicate(board, i*3, i*3 + 3, j*3, j*3 + 3))
                    return false;
            }
        }
        return true;
    }
};
/*
Another approach is to allocate 3 9*9 array, which represent all rows/columns/blocks, and 
check whether we have invalid duplicates in each of the array.
*/
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        //C++ 11 suppor this, or we need more complex way to define 2D array
        auto row = new int[9][9], col = new int[9][9], block = new int[9][9];
        
        for(int i = 0; i < 9; i++)
            for(int j = 0; j < 9; j++)
                row[i][j] = col[i][j] = block[i][j] = 0;

        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                //Note k is the ID for 3*3 bloack, from 0 - 8 , top left to bottom right
                int k = i/3 * 3 + j/3;
                if(board[i][j] != '.'){
                    int n = board[i][j] - '1';
                    if(row[i][n]||col[j][n]||block[k][n])
                        return false;
                    row[i][n] = col[j][n] = block[k][n] = 1;
                }
            }
        }
        delete []row;
        delete []col;
        delete []block;
        return true;
    }
};


//37. Sudoku Solver
//https://leetcode.com/problems/sudoku-solver/
/*
A natural way to solve this problem is using recursion and backtracking. We check each grid
and test whether fill in the 1-9 numbers will make the sudoku invalid, after that, if we fill
in a number which is valid, we then check the next grid, until we finish all the check.
*/
class Solution {
public:
    void solveSudoku(vector<vector<char>>& board) {
        solve(board, 0, 0);
    }
    //i means row, j means column
    bool solve(vector<vector<char>>& B, int i, int j){
        if(j == B[0].size()){
            j = 0;
            //We need increase the row by 1, because j hits the end
            i++;
            if(i == B.size())
                return true;
        }
        if(B[i][j] != '.')
            return solve(B, i, j+1);
        
        for(int k = 1; k <= B.size(); k++){
            //if(isValid(B, i, j, k+'0')){
            if(isValid(B, i, j, k+'0')){
                B[i][j] = k + '0';
                if(solve(B, i, j+1))
                    return true;
            }
            B[i][j] = '.';
        }
        return false;
    }
    bool isValid(vector<vector<char>>& B, int i, int j, char val){
        if(any_of(B[i].begin(), B[i].end(), [val](int a){return a == val;}))
            return false;
        
        for(int p = 0; p < B.size(); p++){
            if(B[p][j] == val)
                return false;
        }
        
        int regionSize = sqrt(B.size()); // 3
        int I = i / regionSize, J = j / regionSize;
        for(int a = 0; a < regionSize; a++){
            for(int b = 0; b < regionSize; b++){
                if(B[I*regionSize + a][J*regionSize + b] == val)
                    return false;
            }
        }
        return true;
    }
};

/*
A slightly more efficient implementation. Note the idea is exactly the same.
*/
class Solution {
public:
    bool checkSodoku(vector<vector<char>> &board, int i, int j, char c){
        int x = i - i % 3, y = j - j % 3;
        for(int k = 0; k < 9; k++) {
            if(board[i][k] == c) return false; 
            if(board[k][j] == c) return false;
        }
        for(int p = 0; p < 3; p++){
            for(int q= 0; q < 3; q++){
                if(board[p+x][q+y] == c) return false;
            }
        }
        return true;
    }
    bool sudoku(vector<vector<char>> &board, int i, int j){
        if(i == 9) return true;
        if(j == 9) return sudoku(board, i+1, 0);
        if(board[i][j] != '.') return sudoku(board, i, j+1);
        
        for(char c = '1'; c <= '9'; c++){
            if(checkSodoku(board, i, j, c)){
                board[i][j] = c;
                if(sudoku(board, i, j + 1)) return true;
                board[i][j] = '.';
            }
        }
        return false;
        
    }
    void solveSudoku(vector<vector<char>>& board) {
        sudoku(board, 0, 0);
    }

};

//289. Game of Life
//https://leetcode.com/problems/game-of-life/
//For this problem, we need to find a way ro keep track of the changes without
//duplicating the original 2D board.
//We do not need to make any change for rule 2
//In this algorithm, if a cell dies, than we set it to -1
//Then we know it changes from 1 (live) to dead(0).
//If a cell becomes alive, we set to 2, then we know
//that it changes from -1(dead) to 2(live). We can use these
//two states to keep track of the potential changes. (e.g. if 
//the state is -1, we know that originally, it should be alive)
class Solution {
public:
    void gameOfLife(vector<vector<int>>& board) {
        int m = board.size(), n = m ? board[0].size() : 0;
        //Shift the row and column to calculate the neighbours of board[row][col]
        int neighbours[3] = {-1, 0, 1};
        
        for(int row = 0; row < m; ++row){
            for(int col = 0; col < n; ++col){
                int liveNeighbours = 0;
                //Check the neighbours of board[row][col]
                for(int i = 0; i < 3; ++i){
                    for(int j = 0; j < 3; ++j){
                        int r = row + neighbours[i];
                        int c = col + neighbours[j];
                        if(r == row && c == col) continue;
                        //abs(board[r][c]) == 1 will give us the exact live neighbours
                        if(r >=0 && r < m && c >=0 && c < n && abs(board[r][c]) == 1)
                            liveNeighbours++;
                    }
                }
                
                //Apply rule 1 and rule 3. Basically change 1 to -1 when necessary
                if(board[row][col] == 1 && (liveNeighbours > 3 || liveNeighbours < 2))
                    board[row][col] = -1;
                //Apply rule 4
                if(board[row][col] == 0 && liveNeighbours == 3)
                    board[row][col] = 2;
                
            }
        }
        
        for(int row = 0; row < m; ++row){
            for(int col = 0; col < n; ++col){
                if(board[row][col] == -1) board[row][col] = 0;
                else if (board[row][col] == 2) board[row][col] = 1;
            }
        }
        
    }
};

//311. Sparse Matrix Multiplication (Locked)
/* Description:
Given two sparse matrices A and B, return the result of AB.
You may assume that A's column number is equal to B's row number.
Example:
Input:

A = [
  [ 1, 0, 0],
  [-1, 0, 3]
]

B = [
  [ 7, 0, 0 ],
  [ 0, 0, 0 ],
  [ 0, 0, 1 ]
]

Output:

     |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
                  | 0 0 1 |

 */
/* The general idea is that we first store the index with non-zero element to
a set. Then when we actually do the multiplication, we can skip the elements 
from each row (B) or column (A). */
class Solution {
public:
    vector<vector<int>> multiply(vector<vector<int>>& A, vector<vector<int>>& B) {
        int M = A.size(), N = B.size(), K = B[0].size();
        vector<unordered_set<int>> m_vs(M, unordered_set<int>()), k_vs(K, unordered_set<int>());
        
        // A: M x N
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (A[i][j]) m_vs[i].insert(j);
            }
        }
        
        // B: N x K
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                if (B[j][i]) k_vs[i].insert(j);
            }
        }
        
        vector<vector<int>> result(M, vector<int>(K, 0));
        
        for (int i = 0; i < M; ++i) {   // M
            for (int j = 0; j < K; ++j) {   // K
                // Pick the one with less non-zero to iterate
                if (m_vs[i].size() <= k_vs[j].size()) {
                    for (auto const index:m_vs[i]) {    // N
                        if (k_vs[j].find(index) != k_vs[j].end()) {
                            result[i][j] += A[i][index] * B[index][j];
                        }   
                    }
                }
                else {
                    for (auto const index:k_vs[j]) {    //  N
                        if (m_vs[i].find(index) != m_vs[i].end()) {
                            result[i][j] += A[i][index] * B[index][j];
                        }   
                    }
                }
            }
        }
        
        return result;
    }
};


//419. Battleships in a Board
//https://leetcode.com/problems/battleships-in-a-board/
//Straightforward solution!
class Solution {
public:
    int countBattleships(vector<vector<char>>& board) {
        int m = board.size();
        int n = m ? board[0].size() : 0;
        int count = 0;
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j){
                if(board[i][j] == '+') continue;
                if(board[i][j] == 'X'){
                    count ++;
                    board[i][j] = '+';
                    int curI = i;
                    int curJ = j;
 
                    while(i+1 < m && board[i+1][j] == 'X'){
                        i++;
                        board[i][j] = '+';
                    }
                        
                    i = curI;
                    while(j+1 < n && board[i][j+1] == 'X'){
                        j++;
                        board[i][j] = '+';
                    }
                        
                    i = curI;
                    j = curJ;
                }
            }
        }
        return count;
    }
};

//Follow up, one pass, no extra space, without modifying the matrix
class Solution {
public:
    int countBattleships(vector<vector<char>>& board) {
        int m = board.size();
        int n = m ? board[0].size() : 0;
        int count = 0;
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j){
                count += board[i][j] == 'X' && (i == 0 || board[i-1][j] != 'X') && (j==0 || board[i][j-1] != 'X');
            }   
        }
        return count;
    }
};


//939. Minimum Area Rectangle
//https://leetcode.com/problems/minimum-area-rectangle/
//Brute Force solution
//Note how to define your own hash function is critical here!
struct Hash{
     size_t operator()(const pair<int, int>& p) const{
         return hash<long long>()(((long long)p.first << 32) ^ ((long long)p.second));
     }
};
class Solution {
public:
    int minAreaRect(vector<vector<int>>& points) {
        unordered_set<pair<int, int>, Hash> uSet;
        int res = numeric_limits<int>::max();
        for(auto&p : points){
            int x1 = p[0], y1 = p[1];
            for(auto&[x2, y2] : uSet){
                if(uSet.count({x1, y2}) && uSet.count({x2, y1})){
                    int localArea = abs(y2 - y1) * abs(x2 - x1);
                    res = min(localArea, res);
                }
            }
            uSet.insert({x1, y1});
        }
        //We could potentially have no rectangle!
        return res == numeric_limits<int>::max() ? 0 : res;
    }
};


//Heavily optimized version. Tricky, too many tricks and not easy to implement
struct Hash{
    unsigned int operator()(const pair<int, int>& p) const {
        return hash<long long>()(((long long)p.first << 32) ^ ((long long)p.second));
    }
};
class Solution {
private:
    //count the maximum different coordinates for x and y
    pair<int, int> countCoord(vector<vector<int>>& points){
        unordered_set<int> sx, sy;
        for(auto& p : points){
            sx.insert(p[0]);
            sy.insert(p[1]);
        }
        return make_pair(sx.size(), sy.size());
    }
public:
    int minAreaRect(vector<vector<int>>& points) {
        auto [nx, ny] = countCoord(points);
        //We need to make sure the key will be sorted! so in the end,
        //we can safely using x2 - x
        map<int, vector<int>> pMap;
        //Make sure we always make the list for each entry shorter. 
        //Optimization! since index x and y has the same logic
        //It does not matter if we switch them
        if(nx > ny) {
            for(auto& p : points)
                pMap[p[0]].push_back(p[1]);
        }else{
            for(auto& p : points)
                pMap[p[1]].push_back(p[0]);
        }
        
        int res = INT_MAX;
        unordered_map<pair<int, int>, int, Hash> uMap;
        for(auto&[x, vy] : pMap){
            sort(vy.begin(), vy.end());
            for(int i = 1; i < vy.size(); ++i){
                for(int j = 0; j < i; ++j){
                    int y1 = vy[j], y2 = vy[i];
                    if(uMap.count({y1, y2})){
                        //since {y1, y2} has already been inserted, we
                        //know that x1 must be smaller.
                        int x1 = uMap[{y1, y2}];
                        int x2 = x;
                        res = min(res, (x2 - x1) * (y2 - y1));
                    }
                    //indicates that current x can form a range between
                    //[y1, y2] (y1 <= y2)
                    uMap[{y1, y2}] = x;
                }
            }
            
        }
        return res == INT_MAX ? 0 : res;
        
    }
};



//463. Island Perimeter
//https://leetcode.com/problems/island-perimeter/
//Naive implementation is trivial! The following are interesting idea!
class Solution {
public://1035
    int islandPerimeter(vector<vector<int>>& grid) {
        if(grid.empty()) return 0;
        int count = 0, repeat = 0;
        for(int i = 0; i < grid.size(); i++)
            for(int j = 0; j < grid[0].size(); j++){
                if(grid[i][j] == 1){
                    count++;
                    if(i && grid[i - 1][j] == 1)  repeat++;
                    if(j && grid[i][j - 1] == 1)  repeat++;
                }
            }
        return 4 * count - 2 * repeat;
    }
};
