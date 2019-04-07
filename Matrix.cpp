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


