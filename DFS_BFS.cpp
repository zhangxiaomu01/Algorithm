//200. Number of Islands
//https://leetcode.com/problems/number-of-islands/
/*
It's a classical problem of BFS... We just sink the island on the fly
*/
//It's a variation of graph traversal (BFS)
class Solution {
private:
    vector<int> d{0,1,0,-1,0};
public:
    int numIslands(vector<vector<char>>& grid) {
        int m = grid.size();
        if(m == 0) return 0;
        int n = grid[0].size();
        int res = 0;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                res += sinkIsland(grid, i, j);
            }
        }
        return res;
    }
    int sinkIsland(vector<vector<char>>& grid, int i, int j){
        if(i < 0 || i == grid.size() || j < 0 || j == grid[0].size() || grid[i][j] == '0')
            return 0;
        grid[i][j] = '0';
        for(int k = 0; k < 4; k++)
            sinkIsland(grid, i+d[k], j+d[k+1]);
        return 1;
    }
};

//130. Surrounded Regions
//https://leetcode.com/problems/surrounded-regions/
/*
Wrong try... Double chek later...
*/
class Solution {
private:
    bool dfs(vector<vector<char>>& B, int i, int j){
        if(i < 0 || i >= B.size() || j < 0 || j >= B[0].size())
            return false;
        if(B[i][j] == 'X' ) return true;
        if(B[i][j] == 'O'){
            B[i][j] = '+';
            if(dfs(B, i+1, j) && dfs(B, i-1, j) && dfs(B, i, j+1) && dfs(B, i, j-1))
                return true;
            B[i][j] = 'O';
        } 
        return false;
    }
public:
    void solve(vector<vector<char>>& board) {
        if(board.empty()) return;
        int row = board.size();
        int col = board[0].size();
        for(int i = 0; i < row; ++i){
            for(int j = 0; j < col; ++j){
                if(board[i][j] == 'O'){
                    dfs(board, i, j);
                }
            }
        }
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                if(board[i][j] == '+')
                    board[i][j] = 'X';
            }
        }
    }
};


