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
Wrong answer... I am trying to flip the 'O' in the middle of the matrix. It's hard to determine when you need to return.
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

/*
Note in this problem, we have to search from the boundary... In my first try, I try to flip the 'O' in the center, but it is very hard to determine when we need to return. 
Then a more natural approach is to search from the boundry, and then flip the 'O' to '+', finally, we will flip all 'O' to 'X', all '+' to 'O'
*/
class Solution {
private:
    void dfs(vector<vector<char>>& B, int i, int j){
        if(i < 0 || i >= B.size() || j < 0 || j >= B[0].size() || B[i][j] == '+')
            return;
        if(B[i][j] == 'O'){
            B[i][j] = '+';
            dfs(B, i+1, j);
            dfs(B, i-1, j);
            dfs(B, i, j+1);
            dfs(B, i, j-1);
        }
    }
public:
    void solve(vector<vector<char>>& board) {
        if(board.empty()) return;
        int row = board.size();
        int col = board[0].size();
        for(int i = 0; i < row; ++i){
            if(board[i][0] == 'O')
                dfs(board, i, 0);
            if(board[i][col-1])
                dfs(board, i, col-1);
        }
        for(int j = 0; j < col; ++j){
            if(board[0][j] == 'O')
                dfs(board, 0, j);
            if(board[row-1][j] == 'O')
                dfs(board, row-1, j);
        }
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                if(board[i][j] == 'O')
                    board[i][j] = 'X';
                else if(board[i][j] == '+')
                    board[i][j] = 'O';
            } 
        }
    }
};

//127. Word Ladder
//https://leetcode.com/problems/word-ladder/
/*
The concept is not hard, the problem is too much coding work.
The bidirectional BFS needs more practice... Note how we build the level one by one and maintain the pointer points to some layer...
*/
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        if(wordList.empty()) return 0;
        unordered_set<string> wl;
        unordered_set<string> start, end;
        unordered_set<string>* pStart, *pEnd;
        for(string& s : wordList) wl.insert(s);
        
        //We return 0 immediately if we cannot find endList in wordList
        if(wl.find(endWord) == wl.end()) return 0;
        
        start.insert(beginWord);
        end.insert(endWord);
        int ladderLength = 2;
        
        //Note we can guarantee in the end one of them will be empty through swap(*pStart, levelWL)
        while(!start.empty() && !end.empty()){
            //We implement bidirectional BFS here, we change the search direction for each iteration
            if(start.size() > end.size()){
                pStart = &end;
                pEnd = &start;
            }else{
                pStart = &start;
                pEnd = &end;
            }
            //Store the potential word for current level
            unordered_set<string> levelWL;
            for(unordered_set<string>::iterator it = pStart->begin(); it != pStart->end(); it++){
                string currentWord = *it;
                wl.erase(currentWord);
                for(int i = 0; i < currentWord.size(); i++){
                    char c = currentWord[i];
                    for(int j = 0; j < 26; j++){
                        currentWord[i] = 'a' + j;
                        //Our search meets from both direction, return the length
                        if(pEnd->find(currentWord) != pEnd->end()) return ladderLength;
                        //We build a new level here, and delete the word from wl to indicate that we already visit this word
                        if(wl.find(currentWord) != wl.end()){
                            levelWL.insert(currentWord);
                            wl.erase(currentWord);
                        }
                    }
                    //Note we have to change back the character to prepare for the next round
                    currentWord[i] = c;
                }
            }
            ladderLength++;
            //Here we mark swap the internal container of pStart and levelWL, which means we go to the current level for the next iteration
            //We cannot simply do pStart = &levelWL, since levelWL is allocated in stack and will be destroyed after the iteration.
            swap(*pStart, levelWL);
        }
        return 0;
    }
};


