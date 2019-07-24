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

//BFS version
class Solution {
public:
    int numIslands(vector<vector<char>>& nums) {
        int m = nums.size();
        int n = m ? nums[0].size() : 0;
        queue<pair<int, int>> Q;
        int count = 0;
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j){
                if(nums[i][j] == '1'){
                    Q.push(make_pair(i, j));
                    nums[i][j] = '0';
                    count++;
                    while(!Q.empty()){
                        pair<int, int> p = Q.front();
                        int pi = p.first, pj = p.second;
                        Q.pop();
                        //Npte we need to set the next entry to '0'
                        //In order to avoid the repetitive visit to 
                        //the some entries. At first, you just update 
                        //nums[pi][pj] in the while loop once, that's
                        // the problem
                        if(pi + 1 < m && nums[pi+1][pj] == '1'){
                            Q.push(make_pair(pi+1, pj));
                            nums[pi+1][pj] = '0';
                        }    
                        if(pi - 1 >=0 && nums[pi-1][pj] == '1'){
                            Q.push(make_pair(pi-1, pj));
                            nums[pi-1][pj] = '0';
                        }
                        if(pj + 1 < n && nums[pi][pj + 1] == '1'){
                            Q.push(make_pair(pi, pj+1));
                            nums[pi][pj+1] = '0';
                        }   
                        if(pj - 1 >=0 && nums[pi][pj-1] == '1'){
                            Q.push(make_pair(pi, pj-1));
                            nums[pi][pj-1] = '0';
                        }    
                    }
                }
            }
        }
        return count;
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
            if(board[i][col-1] == 'O')
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


//51. N-Queens
//https://leetcode.com/problems/n-queens/
/*
Similar to backtracking, how to use flag to mark whether we have a queen in the block is triky...
*/
class Solution {
public:
/**    | | |                / / /             \ \ \
  *    O O O               O O O               O O O
  *    | | |              / / / /             \ \ \ \
  *    O O O               O O O               O O O
  *    | | |              / / / /             \ \ \ \ 
  *    O O O               O O O               O O O
  *    | | |              / / /                 \ \ \
  *   3 columns        5 135° diagonals     5 45° diagonals    (when n is 3)
  * 
  *   Searching direction is important!
  */
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        vector<string> Q(n, string(n, '.'));
        vector<int> flagCol(n, 1), flag45D(2*n-1, 1), flag135D(2*n-1,1);
        solveNQueens(res, Q, flagCol, flag45D, flag135D, n, 0);
        return res;
        
    }
    
    void solveNQueens(vector<vector<string>> & res, vector<string>& Q, vector<int>&f1, vector<int>& f2, vector<int> &f3, int n, int row){
        if(row == n){
            res.push_back(Q);
            return;
        }
        for(int col = 0; col<n; col++){
            if(f1[col]&& f2[col+row] && f3[n-1 + col - row]){
                f1[col] = f2[col+row] = f3[n-1 + col-row] = 0;
                Q[col][row] = 'Q';
                solveNQueens(res, Q, f1, f2, f3, n, row+1);
                Q[col][row] = '.';
                //Need to reset these three vectors as well
                f1[col] = f2[col+row] = f3[n-1 + col-row] = 1;
            }
        }
    }
};

//52. N-Queens II
//https://leetcode.com/problems/n-queens-ii/
//Exactly the same thing as before
class Solution {
public:
    int totalNQueens(int n) {
        int res = 0;
        vector<int> flagCol(n, 1), flag45D(2*n-1, 1), flag135D(2*n-1,1);
        solveNQueens(res, flagCol, flag45D, flag135D, n, 0);
        return res;
        
    }
    void solveNQueens(int &count, vector<int>&f1, vector<int>& f2, vector<int> &f3, int n, int row){
        if(row == n){
            count++;
            return;
        }
        for(int col = 0; col<n; col++){
            if(f1[col]&& f2[col+row] && f3[n-1 + col - row]){
                f1[col] = f2[col+row] = f3[n-1 + col-row] = 0;
                solveNQueens(count, f1, f2, f3, n, row+1);
                //Need to reset these three vectors as well
                f1[col] = f2[col+row] = f3[n-1 + col-row] = 1;
            }
        }
    }
};

//126. Word Ladder II
//https://leetcode.com/problems/word-ladder-ii/
/*
We are actually doing BFS. Actually, we first build the graph based on the relationship between two words. 
Then we retrieve the ancestor information and rebuild the ladder in the end.
*/
class Solution {
private:
    //Form a graph which records the ancestor for each node
    unordered_map<string, vector<string>> ancestors;
    vector<string> ladder;
    vector<vector<string>> ladders;
    
    void getChildren(string& word, unordered_set<string>& next, unordered_set<string>& dict){
        int len = word.size();
        //Make a copy of the word, when we change word, we can easily build the ancestor relationship in genLadders function
        string ancestor = word;
        for(int i = 0; i < len; i++){
            char tempC = word[i];
            for(int j = 0; j < 26; j++){
                char c = 'a' + j;
                word[i] = c;
                if(dict.find(word) != dict.end()){
                    next.insert(word);
                    ancestors[word].push_back(ancestor);
                }
            }
            word[i] = tempC;
        }
    }
    //Rebuild the word ladders based on the ancestor information
    void genLadders(const string& s, const string& e){
        if(s == e){
            reverse(ladder.begin(), ladder.end());
            ladders.push_back(ladder);
            reverse(ladder.begin(), ladder.end());
            return;
        }
        for(string ancestor : ancestors[e]){
            ladder.push_back(ancestor);
            genLadders(s, ancestor);
            ladder.pop_back();
        }
        
    }
    
public:
    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> dict(wordList.begin(), wordList.end());
        if(dict.find(endWord) == dict.end()) return ladders;
        dict.insert(beginWord);
        dict.insert(endWord);
        //We define our current level and next level, and will swap later
        unordered_set<string> cur, next;
        cur.insert(beginWord);
        ladder.push_back(endWord);
        
        while(true){
            //We first need to remove all the potential strings from our dictionary to prevent repetitive visit
            for(string s : cur)
                dict.erase(s);
            //Find the elements in successive layer
            for(string s: cur)
                getChildren(s, next, dict);
            
            //If we cannot find any connection from the above, we immediately return ladders
            if(next.empty()) {
                return ladders;
            }
            
            if(next.find(endWord) != next.end()){
                genLadders(beginWord, endWord);
                return ladders;
            }
            cur.clear();
            swap(cur, next);
        }
        return ladders;
    }
};