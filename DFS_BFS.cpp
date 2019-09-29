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
                        //the same entries. At first, you just update 
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


//397. Integer Replacement
//https://leetcode.com/problems/integer-replacement/
/* Not Efficient 
We can allocate an array to store the result, and do memorization.
It's not that good since the intermediate result will barely be used again.*/
class Solution {
private:
    int helper(long n){
        if(n == 1) return 0;
        if(n == 2) return 1;
        if(n % 2 == 0) return helper(n/2) + 1;
        else return min(helper(n-1), helper(n+1)) + 1;
    }
public:
    int integerReplacement(int n) {
        return helper(n);
    }
};

//Memorization, we cannot use array here for it will exceed
//memory limitation
class Solution {
private:
    unordered_map<int, int> uMap;
    int helper(long n){
        if(n == 1) return 0;
        if(n == 2) return 1;
        if(uMap.count(n) > 0) return uMap[n];
        if(n % 2 == 0) {
            uMap[n] = helper(n/2) + 1;
        }
        else {
            uMap[n] = min(helper(n-1), helper(n+1)) + 1;
        }
        return uMap[n];
    }
public:
    int integerReplacement(int n) {
        return helper(n);
    }
};

//DP solution
//Cannot pass test case, because it exceeds the maximum memory limitation
class Solution {
public:
    int integerReplacement(int n) {
        int* dp = new int[n+1];
        memset(dp, 0x3F3F3F3F, sizeof(dp));
        dp[1] = 0;
        for(int i = 2; i <= n; ++i){
            //dp[i/2 + 1] + 1 == dp[i+1] when i is odd
            //If we use int dp[n], will have stack overflow, because n could be large
            dp[i] = (i%2 == 0) ? (dp[i/2]) + 1 : min(dp[i-1], dp[(i+1)/2] + 1) + 1;
            
        }
        int res = dp[n];
        delete[] dp;
        return res;
    }
};


//The most updated version, however, it's almost impossible to get it during the 
//interview
class Solution 
{
    int res = 0;
public:
    int integerReplacement(int n) 
    {
        if (n == 1)
            return res;
        if (n == 3)
        {
            res += 2;
            return res;
        }
        if (n == INT_MAX)
            return 32;
        
        res ++;
        if (n & 1)
            if ((n + 1) % 4 == 0)
                integerReplacement(n + 1);
            else
                integerReplacement(n - 1);
        else
            integerReplacement(n / 2);
            
        return res;
    }
};


//417. Pacific Atlantic Water Flow
//https://leetcode.com/problems/pacific-atlantic-water-flow/
//Unoptimized DFS solution! too many repetitive search!
class Solution {
private:
    int m_row, m_col;
    
    void dfs(vector<vector<int>>& M, int i, int j, pair<bool, bool>& detector){
        if(i < 0 || j < 0 || i >= m_row || j >= m_col) return;
        if(i == 0 || j == 0){
            detector.first = true;
        }
        if(i == m_row-1 || j == m_col-1){
            detector.second = true;
        }
        if(detector.first && detector.second) return;
        
        int tempNum = M[i][j];
        if(i > 0 && M[i][j] >= M[i-1][j]){
            M[i][j] = INT_MAX;
            dfs(M, i-1, j, detector);
            M[i][j] = tempNum;
        }
        if(j > 0 && M[i][j] >= M[i][j-1]){
            M[i][j] = INT_MAX;
            dfs(M, i, j-1, detector);
            M[i][j] = tempNum;
        }
        if(i < m_row-1 && M[i][j] >= M[i+1][j]){
            M[i][j] = INT_MAX;
            dfs(M, i+1, j, detector);
            M[i][j] = tempNum;
        }
        if(j < m_col-1 && M[i][j] >= M[i][j+1]){
            M[i][j] = INT_MAX;
            dfs(M, i, j+1, detector);
            M[i][j] = tempNum;
        }
    }
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
        m_row = matrix.size();
        m_col = m_row ? matrix[0].size() : 0;
        vector<vector<int>> res;
        
        for(int i = 0; i < m_row; ++i){
            for(int j = 0; j < m_col; ++j){
                pair<bool, bool> localPair({false, false});
                dfs(matrix, i, j, localPair);
                if(localPair.first && localPair.second)
                    res.push_back(vector<int>({i, j}));
            }
        }
        return res;
    }
};


//Heavily optimized solution! Hard to get the idea during the interview!
//The key is to come up with a solution to utilize the m_visited array to store
//the result!
class Solution {
private:
    int m_row, m_col;
    //Note m_visited is global, then we can update it from 4 directions
    //m_visited represents whenther grid[i][j] can reach the boundries!
    vector<vector<int>> m_visited;
    //label means we already reached which side. 
    //when m_visited[i][j] == 3, we know we can reach from both sides.
    void DFS(vector<vector<int>>& M, int i, int j, int pre, int label, vector<vector<int>>& res){
        //Note m_visited[i][j] == label, we need to return as well
        //since we already checked grid[i][j] can come from label's side
        if(i < 0 || i >= m_row || j < 0 || j >= m_col || M[i][j] < pre || m_visited[i][j] == label || m_visited[i][j] == 3)
            return;
        m_visited[i][j] += label;
        //std::cout << m_visited[i][j] << std::endl;
        if(m_visited[i][j] == 3) res.push_back({i, j});
        DFS(M, i-1, j, M[i][j], label, res);
        DFS(M, i+1, j, M[i][j], label, res);
        DFS(M, i, j+1, M[i][j], label, res);
        DFS(M, i, j-1, M[i][j], label, res);
    }
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
        m_row = matrix.size();
        m_col = m_row ? matrix[0].size() : 0;
        vector<vector<int>> res;
        m_visited.resize(m_row, vector<int>(m_col, 0));
        //If we search from pacific to atlantic, then we will have label 1
        //meas that the grid to be visited can reach pacific. Vice versa!
        for(int i = 0; i < m_row; ++i){
            //we start from pacific
            DFS(matrix, i, 0, matrix[i][0], 1, res);
            //we start from atlantic
            DFS(matrix, i, m_col-1, matrix[i][m_col-1], 2, res);
        }
        for(int j = 0; j < m_col; ++j){
            DFS(matrix, 0, j, matrix[0][j], 1, res);
            DFS(matrix, m_row-1, j, matrix[m_row-1][j], 2, res);
        }
        //Since we already searched from all directions, which means 
        //we already get the result!
        return res;
    }
};


//BFS version, in general, slower than DFS version!
class Solution {
private:
    vector<vector<int>> offset = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    void BFS(vector<vector<int>>& M, queue<vector<int>>& Q, vector<vector<int>>& visited, int label){
        int m = M.size();
        int n = m ? M[0].size() : 0;
        while(!Q.empty()){
            auto cur = Q.front();
            Q.pop();
            for(int k = 0; k < offset.size(); ++k){
                int indexI = cur[0] + offset[k][0];
                int indexJ = cur[1] + offset[k][1];
                if(indexI < 0 || indexJ < 0 || indexI >= m || indexJ >= n || visited[indexI][indexJ] == 3 || visited[indexI][indexJ] == label || M[cur[0]][cur[1]] > M[indexI][indexJ])
                    continue;
                visited[indexI][indexJ] += label;
                Q.push(vector<int>({indexI, indexJ}));
            }
        }
    }
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = m ? matrix[0].size() : 0;
        vector<vector<int>> res;
        if(m == 0 || n == 0) return res;
        //stores value represents whether [i,j] can reaches both ocean
        //1 - can reach pacific ocean
        //2 - can reach atlantic ocean
        //3 - can reach both
        //0 - cannot reach either ocean
        vector<vector<int>> visited(m, vector<int>(n, 0));
        queue<vector<int>> pacificQ, atlanticQ;
        for(int i = 0; i < m; ++i){
            pacificQ.push({i, 0});
            visited[i][0] = visited[i][0] == 1 ? 1 : visited[i][0]+1;
            atlanticQ.push({i, n-1});
            visited[i][n-1] = visited[i][n-1]==2 ? 2 : visited[i][n-1]+2;
        }
        for(int j = 0; j < n; ++j){
            pacificQ.push({0, j});
            visited[0][j] = visited[0][j] == 1 ? 1 : visited[0][j]+1;
            atlanticQ.push({m-1, j});
            visited[m-1][j] = visited[m-1][j]==2 ? 2 : visited[m-1][j]+2;
        }
        //Handle the corner case when either row or column has only 1 
        //element!
        if(m == 1) {
            visited[0][0] = 3;
            visited[0][n-1] = 3;
        }
        if(n == 1){
            visited[0][0] = 3;
            visited[m-1][0] = 3;
        }
        BFS(matrix, pacificQ, visited, 1);
        BFS(matrix, atlanticQ, visited, 2);
        
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j){
                if(visited[i][j] == 3)
                    res.push_back({i, j});
            }
        }
        return res;
    }
};


//947. Most Stones Removed with Same Row or Column
//https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/
//DFS solution. We consider each stone as a node in graph (in this 
//code, slightly different). The number of stones we can remove 
//equals to N - number of connected components. (Since each component
//we must leave at least 1 stone)
/*
When we search on points,
we alternately change our view on a row and on a col.

We think:
a row index, connect two stones on this row
a col index, connect two stones on this col.

In another view：
A stone, connect a row index and col.

Have this idea in mind, the solution can be much simpler.
The number of islands of points,
is the same as the number of islands of indexes.
*/
class Solution {
public:
    int removeStones(vector<vector<int>>& stones) {
        //Adjacent list representation!
        unordered_map<int, vector<int>> graph;
        for(auto& v : stones){
            //~x = -x - 1, equivalenet to x = x + 10000
            //We just make sure row and col can be represented in
            //the same map
            //Note the graph, the stone is edge between two indices
            //However, the connected component should be the same
            graph[v[0]].push_back(~v[1]);
            graph[~v[1]].push_back(v[0]);
        }
        
        int numOfComponent = 0;
        unordered_set<int> Visited;
        stack<int> tobeVisited;
        //Iterative DFS
        for(int i = 0; i < stones.size(); ++i){
            for(int j = 0; j < 2; ++j){
                //A little trick to determine the index
                int s = j == 0 ? stones[i][0] : ~stones[i][1];
                if(Visited.count(s) == 0){
                    tobeVisited.push(s);
                    numOfComponent++;
                    while(!tobeVisited.empty()){
                        int index = tobeVisited.top();
                        Visited.insert(index);
                        tobeVisited.pop();
                        for(int neighbor : graph[index]){
                            if(Visited.count(neighbor) == 0)
                                tobeVisited.push(neighbor);
                        }
                    }
                }
            }
        }
        return stones.size() - numOfComponent;
    }
};


//DFS solution ver 2
class Solution {
private:
    void DFS(int index, unordered_map<int, vector<int>>& G, unordered_set<int>& visited){
        visited.insert(index);
        for(int neighbor : G[index]){
            if(visited.count(neighbor) == 0)
                DFS(neighbor, G, visited);
        }
    }
public:
    int removeStones(vector<vector<int>>& stones) {
        unordered_map<int, vector<int>> graph;
        for(auto& v : stones){
            graph[v[0]].push_back(~v[1]);
            graph[~v[1]].push_back(v[0]);
        }
        
        int numOfComponent = 0;
        unordered_set<int> Visited;
        for(int i = 0; i < stones.size(); ++i){
            for(int j = 0; j < 2; ++j){
                int s = j == 0 ? stones[i][0] : ~stones[i][1];
                if(Visited.count(s) == 0){
                    numOfComponent++;
                    DFS(s, graph, Visited);
                }
            }
        }
        
        return stones.size() - numOfComponent;
    }
};


//Union Find Solution!
//Union Find solution!
//Good explanation:
//https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/discuss/197668/Count-the-Number-of-Islands-O(N)
class Solution {
private:
    //maps the UF value with its parent!
    unordered_map<int, int> UFMap;
    int numOfComponents;
    
    int find(int x){
        //If x is new, we map it to its self. 
        //Since potentially x will be a new component, we need 
        //increment numOfComponents
        if(UFMap.count(x) == 0){
            UFMap[x] = x; numOfComponents ++;
        }
        //if(node->parent != node) 
        //  node->parent = find(node->parent);
        //We update the parent pointer here, and map it to the 
        //common root
        if(x != UFMap[x]) UFMap[x] = find(UFMap[x]);
        return UFMap[x];
    }

    void UFOperation(int x, int y){
        int parentX = find(x);
        int parentY = find(y);
        //Since we know a stone joins two indexes together
        //which means we have two groups combined together! We need 
        //to decrement numOfComponents.
        if(parentX != parentY) {
            UFMap[parentX] = parentY;
            numOfComponents --;
        }
    }
    
public:
    int removeStones(vector<vector<int>>& stones) {
        //Initialize here
        numOfComponents = 0;
        for(int i = 0; i < stones.size(); ++i){
            //~stones[i][1] maps col to different index range
            //equivalent to ~x = -x - 1, or x += 10000
            //since this quetion says 0 <= stones[i][j] < 10000
            UFOperation(stones[i][0], ~stones[i][1]);
        }
        return stones.size() - numOfComponents;
    }
};


//Google interview!
//https://leetcode.com/discuss/interview-question/363081
//Rhyme Schemes problem
//Good question, the key insight here is to
//Seems like for each successive character you can either pick any character 
//that has been used already, or the next one in the sequence.

//max represent for position i, we already have how many characters to choose.
//We know that pos i can only have characters from characters already
//in our cur string, or have characters the next one in the sequence.
//For n == 2,  if we alreay have ['A'] in our string, then for pos 1, we can
//only have two possible situation, 'A' or 'A' + 1 == 'B'
void generatePoemsR(int n, int max, string& cur, vector<string>& ans) {
	for (int i = 0; i < max; i++) {
		cur.push_back('A' + i);
		// picking the next character in the sequence expands the number of available characters
		//The most tricky part is how to maintain the max
		if (n > 1) generatePoemsR(n - 1, i == max - 1 ? max + 1 : max, cur, ans);
		else ans.push_back(cur);
		cur.pop_back();
	}
}

void generatePoems(int n) {
	vector<string> ans;
	string cur;
	if (n > 0) generatePoemsR(n, 1, cur, ans);
	for (string& s : ans) {
		cout << s << endl;
	}
	
}

int main() {
	generatePoems(4);
	system("pause");
	return 0;
}


//733. Flood Fill
//https://leetcode.com/problems/flood-fill/
class Solution {
public:
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
        int m = image.size();
        int n = m ? image[0].size() : 0;
        if(!m || !n) return image;
        queue<pair<int, int>> Q;
        Q.push(make_pair(sr, sc));
        int oriColor = image[sr][sc];
        if(oriColor == newColor) return image;
        while(!Q.empty()){
            pair<int, int> pixel = Q.front();
            Q.pop();
            image[pixel.first][pixel.second] = newColor;
            if(pixel.first+1 < m && image[pixel.first+1][pixel.second] == oriColor)
                Q.push({pixel.first+1, pixel.second});
            if(pixel.first-1 >= 0 && image[pixel.first-1][pixel.second]== oriColor)
                Q.push({pixel.first-1, pixel.second});
            if(pixel.second+1 < n && image[pixel.first][pixel.second+1] == oriColor)
                Q.push({pixel.first, pixel.second + 1});
            if(pixel.second-1 >= 0 && image[pixel.first][pixel.second-1] == oriColor)
                Q.push({pixel.first, pixel.second-1});
        }
        return image;
    }
};


//778. Swim in Rising Water
//https://leetcode.com/problems/swim-in-rising-water/
//Dijkstra algorithm. We use a priority_queue to handle the case. The critical
//part is that this is not the typical dijkstra algorithm, since we did not
//record the shortest path for each node! (we can do it on the fly though, not
//necessarily for this problem!) Then when we reach the last node, we need to
//immediately return the result!
class Solution {
public:
    int swimInWater(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = m ? grid[0].size() : 0;
        if(!m || !n) return 0;
        priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> pq;
        vector<vector<int>> visited(m, vector<int>(n, 0));
        pq.push(vector<int>({grid[0][0], 0, 0}));
        visited[0][0] = 1;
        const int dir[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int res = 0;
        while(!pq.empty()){
            auto& v = pq.top();
            int indexI = v[1];
            int indexJ = v[2];
            res = max(res, v[0]);
            pq.pop();  
            //We need to return earlier since when we reach the end
            //or we can no longer guarantee we find the shortest path
            if(indexI == m-1 && indexJ == n-1) return res;
            for(int i = 0; i < 4; ++i){
                int nextI = indexI + dir[i][0];
                int nextJ = indexJ + dir[i][1];
                
                if(nextI >= 0 && nextI < m && nextJ >= 0 && nextJ < n && visited[nextI][nextJ] == 0){
                    visited[nextI][nextJ] = 1;
                    pq.push(vector<int>({grid[nextI][nextJ], nextI, nextJ}));
                }
            }
        }
        
        return res;
    }
};


//Optimized version! note we can use BFS to search all the value less than res
//earlier. Then we do not need to push these value to pq, and save some time!
class Solution {
public:
    int swimInWater(vector<vector<int>>& grid) {
        int n = grid.size();
        int m = n ? grid[0].size() : 0;
        if(!m || !n) return 0;
        int res = max(grid[0][0], grid[n-1][n-1]);
        priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> pq;
        vector<vector<int>> visited(n, vector<int>(n, 0));
        
        //const int dir[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        vector<int> dir({-1, 0, 1, 0, -1});
        visited[0][0] = 1;
        pq.push(vector<int>({grid[0][0], 0, 0}));

        while(!pq.empty()){
            auto v = pq.top();
            pq.pop();  
            res = max(res, v[0]);
            queue<pair<int, int>> myq;
            myq.push({v[1], v[2]});
            while(!myq.empty()){
                auto vec = myq.front();
                myq.pop();
                //We need to return earlier since when we reach the end
                //or we can no longer guarantee we find the shortest path
                if(vec.first == n-1 && vec.second == n-1) return res;
                for(int i = 0; i < 4; ++i){
                    int nextI = vec.first + dir[i];
                    int nextJ = vec.second + dir[i+1];
                    if(nextI >= 0 && nextI < n && nextJ >= 0 && nextJ < n && visited[nextI][nextJ] == 0){
                        visited[nextI][nextJ] = 1;
                        if(grid[nextI][nextJ] <= res){
                            myq.push({nextI, nextJ});
                        }else
                            pq.push({grid[nextI][nextJ], nextI, nextJ});
                    }
                }
            }
            
        }
        
        return -1;
    }
};


//464. Can I Win
//https://leetcode.com/problems/can-i-win/
/*
https://leetcode.com/problems/can-i-win/discuss/95320/
Good problem: Actually, an NP problem! We need to determine when player 1
pick up a number, all the possible pick of player 2. Then determine which
state can gurantee player 1 a win! 
Since maxChoosableInteger <= 20, we can use a 32bit interger to record 
whether a number has been picked up or not. 
*/
class Solution {
private:
    //We need to have memorization here to boost the efficiency
    bool helper(int M, int T, int selectK, int* memo){
        if(T <= 0)  return false; //player for this turn loses
        if(memo[selectK] != 0) return memo[selectK] == 1;
        
        //exhaustive search!
        for(int i = 0; i < M; ++i){
            //We check player 2's situation, if we select ith number,
            //and turns out that player 2 can always lose, then return 
            //true!
            if(!(selectK & (1 << i)) && !helper(M, T - (i+1), (selectK | (1 << i)), memo))
                return memo[selectK] = 1;
        }
        //current player cannot win!
        memo[selectK] = -1;
        return false;
        
    }
public:
    bool canIWin(int maxChoosableInteger, int desiredTotal) {
        //if ith bit of selectK is 1, then i is selected from 
        //maxChoosableInteger
        int selectK = 0;
        int totalSum = (maxChoosableInteger+1) * maxChoosableInteger / 2;
        if(totalSum < desiredTotal) return false;
        if(desiredTotal < 2) return true; 
        //If we have odd number choice, then player 1 gurantee a win!
        if(totalSum == desiredTotal) return maxChoosableInteger%2 == 1;
        //We need to allocate the same amount of space to store 2^M 
        //possible situation. if memo[i] == -1, player 2 can force a win
        //if memo[i] == 1, then player can force a win
        int memo[(1 << maxChoosableInteger)] = {0};
        cout << memo[2];
        return helper(maxChoosableInteger, desiredTotal, selectK, memo);
    }
};


//433. Minimum Genetic Mutation
//https://leetcode.com/problems/minimum-genetic-mutation/
//BFS
class Solution {
private:
    int diffStrs(string& s1, string& s2){
        int cnt = 0;
        for(int i = 0; i < s1.size(); ++i){
            if(s1[i] != s2[i]){
                cnt++;
                if(cnt > 1) return 0;
            }
        }
        return cnt == 1;
    }
public:
    int minMutation(string start, string end, vector<string>& bank) {
        if(bank.empty()) return -1;
        unordered_set<string> uSet(bank.begin(), bank.end());
        if(uSet.count(end) == 0) return -1;
        queue<string> Q;
        for(auto& s : bank){
            if(s == start){
                uSet.erase(s);
            }else if(diffStrs(s, start)){
                uSet.erase(s);
                Q.push(s);
            }
        }
        if(Q.empty()) return -1;
        
        int level = 0;
        while(!Q.empty()){
            int lenQ = Q.size();
            level++;
            
            for(int i = 0; i < lenQ; ++i){
                string str = Q.front();
                Q.pop();
                if(str == end) return level;
                for(auto it = uSet.begin(); it != uSet.end();){
                    string temp = *it;
                    //cout << temp << endl;
                    if(diffStrs(str, temp)){
                        Q.push(temp);
                        it = uSet.erase(it);
                    }else{
                        ++it;
                    }
                }
            }
            
        }
        return -1;
    }
};

//Bi-directional BFS
//Not implement by me!
class Solution {
public:
    int minMutation(string start, string end, vector<string>& bank) {
        unordered_set<string> dict(bank.begin(), bank.end());
        if (!dict.count(end)) return -1;
        unordered_set<string> bset, eset, *set1, *set2;
        bset.insert(start), eset.insert(end);
        int step = 0, n = start.size();
        while (!bset.empty() and !eset.empty()) {
            if (bset.size() <= eset.size())
                set1 = &bset, set2 = &eset;
            else set2 = &bset, set1 = &eset;
            unordered_set<string> tmp;
            step ++;
            for (auto itr = set1->begin(); itr != set1->end(); ++itr) {
                for (int i = 0; i < n; ++i) {
                    string dna = *itr;
                    for (auto g : string("ATGC")) {
                        dna[i] = g;
                        if (set2->count(dna)) return step;
                        if (dict.count(dna)) {
                            tmp.insert(dna);
                            dict.erase(dna);
                        }
                    }
                }
            }
            *set1 = tmp;
        }
        return -1;
    }
};


//407. Trapping Rain Water II
//https://leetcode.com/problems/trapping-rain-water-ii/
/*
//The first try, not work! Extension from 1D trapping rain water
class Solution {
public:
    int trapRainWater(vector<vector<int>>& heightMap) {
        int m = heightMap.size();
        int n = m ? heightMap[0].size() : 0;
        if(m == 0 || n == 0) return 0;
        //Save the maximum height from four directions:
        //l->r dict[1], t->b dict[0], r->l dict[3], b->t dict[2]
        int hDict[m][n][4];
        for(int i = 1; i < m-1; ++i){
            for(int j = 1; j < n-1; ++j){
                hDict[i][j][1] = hDict[i][j][1] = 0;
                hDict[i][j][3] = hDict[i][j][3] = 0;
            }
        }
        for(int i = 0; i < m; ++i){
            hDict[i][0][1] = heightMap[i][0];
            hDict[i][n-1][3] = heightMap[i][n-1];
        }
        for(int j = 0; j < n; ++j){
            hDict[0][j][0] = heightMap[0][j];
            hDict[m-1][j][2] = heightMap[m-1][j];
        }
        int res = 0;
        for(int i = 1; i < m-1; ++i){
            for(int j = 1; j < n-1; ++j){
                hDict[i][j][0] = max(hDict[i-1][j][0], heightMap[i][j]);
                hDict[i][j][1] = max(hDict[i][j-1][1], heightMap[i][j]);
            }
        }
        for(int i = m-2; i >= 1; --i){
            for(int j = n-2; j >= 1; --j){
                hDict[i][j][2] = max(hDict[i+1][j][2], heightMap[i][j]);
                hDict[i][j][3] = max(hDict[i][j+1][3], heightMap[i][j]);
                int minHeight = min(min(hDict[i][j][0], hDict[i][j][1]), min(hDict[i][j][2], hDict[i][j][3]));
                //cout << i << " " << j << " " << hDict[i][j][0] << " " << hDict[i][j][1] << " " << hDict[i][j][2] << " " << hDict[i][j][3] << " " <<minHeight << endl;
                res +=(minHeight - heightMap[i][j] > 0 ? minHeight - heightMap[i][j] : 0);
            }
        }
        
        return res;
    }
};
*/
//BFS with priority queue. Like greedy! Not easy to figure it out at first 
//glance! Once get the idea, the problem is not hard!
class Solution {
public:
    int trapRainWater(vector<vector<int>>& heightMap) {
        int m = heightMap.size();
        int n = m ? heightMap[0].size() : 0;
        if(m < 3 || n < 3) return 0;
        priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, greater<pair<int, pair<int, int>>>> pq;
        bool visited[m][n] = {0};
        for(int i = 0; i < n; ++i){
            pq.push({heightMap[0][i], {0, i}});
            visited[0][i] = 1;
            pq.push({heightMap[m-1][i], {m-1, i}});
            visited[m-1][i] = 1;
            
        }
        for(int j = 1; j < m-1; ++j){
            pq.push({heightMap[j][0], {j, 0}});
            visited[j][0] = 1;
            pq.push({heightMap[j][n-1], {j, n-1}});
            visited[j][n-1] = 1;
        }
        
        int maxH = 0;
        const int offset[] = {-1, 0, 1, 0, -1};
        int res = 0;
        while(!pq.empty()){
            auto grid = pq.top();
            pq.pop();
            int curX = grid.second.first;
            int curY = grid.second.second;
            maxH = max(maxH, grid.first);
            for(int i = 0; i < 4; ++i){
                int nextX = curX + offset[i];
                int nextY = curY + offset[i + 1];
                //only care about those unvisited
                if(nextX < 0 || nextX >= m || nextY < 0 || nextY >= n || visited[nextX][nextY])
                    continue;
                int diff = maxH - heightMap[nextX][nextY];
                if(diff > 0) res += diff;
                pq.push({heightMap[nextX][nextY], {nextX, nextY}});
                visited[nextX][nextY] = 1;
            }
            
        }
        
        return res;
    }
    
};



//909. Snakes and Ladders
//https://leetcode.com/problems/snakes-and-ladders/
class Solution {
public:
    int snakesAndLadders(vector<vector<int>>& board) {
        int m = board.size();
        int n = m ? board[0].size() : 0;
        if(!m || !n) return -1;
        if(m == 1 && n == 1) return 0;
        if(m * n <= 6) return 1;
        
        int totalLen = m * n;
        int uBoard[totalLen] = {0};
        int visited[totalLen] = {0};
        int k = 0;
        int dir = 0;
        for(int i = m-1; i >= 0; --i){
            dir = (dir == 0 ? 1 : 0);
            for(int j = 0; j < n; ++j){
                if(dir)
                    uBoard[k] = board[i][j];
                else
                    uBoard[k] = board[i][n-1-j];
                k++;
            }
        }

        queue<int> Q;
        int start = uBoard[0] > 0 ? uBoard[0]-1 : 0;
        Q.push(start);
        visited[start] = 1;
        //cancel the first move!
        int level = 0;
        while(!Q.empty()){
            int lenQ = Q.size();
            for(int size = 0; size < lenQ; ++size){
                int index = Q.front();
                Q.pop();
                if(index == totalLen-1) return level;
                
                for(int i = 1; i <= 6; ++i){
                    int nextI = index + i;
                    /*This is the only part you did not get it!*/
                    //We cannot add visited[nextI] == 0 here. Since we 
                    //potentially visited an entry which is euqals to 
                    //current nextI before, With visited[nextI] == 0 here
                    //we will potentially skip the possibility that 
                    //uBoard[nextI] is still greater than 0. (Double jump)
                    //Which loose the optimal solution!
                    //For example: [-1, 3, 9, -1, -1, -1, -1, -1, -1]
                    //If we first visit 3, then we will go to 9, and mark 
                    //it as visited. Then with the constraint, we will skip
                    //9 in the next loop. However, we need to visit it 
                    //because we can directly jump to the destination!
                    
                    //if(nextI < totalLen && visited[nextI] == 0)
                    if(nextI < totalLen){
                        int dest = uBoard[nextI] > -1 ? uBoard[nextI]-1 : nextI;
                            if(visited[dest] == 0){
                                visited[dest] = 1;
                                Q.push(dest);
                        }
                    }
                    
                }
                
            }
            level++;
        }
        return -1;
    }
};


