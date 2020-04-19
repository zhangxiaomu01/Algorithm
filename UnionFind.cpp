//305. Number of Islands II
//https://leetcode.com/problems/number-of-islands-ii/
struct UFNode{
    UFNode* parent;
    int rank;
    UFNode(){
        rank = 0;
        parent = this;
    }
};


class Solution {
private:
    unordered_set<UFNode*> UFMap;
    int numOfIsland;
    
    UFNode* unionFind(UFNode* g1){
        //base case
        if(g1 == g1->parent) return g1;
        
        if(g1->parent != g1){
            g1->parent = unionFind(g1->parent);
        }
        return g1->parent;
    }

    void unionOP(UFNode* g1, UFNode* g2){
        UFNode* parentG1 = unionFind(g1);
        UFNode* parentG2 = unionFind(g2);
        
        if(parentG1 != parentG2) {
            if(parentG1->rank < parentG2->rank){
                parentG1->parent = parentG2;
                
            }else if(parentG1->rank > parentG2->rank){
                parentG2->parent = parentG1;
               
            }else{
                parentG1->parent = parentG2;
                parentG2->rank++;
               
            }
        }
    }

public:
    vector<int> numIslands2(int m, int n, vector<vector<int>>& positions) {
        if(!m || !n || positions.empty()) return vector<int>();
        int numGrids = m * n;
        numOfIsland = 0;
        const int offset[] = {-1, 0, 1, 0, -1};
        unordered_map<int, UFNode*> uMap;
        vector<int> res;
        
        int lenP = positions.size();
        
        for(int i = 0; i < lenP; ++i){
            int key = positions[i][0] * n + positions[i][1];
            
            if(uMap.count(key) > 0) {
                res.push_back(res.back());
                continue;
            }
            
            if(uMap.count(key) == 0){
                uMap[key] = new UFNode();
                if(i == 0) {
                    res.push_back(1);
                    continue;
                }
                
            }
            
            int curIsland = res.back();
            
            for(int j = 0; j < 4; ++ j){
                int indexI = positions[i][0]+offset[j];
                int indexJ = positions[i][1]+offset[j+1];
                if(indexI < 0 || indexI >= m || indexJ < 0 || indexJ >= n) continue;
                
                //cout << positions[indexI][0] << " " << positions[indexJ][1] << endl;
                
                int neighbour = indexI * n + indexJ;
                if(uMap.count(neighbour) > 0){
                    if(unionFind(uMap[key]) != unionFind(uMap[neighbour])){
                        unionOP(uMap[key], uMap[neighbour]);
                        curIsland--;
                    }
                }
            }
            curIsland += 1;
            res.push_back(curIsland);
        }
        return res;
        
    }
};


//261. Graph Valid Tree
//https://leetcode.com/problems/graph-valid-tree/
//Union Find approach: implemented by me!
class Solution {
private:
    int totalComp;
    bool isTree;
    
    int FindOp(int* UF, int n, int x){
        if(UF[x] == x) return UF[x];
        
        int curP = UF[x];
        UF[x] = FindOp(UF, n, curP);
        
        return UF[x];
    }
    void UnionOp(int* UF, int n, int x, int y){
        int pX = FindOp(UF, n, x);
        int pY = FindOp(UF, n, y);
        
        //Make UF[pX] = pY not UF[x] here
        //We need to guarantee that the parent of pX points to pY
        if(pX != pY) {
            UF[pX] = pY;
            totalComp--;
        }
        
    }
public:
    bool validTree(int n, vector<vector<int>>& edges) {
        if(n <= 0) return false;
        totalComp = n;
        isTree = true;
        
        int UF[n];
        for(int i = 0; i < n; ++i){
            UF[i] = i;
        }
        
        for(auto& e : edges){
            int start = e[0];
            int end = e[1];
            if(FindOp(UF, n, start) != FindOp(UF, n, end))
                UnionOp(UF, n, start, end);
            else // We have a cycle here
                isTree = false;
        }
        //This two conditions must be satified both
        //One for cycle detection, one for one component
        return isTree && totalComp == 1;
    }
};


//323. Number of Connected Components in an Undirected Graph
//https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/
class Solution {
private:
    int numOfComp;
    int FindOP(int* UF, int n, int x){
        if(UF[x] == x) return x;
        
        int curP = UF[x];
        UF[x] = FindOP(UF, n, curP);
        return UF[x];
    }
    void UnionOP(int* UF, int n, int x, int y){
        int pX = FindOP(UF, n, x);
        int pY = FindOP(UF, n, y);
        
        if(pX != pY) {
            UF[pX] = pY;
            numOfComp--;
        }
    }
public:
    int countComponents(int n, vector<vector<int>>& edges) {
        if(n <= 0) return 0;
        int UF[n];
        numOfComp = n;
        for(int i = 0; i < n; ++i){
            UF[i] = i;
        }
        for(auto& e : edges){
            int start = e[0];
            int end = e[1];
            if(FindOP(UF, n, start) != FindOP(UF, n, end)){
                UnionOP(UF, n, start, end);
            }
        }
        return numOfComp;
    }
};


// 1391. Check if There is a Valid Path in a Grid
// https://leetcode.com/problems/check-if-there-is-a-valid-path-in-a-grid/
// Union Find approach 
// This is also from Lee:
// https://leetcode.com/problems/check-if-there-is-a-valid-path-in-a-grid/discuss/547229/Python-Union-Find
/*
The general idea is to put connected component to the same group, and in the end, we can check
whether the first grid and the last grid have the same parent node

The center of A[0][0] has coordinates [0, 0]
The center of A[i][j] has coordinates [2i, 2j]
The top edge of A[i][j] has coordinates [2i-1, 2j]
The left edge of A[i][j] has coordinates [2i, 2j-1]
The bottom edge of A[i][j] has coordinates [2i+1, 2j]
The right edge of A[i][j] has coordinates [2i, 2j+1]

Then we apply Union Find:
if A[i][j] in [2, 5, 6]: connect center and top
if A[i][j] in [1, 3, 5]: connect center and left
if A[i][j] in [2, 3, 4]: connect center and bottom
if A[i][j] in [1, 4, 6]: connect center and right

Not very efficient though..
*/
class Solution {
private:
    vector<vector<pair<int, int>>> G;
    pair<int, int> find(int x, int y){
        if(G[x][y] == pair<int, int>({x, y})) return {x, y};
        return G[x][y] = find(G[x][y].first, G[x][y].second);
    }
    
    void merge(int x, int y, int dx, int dy){
        auto p1 = find(x, y);
        auto p2 = find(x + dx, y + dy);
        //Include {x+dx, y+dy} to the {x, y}'s current parent node  
        if(p1 != p2) G[p1.first][p1.second] = p2;
    }
    
    
public:
    bool hasValidPath(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        G = vector<vector<pair<int, int>>>(2*m + 2, vector<pair<int, int>>(2*n+2));
        for(int i = 0; i < 2*m+2; ++i){
            for(int j = 0; j < 2*n+2; ++j){
                G[i][j] = {i, j};
            }
        }        
        
        //We meed to start from 1 ... m. Or merge will be overflow.
        for(int i = 1; i <= m; ++i){
            for(int j = 1; j <= n; ++j){
                int cur = grid[i-1][j-1];
                // This should be a bunch of if instead of if..else
                // We have to handle cross situation
                if (cur == 2 || cur == 5 || cur == 6) merge(i * 2, j * 2, -1, 0);
                if (cur == 1 || cur == 3 || cur == 5) merge(i * 2, j * 2, 0, -1);
                if (cur == 2 || cur == 3 || cur == 4) merge(i * 2, j * 2, 1, 0);
                if (cur == 1 || cur == 4 || cur == 6) merge(i * 2, j * 2, 0, 1);
            }
        }
        return find(G[2][2].first, G[2][2].second) == find(G[2*m][2*n].first, G[2*m][2*n].second);
    }
};

// We can also have DFS solution, it's much more efficient. 
// The implementation part is more trickier!
class Solution {
public:
	bool help(int i,int j,int n,int m,vector<vector<int>> &dp,vector<vector<int>>& grid,char last)
	{
		if(i<0||i>=n||j<0||j>=m||dp[i][j])return false;//handling the corner cases, dp represents if a cell is visited or not
		if(last=='l'&&(grid[i][j]==2||grid[i][j]==3||grid[i][j]==5))return false;
		else if(last=='d'&&(grid[i][j]==1||grid[i][j]==3||grid[i][j]==4))return false;
		else if(last=='r'&&(grid[i][j]==2||grid[i][j]==4||grid[i][j]==6))return false;
		else if(last=='u'&&(grid[i][j]==1||grid[i][j]==5||grid[i][j]==6))return false;
		//character last will tell us the direction from where we are reaching the current cell
		//e.g, if last=='l' it means we are reching the current cell from the left cell
		//above 4 conditions are checking whether it is possible to reach the current cell from the last cell or not
		//e.g, if last=='l' and current cell is street 1 or 4 or 6 then only we can continue
		else if(i==n-1&&j==m-1)return true;//we have reached the last cell and can return true
		dp[i][j]=1;//marking current cell as visited
		switch(grid[i][j])
		{
			case 1:return help(i,j-1,n,m,dp,grid,'l')||help(i,j+1,n,m,dp,grid,'r');
			case 2:return help(i-1,j,n,m,dp,grid,'u')||help(i+1,j,n,m,dp,grid,'d');
			case 3:return help(i,j-1,n,m,dp,grid,'l')||help(i+1,j,n,m,dp,grid,'d');
			case 4:return help(i,j+1,n,m,dp,grid,'r')||help(i+1,j,n,m,dp,grid,'d');
			case 5:return help(i,j-1,n,m,dp,grid,'l')||help(i-1,j,n,m,dp,grid,'u');
			case 6:return help(i,j+1,n,m,dp,grid,'r')||help(i-1,j,n,m,dp,grid,'u');
			default:return 0;
		}
		//there are always 2 possible ways to go from the current cell to any other cell which are being handled by above cases
		//e.g, if we are currently on street 1 we can either go to left i.e, j-1 or to right i.e, j+1
	}
	bool hasValidPath(vector<vector<int>>& grid) 
	{
		int n=grid.size(),m=grid[0].size();
		vector<vector<int>> dp(n,vector<int> (m,0));//mark a cell visited if we have visited the cell so as to avoid infinite loop
		char last;
		return help(0,0,n,m,dp,grid,last);
	}
};
