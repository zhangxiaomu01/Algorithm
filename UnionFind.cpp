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
