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

