//310. Minimum Height Trees
//https://leetcode.com/problems/minimum-height-trees/
/* 
The general idea is to delete the leaves layer by layer. At the end of the
day, when we have only 1 or 2 nodes, then that is the result. Please note
in the problem statement, our undirected graph has tree characteristics, which
means we do not have cycles in this graph.
*/
class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        if(n == 1) return vector<int>(1, 0);
        vector<unordered_set<int>> graph(n, unordered_set<int>());
        vector<int> leaves;
        for(auto& v : edges){
            graph[v[0]].insert(v[1]);
            graph[v[1]].insert(v[0]);
        }
        for(int i = 0; i < n; ++i){
            //degree of 1, leaf node
            if(graph[i].size() == 1)
                leaves.push_back(i);
        }
        //Note our final result could either be 1 node
        //or 2 nodes
        while(n > 2){
            int nLeaves = leaves.size();
            vector<int> newLeaves;
            n -= nLeaves;
            while(!leaves.empty()){
                int leafIndex = leaves.back();
                leaves.pop_back();
                
                int leafPar = *(graph[leafIndex].begin());
                graph[leafIndex].erase(leafPar);
                graph[leafPar].erase(leafIndex);
                if(graph[leafPar].size() == 1)
                    newLeaves.push_back(leafPar);
            }
            leaves.swap(newLeaves);
        }
        return leaves;
    }
};

