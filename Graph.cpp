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

/* Another implementation. The general idea is first pick up a random node and 
calculate the longest path. We find a potential end node in this longest path, 
and then do BFS again and search for the entire path. The potential root node
is either path[len/2] (odd) or {path[len/2 - 1], path[len/2]} (even)*/
class Solution {
private:
    void BFS(vector<unordered_set<int>>& G, vector<int>& pre, vector<int>& depth, int index){
        queue<int> Q;
        pre[index] = -1;
        depth[index] = 0;
        vector<int> visited(G.size(), 0);
        
        Q.push(index);
        while(!Q.empty()){
            int lenQ = Q.size();
            for(int i = 0; i < lenQ; ++i){
                int node = Q.front();
                Q.pop();
                for(auto it = G[node].begin(); it != G[node].end(); ++it){
                    if(visited[*it] != 1){
                        visited[node] = 1;
                        Q.push(*it);
                        pre[*it] = node;
                        depth[*it] = depth[node]+1;
                    }    
                }
            }
        }
    }
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        if(n == 1) return vector<int>(1, 0);
        vector<unordered_set<int>> graph(n, unordered_set<int>());
        vector<int> res;
        for(auto& v : edges){
            graph[v[0]].insert(v[1]);
            graph[v[1]].insert(v[0]);
        }
        vector<int> pre(n, 0);
        vector<int> depth(n, 0);
        
        BFS(graph, pre, depth, 0);
        int index = 0;
        for(int i = 0; i < n; ++i){
            if(depth[i] > depth[index])
                index = i;
            depth[i] = 0;
            pre[i] = 0;
        }
        BFS(graph, pre, depth, index);
        //reset index
        index = 0;
        for(int i = 0; i < n; ++i){
            if(depth[i] > depth[index])
                index = i;
        }
        vector<int> longestPath;
        //cout << index << endl;
        while(index != -1){
            longestPath.push_back(index);
            index = pre[index];
        }
        for(int n : longestPath)
            cout << n << " " << endl;
        
        int len = longestPath.size();
        if(len % 2 == 1) 
            res.push_back(longestPath[len/2]);
        else{
            res.push_back(longestPath[len/2-1]);
            res.push_back(longestPath[len/2]);
        }
        
        return res;
    }
};

