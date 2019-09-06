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

/*Very good explanation:
https://leetcode.com/problems/minimum-height-trees/discuss/76052/Two-O(n)-solutions*/
/* Another implementation. The general idea is first pick up a random node and 
calculate the longest path. We find a potential end node in this longest path, 
and then do BFS again and search for the entire path. The potential root node
is either path[len/2] (odd) or {path[len/2 - 1], path[len/2]} (even)*/
//The following implementation has a bug. //Double check later
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

//399. Evaluate Division
//https://leetcode.com/problems/evaluate-division/
//Union-Find solution: Hard to get it right during the interview!
//The idea is to find the common root which stands as a standard for all
//potential values. For example, we may have 
// a / c = 2.0 | a / b = 3.0 | d / e = 4.0 | a / d = 5.0
//In this example, given the first three equations, we can build up the 
//relationship like below. 
// b(2.0/3) -> a(2.0) -> c(1.0)       (1)
// d(4.0) -> e(1.0)                   (2)
//When the fourth equation appears, then we need to map all values from 
//list (1) to list (2) (or map (2) to (1)), since we know a / d = 5.0
//Then we calculate the scaling factor (ratio) of the conversion:
//ratio = d * 5.0 / a = 4.0 * 5.0 / 2.0 = 10.0;
//Which means all the values in (1) should be multiplied by this ratio
//in order to consider e(1.0) as the common root! After multiplying
//the ratio we can merge two lists together:
// b(20.0/3) -> a(20.0) -> c(10. 0)
//                            v
//              d(4.0) -> e(1.0)
//This is the idea of the algorithm! We have some small optimization here
//Like flatten the tree when find the parent. Please pay attention to 
//these tricks!
class Solution {
private:
    struct gNode{
        double val;
        gNode* parent;
        gNode(){ parent = this; }
    };
    
    gNode* findParent(gNode* g){
        if(g->parent == g)
            return g;
        //Note we also flatten the tree by setting the current parent 
        //to be the root!
        g->parent = findParent(g->parent);
        return g->parent;
    }
    
    void unionNodes(gNode* g1, gNode* g2, unordered_map<string, gNode*>& uMap, double curValue){
        gNode* parent1 = findParent(g1);
        gNode* parent2 = findParent(g2);
        //g1 and g2 already have the same common root
        if(parent1 == parent2) return;
        double ratio = curValue * g2->val / g1->val;
        for(auto it = uMap.begin(); it != uMap.end(); ++it){
            //nodes in the first list
            if(findParent(it->second) == parent1)
                it->second->val *= ratio;
        }
        //Attach parent 1 to parent 2
        parent1->parent = parent2;
    }
    
public:
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        int len = equations.size();
        unordered_map<string, gNode*> uMap;
        //Build up the union-find tree here!
        for(int i = 0; i < len; ++i){
            string s1 = equations[i][0], s2 = equations[i][1];
            if(uMap.count(s1) == 0 && uMap.count(s2) == 0){
                uMap[s1] = new gNode();
                uMap[s2] = new gNode();
                uMap[s1]->val = values[i];
                uMap[s2]->val = 1.0;
                uMap[s1]->parent = uMap[s2];
            }else if(uMap.count(s1) == 0){
                uMap[s1] = new gNode();
                uMap[s1] -> val = values[i] * uMap[s2] -> val;
                uMap[s1] -> parent = uMap[s2];
            }else if(uMap.count(s2) == 0){
                uMap[s2] = new gNode();
                uMap[s2] -> val = uMap[s1] -> val / values[i];
                uMap[s2] -> parent = uMap[s1];
            }else
                unionNodes(uMap[s1], uMap[s2], uMap, values[i]);
        }
        
        int qLen = queries.size();
        vector<double> res(qLen, 0);
        for(int i = 0; i < qLen; ++i){
            string s1 = queries[i][0], s2 = queries[i][1];
            if(uMap.count(s1) == 0 || uMap.count(s2) == 0 || findParent(uMap[s1]) != findParent(uMap[s2]))
                res[i] = -1;
            else{
                double tempVal = uMap[s1]->val / uMap[s2]->val;
                res[i] = tempVal;
            }
        }
        return res;
        
    }
};

//Graph approach
//A very good problem for practicing graph build and graph traversal
//The graph approach is more intuitive! and easy to code.
//Note how we build the graph!
typedef unordered_map<string, unordered_map<string, double>> Graph;
class Solution {
private:
    void BuildGraph(Graph& g, vector<vector<string>>& e, vector<double>& v){
        for(int i = 0; i < e.size(); ++i){
            string s1 = e[i][0];
            string s2 = e[i][1];
            g[s1][s2] = v[i];
            g[s2][s1] = 1.0 / v[i];
        }
    }
    double getPathWeight(Graph& g, string& s, string& t, unordered_set<string>& visited){
        if(g.count(s) == 0 || g.count(t) == 0)
            return -1.0;
        if(g[s].count(t) > 0)
            return g[s][t];
        
        visited.insert(s);
        for(auto it = g[s].begin(); it != g[s].end(); ++it){
            if(visited.count(it->first) == 0){
                string tempStr = it->first;
                double tempRes = getPathWeight(g, tempStr, t, visited);
                //we cannot return -1.0 when tempRes == -1.0
                //we have to exhaustively search all the paths
                if(tempRes != -1.0)
                    return tempRes * g[s][tempStr];
            } 
        }
        return -1.0;
    }
    
public:
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        Graph eGraph;
        BuildGraph(eGraph, equations, values);
        
        int qLen = queries.size();
        vector<double> res(qLen, 0.0);
        for(int i = 0; i < qLen; ++i){
            unordered_set<string> uSet;
            res[i] = getPathWeight(eGraph, queries[i][0], queries[i][1], uSet);
        }
        return res;
    }
};


//133. Clone Graph
//https://leetcode.com/problems/clone-graph/
//Come up with the mapping is the key to success!
class Solution {
public:
    Node* cloneGraph(Node* node) {
        if(!node) return node;
        queue<Node*> Q;
        //Mapping each node we created with original node
        //It also serves as a visited set that prevent repetitively visit the 
        //same element!
        unordered_map<Node*, Node*> uMap;
        Node* head = new Node(node->val, vector<Node*>());
        Q.push(node);
        uMap[node] = head;
        while(!Q.empty()){
            Node* tempNode = Q.front();
            Q.pop();
            for(Node* neighbor : tempNode->neighbors){
                //Only if we do not have visited that node, we will create a 
                //new node
                if(uMap.count(neighbor) == 0){
                    Node* newNeighbor = new Node(neighbor->val, vector<Node*>());
                    //Build the connection!
                    uMap[neighbor] = newNeighbor;
                    Q.push(neighbor);
                }
                uMap[tempNode]->neighbors.push_back(uMap[neighbor]);
            }
        }
        return head;
    }
};

//Recursive version. I am not so confident to come up with
//this solution. Even though it's actually DFS
class Solution {
private:
    unordered_map<Node*, Node*> uMap;
public:
    Node* cloneGraph(Node* node) {
        if(!node) return node;
        //when node is visited, we will simply return uMap[node]
        if(uMap.find(node) == uMap.end()){
            Node* newNode = new Node(node->val, vector<Node*>());
            //Build the mapping here
            uMap[node] = newNode;
            for(Node* neighbor : node->neighbors){
                newNode->neighbors.push_back(cloneGraph(neighbor));
            }
        }
        return uMap[node];
    }
};



