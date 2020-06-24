#include<windows.h>
#include<iostream>
#include<algorithm>
#include<vector>
#include<array>
#include<cmath>
#include<random>
#include<sstream>
#include<unordered_map>
#include<numeric>
#include<iterator>
#include<unordered_set>
#include<queue>
#include<set>
#include<map>

using namespace std;

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


//149. Max Points on a Line
//https://leetcode.com/problems/max-points-on-a-line/
//Note unordered_map cannot hash std::pair, you need to define your own hash
//function if you want to use unordered_map.
//Or we can convert to string: 
//counter[to_string(dx / g) + '_' + to_string(dy / g)]++;

//This solution is actually a bruteforce solution, we cache the ratio in our
//map, and always check the maximum points after the cache.
//Using map and cache ratios is key to success
//Note we may have two points on the same location
class Solution {
private:
    int gcd(int a, int b){
        while(b != 0){
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
public:
    int maxPoints(vector<vector<int>>& points) {
        int len = points.size();
        map<pair<int, int>, int> pMap;
        int maxNumPoints = 0;
        for(int i = 0; i < len; ++i){
            pMap.clear();
            int duplicate = 1;
            //j should start with i+1. For 0..i we have already checked in the
            //previous loop
            for(int j = i + 1; j < len; ++j){
                if(points[j][0] == points[i][0] && points[j][1] == points[i][1]){
                    duplicate++;
                    continue;
                }
                
                int deltaX = points[j][0] - points[i][0];
                int deltaY = points[j][1] - points[i][1];
                int gcdDelta = gcd(deltaX, deltaY);
                pMap[{deltaX/gcdDelta, deltaY/gcdDelta}]++;
                
            }
            maxNumPoints = max(maxNumPoints, duplicate);
            for(auto it = pMap.begin(); it != pMap.end(); ++it){
                maxNumPoints = max(maxNumPoints, it->second + duplicate);
            }
        }
        return maxNumPoints;
    }
};


//207. Course Schedule
//https://leetcode.com/problems/course-schedule/
//First build the graph as ajacent list, then graph traversal!
//BFS
typedef vector<vector<int>> graph;
class Solution {
private:
    void buildGraph(int n, vector<pair<int, int>>& p, graph& g, vector<int>& s){
        int len = p.size();
        for(int i = 0; i < len; i++){
            g[p[i].second].push_back(p[i].first);
            s[p[i].first]++;
        }
    }
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        int len = prerequisites.size();
        graph G(numCourses);
        vector<int> Ideg(numCourses, 0);
        buildGraph(numCourses, prerequisites, G, Ideg);
        queue<int> Q;
        for(int i = 0; i < numCourses; i++){
            //We need to push all possible j here
            //since graph is not necessarily connected!
            if(Ideg[i] == 0){
                Q.push(i);
            }
        }
        int counter = 0;
        while(!Q.empty()){
            int node = Q.front();
            Q.pop();
            counter++;
            for(int i : G[node]){
                Ideg[i]--;
                if(Ideg[i]==0)
                    Q.push(i);
            }
        }
        return counter == numCourses;
    }
};


//DFS: Cycle detection! (Unoptimized version!)
class Solution {
private:
    //tobeVisited is a flag which keep track of the nodes we will visit
    //later. 
    bool DFS(vector<vector<int>>& G, vector<int>& tobeVisited, int node){
        //When we explore the graph, and find some node has already been
        //marked as tobeVisited, we find a cycle!
        if(tobeVisited[node]) return false;
        //backtracking, when we finish explore node, we need to reset
        //to be visited[node] to be 0
        tobeVisited[node] = 1;
        for(int n : G[node]){
            if(!DFS(G, tobeVisited, n))
                return false;
        }
        tobeVisited[node] = 0;
        return true;
    }
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        if(numCourses <= 1 || prerequisites.empty()) return true;
        vector<vector<int>> graph(numCourses, vector<int>());
        unsigned int len = prerequisites.size();
        for(int i = 0; i < len; ++i){
            graph[prerequisites[i][1]].push_back(prerequisites[i][0]);
        }
        
        vector<int> tobeVisited(numCourses, 0);
        
        for(int i = 0; i < numCourses; ++i){
            if(!DFS(graph, tobeVisited, i))
                return false;
        }
        return true;
    }
};

//DFS: Cycle detection! Optimized version. Using another array to keep track
//of already explored nodes!
class Solution {
private:
    bool DFS(vector<vector<int>>& G, vector<int>& tobeVisited, vector<int>& explored, int node){
        if(tobeVisited[node]) return false;
        //Whenever we go to an already visited node branch, we know we have
        //explored the following graph, can safely return true here!
        if(explored[node]) return true;
        tobeVisited[node] = 1;
        explored[node] = 1;
        for(int n : G[node]){
            if(!DFS(G, tobeVisited, explored, n))
                return false;
        }
        tobeVisited[node] = 0;
        return true;
    }
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        if(numCourses <= 1 || prerequisites.empty()) return true;
        vector<vector<int>> graph(numCourses, vector<int>());
        unsigned int len = prerequisites.size();
        for(int i = 0; i < len; ++i){
            graph[prerequisites[i][1]].push_back(prerequisites[i][0]);
        }
        
        vector<int> tobeVisited(numCourses, 0);
        //Add another vector to keep track of thouse already explored node
        vector<int> explored(numCourses, 0);
        //We need a loop here since the graph is not necessarily connected!
        for(int i = 0; i < numCourses; ++i){
            //if a node has been explored, we can skip the recursion!
            if(!explored[i] && !DFS(graph, tobeVisited, explored, i))
                return false;
        }
        return true;
    }
};


//210. Course Schedule II
//https://leetcode.com/problems/course-schedule-ii/
class Solution {
private:
    //It's similar to post order traversal! we need first to make sure that
    //we can have a valid leaf node, then push the leaf to our res first
    //That's the reason why in the end, we need to reverse the order of
    //res. It must be post-order
    bool helper(vector<vector<int>>& G, vector<int>& tobeVisited, vector<int>& explored, vector<int>& res, int node){
        if(tobeVisited[node]) return false;
        if(explored[node]) return true;
        explored[node] = true;
        tobeVisited[node] = true;
        
        for(int n : G[node]){
            if(!helper(G, tobeVisited, explored, res, n))
                return false;
        }
        tobeVisited[node] = false;
        //Did not get this! 
        res.push_back(node);
        return true;
    }
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> res;
        int len = prerequisites.size();
        vector<vector<int>> graph(numCourses, vector<int>());
        for(int i = 0; i < len; ++i){
            graph[prerequisites[i][1]].push_back(prerequisites[i][0]);
        }
        
        vector<int> tobeVisited(numCourses, 0);
        vector<int> explored(numCourses, 0);
        for(int i = 0; i < numCourses; ++i){
            //if we find a cycle in the graph
            if(!explored[i] && !helper(graph, tobeVisited, explored, res, i))
                return vector<int>();
        }
        
        reverse(res.begin(), res.end());
        return res;        
    }
};

//BFS solution, be careful about when we push element to the queue!
class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> res;
        int len = prerequisites.size();
        vector<int> inDeg(numCourses, 0);
        vector<vector<int>> graph(numCourses, vector<int>());
        for(int i = 0; i < len; ++i){
            graph[prerequisites[i][1]].push_back(prerequisites[i][0]);
            inDeg[prerequisites[i][0]]++;
        }
        
        queue<int> Q;
        vector<int> visited(numCourses, 0);
        for(int i = 0; i < numCourses; ++i){
            if(inDeg[i] == 0){
                Q.push(i);
                res.push_back(i);
            }    
        }

        while(!Q.empty()){
            int index = Q.front();
            Q.pop();
            visited[index] = 1;
            for(int i = 0; i < graph[index].size(); ++i){
                int neighbor = graph[index][i];
                if(!visited[neighbor]){
                    --inDeg[neighbor];
                    //only push back when incoming degree becomes 0
                    if(inDeg[neighbor] == 0){
                        res.push_back(neighbor);
                        Q.push(neighbor);
                    }
                        
                }
            }
        }
        
        //Only if we include all the nodes, we return res
        if(res.size() == numCourses)
            return res;
        
        return vector<int>();        
    }
};

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


//Google | Onsite | Min Diff Between Given Path and Path in the Graph
//https://leetcode.com/discuss/interview-question/378687/
//Very good problem for practicing Dijkstra algorithm. 


//1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance
//https://leetcode.com/contest/weekly-contest-173/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/
//Problem from weekly contest 173th, you did not get it right in the contest because you are not 
//familiar with Dijkstra any more. You almost get it right though.
class Solution {
    int numOfNeighbouring(vector<vector<pair<int, int>>>& G, int city, int dT){
        auto myComp = [](pair<int, int> p1, pair<int, int> p2){
            return p1.second < p2.second;  
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(myComp)> Q(myComp);
        int len = G.size();
        int res = 0;
        vector<int> visited(len, 0);
        Q.push({city, dT});
        
        while(!Q.empty()){
            pair<int, int> C = Q.top();
            Q.pop();
            
            if(visited[C.first] == 1) continue;
            else{
                visited[C.first] = 1;
                res++;
            }
            for(int j = 0; j < G[C.first].size(); ++j){
                //Note we should not put visited[G[C.first][j].first] = 1 here, because potentially we still need to revisit
                //this node. We defer the judgement to the beginning of the next while loop
                //You get this wrong during the contest
                if(visited[G[C.first][j].first] == 0 && G[C.first][j].second <= C.second ){
                    Q.push({G[C.first][j].first, C.second - G[C.first][j].second});
                    //visited[G[C.first][j].first] = 1;
                    //res++;
                }
            }
        }
        return res;
    }
    
public:
    int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold) {
        vector<vector<pair<int, int>>> adjGraph(n);
        for(int i = 0; i < edges.size(); ++i){
            adjGraph[edges[i][0]].push_back({edges[i][1], edges[i][2]});
            adjGraph[edges[i][1]].push_back({edges[i][0], edges[i][2]});
        }
        int toTal = n-1;
        int res = n-1;
        
        for(int i = n-1; i >= 0; --i){
            int k = numOfNeighbouring(adjGraph, i, distanceThreshold);
            if(k == 0) return i;
            if(toTal > k){
                toTal = k;
                res = i;
            } 
        }
        
        return res;
    }
};


//A celever solution you know, but you forget how to implement Floyd-Warshall algorithm.
//The following code is a variant of Floyd-Warshall algorithm
class Solution {
public:
    int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold) {
        vector<int>ranks(n,0);
        for(int i=0;i<n;++i){
            vector<int>costs(n,1e7);
            costs[i]=0;
            for(int j=0;j<n-1;++j){
                for(auto&e:edges){
                    int u = e[0], v = e[1], w = e[2];
                    costs[v]=min(costs[v],costs[u]+w);
                    costs[u]=min(costs[u],costs[v]+w);
                }
            }
            for(int c:costs)    ranks[i] += c<=distanceThreshold;
        }
        int r = *min_element(begin(ranks),end(ranks));
        for(int i=n-1;i>-1;--i){
            if(ranks[i]==r) return i;
        }
        return 0;
    }
};



//1368. Minimum Cost to Make at Least One Valid Path in a Grid
//https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/
//You did not make it in the contest! It should not be a hard problem!
//Classic Graph problem, note we can consider each grid as a node in our graph,
//with at most 4 edges connect to other grid, if the sign pointing to another node,
//then that edge will have weight 0, else, we have edge weight 1. We want to know, 
//what will be the optimal solution if we want to move from (0, 0) to (m-1, n-1).
//We need to greedily explore all the nodes with the smallest cost then 1 cost higher
//We can do it either by BFS or DFS.
//BFS implementation!
class Solution {
private:
    bool isValid(int x, int y, int m, int n){
        return x >= 0 && x < m && y >= 0 && y < n;
    }
    
public:
    int minCost(vector<vector<int>>& grid) {
        const int m = grid.size();
        const int n = m ? grid[0].size() : 0;
        
        // vector<0> - x
        // vector<1> - y
        // vector<2> - cost
        deque<vector<int>> dQ;
        
        int nRow[4] = {0, 0, 1, -1};
        int nCol[4] = {1, -1, 0, 0};
        //Record current visited nodes
        int visited[m][n];
        
        //I am not so sure why
        //int visted[m][n] = {0} cannot initialize all 2D array with 0 any more
        for(int i = 0; i < m; ++i){
            for(int j = 0; j < n; ++j)
                visited[i][j] = 0;
        }

        dQ.push_back({0, 0, 0});
        
        
        while (!dQ.empty()){
            auto it = dQ.front();
            dQ.pop_front();
            visited[it[0]][it[1]] = 1;

            if (it[0] == m-1 && it[1] == n-1) return it[2];
            
            int dir = grid[it[0]][it[1]] - 1;
            
            for (int i = 0; i < 4; ++i){
                int nX = it[0] + nRow[i];
                int nY = it[1] + nCol[i];
                
                //Valid and not visited!
                if (isValid(nX, nY, m, n) && !visited[nX][nY]){
                    //We cannot set visited[nX][nY] to be 1 here because we will potentially
                    //lose some possible conditions!
                    //visited[nX][nY] = 1;
                    if (nX == it[0] + nRow[dir] && nY == it[1] + nCol[dir]){
                        // Note we push the smaller weight to the front of the queue
                        dQ.push_front({nX, nY, it[2]}); 
                    }    
                    else{
                        // And we push the larger weight to the back of the queue
                        // which ensure us to always examine the cost of 0 first, then cost 
                        // of 1, 2, 3, ... k.
                        dQ.push_back({nX, nY, it[2] + 1});
                    }
                        
                }
            
            }
        }
        return -1;
        
    }
};


//DFS implementation! Much trickier!
//With early termination, this approach is a little bit faster!
class Solution {
private:
    int dir[4][2] = {{0, 1}, {0, -1}, {1, 0}, { -1, 0}};
    
    void dfs(vector<vector<int>>& dp, int curX, int curY, queue<vector<int>>& Q, int cost, vector<vector<int>>& grid){
        int m = dp.size();
        int n = dp[0].size();
        //note for dp[x][y], we always update the optimal solution. Because our cost is built from 0
        if(curX < 0 || curX >= m || curY < 0 || curY >= n || dp[curX][curY] != INT_MAX) return;
        dp[curX][curY] = cost;
        Q.push({curX, curY});
        int nextDir = grid[curX][curY] - 1;
        dfs(dp, curX + dir[nextDir][0], curY + dir[nextDir][1], Q, cost, grid);
    }
    
public:
    int minCost(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        int cost = 0;
        
        vector<vector<int>> dp(m, vector<int>(n, INT_MAX));
        
        //0 - x, 1 - y
        queue<vector<int>> Q;
        //We need to call dfs first to get all the nodes with cost 0 to get there
        dfs(dp, 0, 0, Q, cost, grid);
        
        while(!Q.empty()){
            int lenQ = Q.size();
            cost ++;
            
            for(int i = 0; i < lenQ; ++i){
                auto v = Q.front();
                Q.pop();
                //Early termination!
                if(dp[m-1][n-1] != INT_MAX) return dp[m-1][n-1];
                
                for(int j = 0; j < 4; ++j){
                    int nextX = v[0] + dir[j][0];
                    int nextY = v[1] + dir[j][1];
                    //We already increment cost right now!
                    dfs(dp, nextX, nextY, Q, cost, grid);
                }

            }
        }
        
        return dp[m-1][n-1];
        
    }
};


//1377. Frog Position After T Seconds
//https://leetcode.com/problems/frog-position-after-t-seconds/
//You did not make it during contest 179th 
//This clever solution is from: 
//https://leetcode.com/problems/frog-position-after-t-seconds/discuss/532571/JavaC%2B%2BPython-DFS
class Solution {
private:
    double dfs(int t, int target, int cur, vector<vector<int>>& G, vector<int> visited){
        if(t < 0) return 0.0;
        // We found target that we return maximum possibility
        //This condition is very tricky! You tried several times but failed!
        //Test case: 3 [[2,1],[3,2]] 1 2
        //if(cur == target && t >= 0) return 1.0;
        //else if(cur != 1 && G[cur].size() == 1 && t >= 0) return 0.0;
        
        //Handle leaf node or time is over
        //Note we need to return cur == target, make sure that we can find this value.
        //This is neat and clever!
        if(cur != 1 && G[cur].size() == 1 || t == 0)
            return cur == target;
        
        double res = 0.0;
        visited[cur] = 1;
        for(int i = 0; i < G[cur].size(); ++i){
            if(visited[G[cur][i]] == 0)
                res += dfs(t-1, target, G[cur][i], G, visited);
        }

        // For node aside from 1
        return res * (1.0 / double(int(G[cur].size()) - int(cur != 1)));
    }

public:
    double frogPosition(int n, vector<vector<int>>& edges, int t, int target) {
        //Note this is wrong, the frog cannot stop jump if he can
        //Test case: 4 [[2,1],[3,2],[4,1]] 4 1
        //if(target == 1) return 1.0;
        
        if(n == 1) return 1.0;
        vector<vector<int>> tree(n+1);
        for(int i = 0; i < edges.size(); ++i){
            int from = edges[i][0];
            int to = edges[i][1];
            tree[from].push_back(to);
            tree[to].push_back(from);
        }
        
        vector<int> visited(n+1, 0);
        return dfs(t, target, 1, tree, visited);        
    }
};


// 1443. Minimum Time to Collect All Apples in a Tree
// https://leetcode.com/contest/weekly-contest-188/problems/minimum-time-to-collect-all-apples-in-a-tree/
// Post order traversal. My implementation.
class Solution {
    // post order traversal!
    int dfs(vector<vector<int>>& G, int cur, vector<int>& visited, vector<bool>& hasApple){
        if(G[cur].size() == 1 && visited[G[cur][0]] == 1){
            return hasApple[cur] ? 2 : 0;
        }
        
        int res = 0;
        for(int i = 0; i < G[cur].size(); ++i){
            if(visited[G[cur][i]] == 0){
                visited[G[cur][i]] = 1;
                res += dfs(G, G[cur][i], visited, hasApple);
            }
        }
        if(cur == 0) return res; 
        return res == 0 ? (hasApple[cur] ? 2 : 0) : res + 2;
    }
    
public:
    int minTime(int n, vector<vector<int>>& edges, vector<bool>& hasApple) {
        vector<vector<int>> G(n);
        vector<int> visited(n, 0);
        for(int i = 0; i < edges.size(); ++i){
            G[edges[i][0]].push_back(edges[i][1]);
            G[edges[i][1]].push_back(edges[i][0]);
        }
        visited[0] = 1;
        return dfs(G, 0, visited, hasApple);
    }
};


// 1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree
// https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
// A very brilliant way to implement prime's algorithm!
// This great solution is from:
// https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/discuss/698182/C%2B%2B-DijkstraPrim's-Algorithm-O(EV2)
// I did not write this myself! It's easy to understand, but lengthy and maybe tricky to implement
// O(EV^2) It's slow
struct Edge{
    int idx, to, cost;
    Edge(int idx, int to, int cost) : idx(idx), to(to), cost(cost) {};
	// Operator is inverted to make priority_queue a min heap
    bool operator < (const Edge& other) const { return cost > other.cost; }
};

class Solution {
public:
    vector<vector<int>> findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges) {
        vector<vector<Edge>> adj(n);
        for(int i = 0; i < edges.size(); i++){
            adj[edges[i][0]].emplace_back(i, edges[i][1], edges[i][2]);
            adj[edges[i][1]].emplace_back(i, edges[i][0], edges[i][2]);
        }
        int mn = prim(adj, -1, 0, -1); // Cost of MST on original graph
        vector<vector<int>> ans(2);
        for(int i = 0; i < edges.size(); i++){
			// Does force-taking this edge increase the MST cost? -> neither critical nor pseudo-critical
            if(prim(adj, i, edges[i][0], -1) > mn) continue;
			// Does ignoring this edge increase the MST cost? -> critical if true, pseudo-critical if false
            if(prim(adj, -1, 0, i) > mn) ans[0].push_back(i);
            else ans[1].push_back(i);
        }
        return ans;
    }
    
    int prim(vector<vector<Edge>>& adj, int startEdgeIdx, int startEdgeFrom, int ignoreEdgeIdx){
        int cnt = 0, total = 0;
        vector<bool> visited(adj.size());
        priority_queue<Edge> pq;
        pq.emplace(-1, startEdgeFrom, 0);
        while(pq.size()){
            auto [idx, to, cost] = pq.top();
            pq.pop();
            if(visited[to]) continue;
            visited[to] = true;
            cnt++;
            total += cost;
            if(cnt == adj.size()) break;
            for(auto edge : adj[to]){
                if(visited[edge.to] || edge.idx == ignoreEdgeIdx) continue;
				// If it's the start edge we assigned, we take the edge upfront and give it the highest priority by making it free
                if(edge.idx == startEdgeIdx){
                    total += edge.cost;
                    edge.cost = 0;
                }
                pq.push(edge);
            }
        }
        return cnt == adj.size() ? total : INT_MAX;
    }
};

// UF + enumerate edges
// A different way to implement MST
// https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/discuss/697761/C%2B%2B-Solution-enumerating-edges-with-explanation
// Much faster
class UnionFind {
public:
    UnionFind(int n) {
        rank = vector<int>(n, 1);
        f.resize(n);
        for (int i = 0; i < n; ++i) f[i] = i;
    }
    
    int Find(int x) {
        if (x == f[x]) return x;
        else return f[x] = Find(f[x]);
    }
    
    void Union(int x, int y) {
        int fx = Find(x), fy = Find(y);
        if (fx == fy) return;
        if (rank[fx] > rank[fy]) swap(fx, fy);
        f[fx] = fy;
        if (rank[fx] == rank[fy]) rank[fy]++;
    }
    
private:
    vector<int> f, rank;
};

class Solution {
public:
    vector<vector<int>> findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges) {
        for (int i = 0; i < edges.size(); ++i) {
            edges[i].push_back(i);
        }
        sort(edges.begin(), edges.end(), [](const vector<int>& a, const vector<int>& b) {
            return a[2] < b[2];
        });
        int origin_mst = GetMST(n, edges, -1);
        vector<int> critical, non_critical;
        for (int i = 0; i < edges.size(); ++i) {
            if (origin_mst < GetMST(n, edges, i)) {
                critical.push_back(edges[i][3]);
            } else if (origin_mst == GetMST(n, edges, -1, i)) {
                non_critical.push_back(edges[i][3]);
            }
        }
        return {critical, non_critical};
    }
    
private:
    int GetMST(const int n, const vector<vector<int>>& edges, int blockedge, int pre_edge = -1) {
        UnionFind uf(n);
        int weight = 0;
        if (pre_edge != -1) {
            weight += edges[pre_edge][2];
            uf.Union(edges[pre_edge][0], edges[pre_edge][1]);
        }
        for (int i = 0; i < edges.size(); ++i) {
            if (i == blockedge) continue;
            const auto& edge = edges[i];
            if (uf.Find(edge[0]) == uf.Find(edge[1])) continue;
            uf.Union(edge[0], edge[1]);
            weight += edge[2];
        }
        for (int i = 0; i < n; ++i) {
            if (uf.Find(i) != uf.Find(0)) return 1e9+7;
        }
        return weight;
    }
};

