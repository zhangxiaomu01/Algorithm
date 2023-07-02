/**
 * @file Graph.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2022-06-18
 * 
 * A quick second pass with the common 'Graph' algorithm problems.
 * 
 */

 /*
    463. Island Perimeter
    https://leetcode.com/problems/island-perimeter/
    You are given row x col grid representing a map where grid[i][j] = 1 represents land and 
    grid[i][j] = 0 represents water.

    Grid cells are connected horizontally/vertically (not diagonally). The grid is completely 
    surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

    The island doesn't have "lakes", meaning the water inside isn't connected to the water around 
    the island. One cell is a square with side length 1. The grid is rectangular, width and height 
    don't exceed 100. Determine the perimeter of the island.

    

    Example 1:
    Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
    Output: 16
    Explanation: The perimeter is the 16 yellow stripes in the image above.

    Example 2:
    Input: grid = [[1]]
    Output: 4

    Example 3:
    Input: grid = [[1,0]]
    Output: 4
    

    Constraints:
    row == grid.length
    col == grid[i].length
    1 <= row, col <= 100
    grid[i][j] is 0 or 1.
    There is exactly one island in grid.
 */
// Solution 1: BFS
class Solution {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int res = 0;
        queue<pair<int, int>> Q;
        vector<vector<int>> visited(grid.size(), vector<int>(grid[0].size(), 0));
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[0].size(); ++ j) {
                if (grid[i][j] == 1) {
                    Q.push({i, j});
                    grid[i][j] = 2;
                    while (!Q.empty()) {
                        pair<int, int> island = Q.front();
                        int p = island.first, q = island.second;
                        Q.pop();
                        if (p - 1 < 0 || grid[p - 1][q] == 0) res ++;
                        if (p + 1 >= grid.size() || grid[p+1][q] == 0) res++;
                        if (q - 1 < 0 || grid[p][q-1] == 0) res ++;
                        if (q + 1 >= grid[0].size() || grid[p][q+1] == 0) res ++;

                        if (p - 1 >= 0 && grid[p-1][q] == 1) { 
                            Q.push({p-1, q});
                            // Needs to flip the value here to avoid duplicate visit.
                            grid[p-1][q] = 2;
                        }
                        if (p + 1 < grid.size() && grid[p+1][q] == 1) {
                            Q.push({p+1, q});
                            grid[p+1][q] = 2;
                        }
                        if (q - 1 >= 0 && grid[p][q-1] == 1) {
                            Q.push({p, q-1});
                            grid[p][q-1] = 2;
                        }
                        if (q + 1 < grid[0].size() && grid[p][q+1] == 1) {
                            Q.push({p, q+1});
                            grid[p][q+1] = 2;
                        }
                    }
                }
            }
        }
        return res;
    }
};

// Solution 2: get rid of the BFS. This problem does not require BFS.
class Solution {
public:
    int direction[4][2] = {0, 1, 1, 0, -1, 0, 0, -1};
    int islandPerimeter(vector<vector<int>>& grid) {
        int result = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 1) {
                    for (int k = 0; k < 4; k++) {       // 上下左右四个方向
                        int x = i + direction[k][0];
                        int y = j + direction[k][1];    // 计算周边坐标x,y
                        if (x < 0                       // i在边界上
                                || x >= grid.size()     // i在边界上
                                || y < 0                // j在边界上
                                || y >= grid[0].size()  // j在边界上
                                || grid[x][y] == 0) {   // x,y位置是水域
                            result++;
                        }
                    }
                }
            }
        }
        return result;
    }
};

// Count the maximum islands, then substract the repeated edges.
class Solution {
public://1035
    int islandPerimeter(vector<vector<int>>& grid) {
        if(grid.empty()) return 0;
        int count = 0, repeat = 0;
        for(int i = 0; i < grid.size(); i++)
            for(int j = 0; j < grid[0].size(); j++){
                if(grid[i][j] == 1){
                    count++;
                    // Only need to count two sides to avoid duplicate calculation.
                    if(i && grid[i - 1][j] == 1)  repeat++;
                    if(j && grid[i][j - 1] == 1)  repeat++;
                }
            }
        return 4 * count - 2 * repeat;
    }
};

 /*
    841. Keys and Rooms
    https://leetcode.com/problems/keys-and-rooms/
    There are n rooms labeled from 0 to n - 1 and all the rooms are locked except for room 0. 
    Your goal is to visit all the rooms. However, you cannot enter a locked room without having 
    its key.

    When you visit a room, you may find a set of distinct keys in it. Each key has a number on it, 
    denoting which room it unlocks, and you can take all of them with you to unlock the other rooms.

    Given an array rooms where rooms[i] is the set of keys that you can obtain if you visited 
    room i, return true if you can visit all the rooms, or false otherwise.

    

    Example 1:
    Input: rooms = [[1],[2],[3],[]]
    Output: true
    Explanation: 
    We visit room 0 and pick up key 1.
    We then visit room 1 and pick up key 2.
    We then visit room 2 and pick up key 3.
    We then visit room 3.
    Since we were able to visit every room, we return true.

    Example 2:
    Input: rooms = [[1,3],[3,0,1],[2],[0]]
    Output: false
    Explanation: We can not enter room number 2 since the only key that unlocks it is in that room.
    

    Constraints:

    n == rooms.length
    2 <= n <= 1000
    0 <= rooms[i].length <= 1000
    1 <= sum(rooms[i].length) <= 3000
    0 <= rooms[i][j] < n
    All the values of rooms[i] are unique.
 */
// BFS
class Solution {
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        int n = rooms.size();
        queue<int> Q;
        vector<int> visited(n, 0);
        Q.push(0);
        int count = 0;
        visited[0] = 1;
        while (!Q.empty()) {
            int room = Q.front();
            Q.pop();
            count++;
            if (count == n) return true;
            for (int i = 0; i < rooms[room].size(); ++i) {
                if (visited[rooms[room][i]] == 1) continue;
                Q.push(rooms[room][i]);
                visited[rooms[room][i]] = 1;
            }
        }
        return count == n;
    }
};

// DFS
class Solution {
private:
    int count = 0;
    bool dfs(vector<vector<int>>& rooms, vector<bool>& visited, int currentRoom) {
        if (visited[currentRoom]) return false;
        
        visited[currentRoom] = true;
        // We need to count all rooms that we can visit.
        count ++;
        if (count == rooms.size()) return true;
        for(int i = 0; i < rooms[currentRoom].size(); ++i) {
            if (dfs(rooms, visited, rooms[currentRoom][i])) return true;
        }
        return false;
    }
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        vector<bool> visited(rooms.size(), false);
        return dfs(rooms, visited, 0);
    }
};

 /*
    127. Word Ladder
    https://leetcode.com/problems/word-ladder/
    A transformation sequence from word beginWord to word endWord using a dictionary wordList 
    is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

    Every adjacent pair of words differs by a single letter.
    Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
    sk == endWord
    Given two words, beginWord and endWord, and a dictionary wordList, return the number of words 
    in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence 
    exists.

    

    Example 1:
    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    Output: 5
    Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
    
    Example 2:
    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
    Output: 0
    Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
    

    Constraints:
    1 <= beginWord.length <= 10
    endWord.length == beginWord.length
    1 <= wordList.length <= 5000
    wordList[i].length == beginWord.length
    beginWord, endWord, and wordList[i] consist of lowercase English letters.
    beginWord != endWord
    All the words in wordList are unique.
 */
// BFS
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> uSet(wordList.begin(), wordList.end());
        if(uSet.find(endWord) == uSet.end()) return 0;
        int pathLength = 0;
        queue<string> Q;
        unordered_set<string> visited;
        Q.push(beginWord);
        visited.insert(beginWord);

        while(!Q.empty()) {
            int qSize = Q.size();
            pathLength++;
            for (int k = 0; k < qSize; ++k) {
                string cur = Q.front();
                Q.pop();
                for(int i = 0; i < cur.size(); ++i) {
                    for (int j = 'a'; j <= 'z'; ++j) {
                        if (j == cur[i]) continue;
                        char temp = cur[i];
                        cur[i] = j;
                        if (cur == endWord) return pathLength + 1;
                        if (uSet.find(cur) != uSet.end() && visited.find(cur) == visited.end()) {
                            visited.insert(cur);
                            Q.push(cur);
                        }
                        // Need to revert back.
                        cur[i] = temp;
                    }
                }
            }
        }
        return 0;
    }
};

 /*
    743. Network Delay Time
    https://leetcode.com/problems/network-delay-time/
    You are given a network of n nodes, labeled from 1 to n. You are also given times, a list 
    of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi 
    is the target node, and wi is the time it takes for a signal to travel from source to target.

    We will send a signal from a given node k. Return the minimum time it takes for all the n 
    nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, 
    return -1.

    

    Example 1:
    Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
    Output: 2

    Example 2:
    Input: times = [[1,2,1]], n = 2, k = 1
    Output: 1

    Example 3:
    Input: times = [[1,2,1]], n = 2, k = 2
    Output: -1
    

    Constraints:
    1 <= k <= n <= 100
    1 <= times.length <= 6000
    times[i].length == 3
    1 <= ui, vi <= n
    ui != vi
    0 <= wi <= 100
    All the pairs (ui, vi) are unique. (i.e., no multiple edges.)
 */
// A good problem for Dijkstra/Bellman-ford/SPFA/Floyd. Note how we save the path when running 
// the algorithm!

// Dijkstra impl: Dijkstra can only be used to detect the weighted graph with positive weights.
// It can handle cycles (given every cycle the path weight will be longer). O(|E| + |V|*log|V|)
class Solution {
private:
    vector<vector<pair<int, int>>> buildGraph(vector<vector<int>>& edges, int n) {
        // The entry represents each node index, the pair.first is the neighbors index,
        // the pair.second is the weight.
        vector<vector<pair<int, int>>> G(n+1);
        for (auto& e : edges) {
            G[e[0]].push_back({e[1], e[2]});
        }
        return G;
    }

    void printShortestPath(int currentIndex, const vector<int>& parent) {
        if (currentIndex <= 0 || currentIndex >= parent.size()) return;
        // Reaches the source.
        if (parent[currentIndex] == -1) {
            cout << currentIndex << endl;
            return;
        }
        cout << currentIndex << " >> ";
        printShortestPath(parent[currentIndex], parent);
    }
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        // We have no way to formulate send the signal to all nodes.
        if (times.size() < n-1) return -1;
        // Creates our graph
        vector<vector<pair<int, int>>> G = buildGraph(times, n);

        auto comp = [](pair<int, int> p1, pair<int, int> p2) {
            return p1.second > p2.second;
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(comp)> minQ(comp);
        vector<int> dist(n+1, INT_MAX);
        vector<bool> visited(n+1, false);
        // Save the path if we need to print the path!
        vector<int> parent(n+1, 0);
        // Indicates the starting position.
        parent[k] = -1;

        dist[k] = 0;
        minQ.push({k, dist[k]});

        while(!minQ.empty()) {
            auto minNode = minQ.top();
            minQ.pop();
            visited[minNode.first] = true;
            int distSofar = minNode.second;

            for(int i = 0; i < G[minNode.first].size(); ++i) {
                int next = G[minNode.first][i].first;
                int curDist = G[minNode.first][i].second;
                if (dist[next] > distSofar + curDist) {
                    dist[next] = distSofar + curDist;
                    // Saves the path information!
                    parent[next] = minNode.first;
                    if(!visited[next]) minQ.push({next, dist[next]});
                }
            }
        }

        int res = 0;
        for (int i = 1; i < dist.size(); ++i) {
            res = max(res, dist[i]);
        }

        printShortestPath(n, parent);

        return res == INT_MAX ? -1 : res;
    }
};

// Bellman-ford algorithm with adjacent list impl. O(|V|*|E|)
// Bellman-ford algorithm can work with directed weighted graph with negative weight. It can not
// work with graph with negative cycles. Please note it does not work with un-directed graph with
// negative weight as well, given un-directed graph with negative weight is considered a negative 
// cycle by itself.
class Solution {
private:
    vector<vector<pair<int, int>>> buildGraph(vector<vector<int>>& edges, int n) {
        // The entry represents each node index, the pair.first is the neighbors index,
        // the pair.second is the weight.
        vector<vector<pair<int, int>>> G(n+1);
        for (auto& e : edges) {
            G[e[0]].push_back({e[1], e[2]});
        }
        return G;
    }
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        // We have no way to formulate send the signal to all nodes.
        if (times.size() < n-1) return -1;
        // Creates our graph
        vector<vector<pair<int, int>>> G = buildGraph(times, n);
        // Note we creates n+1 nodes to make the index alignment easier.
        vector<int> dist(n+1, INT_MAX);
        dist[k] = 0;

        // Relax all edges |V| - 1 times. A simple
        // shortest path from src to any other vertex can have
        // at-most |V| - 1 edges
        for (int v = 0; v < n - 1; ++v) {
            // Iterate all edges!
            for (int i = 1; i <= n; ++i) {
                for (int j = 0; j < G[i].size(); ++j) {
                    
                    int nodeSrc = i; 
                    int nodeDst = G[i][j].first;
                    int weight = G[i][j].second;
                    if (dist[nodeSrc] != INT_MAX && 
                        dist[nodeDst] > dist[nodeSrc] + weight) {
                            dist[nodeDst] = dist[nodeSrc] + weight;
                        }
                }
            }
        }

        /* Detect the negative cycle. Not needed for this problem.
        ** We run all the edges one more time, if we can still find even
        ** shorter path weight, we know there is a negative cycle. */
        // Iterate all edges!
        // for (int i = 1; i <= G.size(); ++i) {
        //     for (int j = 0; j < n; ++j) {
        //         int nodeSrc = i;
        //         int nodeDst = G[i][j].first;
        //         int weight = G[i][j].second;
        //         if (dist[nodeSrc] != INT_MAX && 
        //             dist[nodeDst] < dist[nodeSrc] + weight) {
        //                 cout << "We've detected a negative cycle!";
        //                 return -1;
        //             }
        //     }
        // }

        int res = 0;
        // The first node in dist is a dummy node. We will skip it.
        for (int i = 1; i < dist.size(); ++i) {
            cout << i << " the weight is: " << dist[i] << endl;
            res = max(res, dist[i]);
        }
        return res == INT_MAX ? -1 : res;
    }
};

// Bellman-ford algorithm with edge list impl.
// Bellman-ford algorithm with edge list impl.
class Solution {
private:
    void printShortestPath(int currentIndex, const vector<int>& parent) {
        if (currentIndex <= 0 || currentIndex >= parent.size()) return;
        // Reaches the source.
        if (parent[currentIndex] == -1) {
            cout << currentIndex << endl;
            return;
        }
        cout << currentIndex << " >> ";
        printShortestPath(parent[currentIndex], parent);
    }
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        // Note we creates n+1 nodes to make the index alignment easier.
        vector<int> dist(n+1, INT_MAX);
        dist[k] = 0;

        // Saves the path
        vector<int> parent(n+1, 0);
        parent[k] = -1;

        // Relax all edges |V| - 1 times. A simple
        // shortest path from src to any other vertex can have
        // at-most |V| - 1 edges
        for (int v = 0; v < n - 1; ++v) {
            // Iterate all edges!
            for (int i = 0; i < times.size(); ++i){
                int nodeSrc = times[i][0]; 
                int nodeDst = times[i][1];
                int weight = times[i][2];
                if (dist[nodeSrc] != INT_MAX && 
                    dist[nodeDst] > dist[nodeSrc] + weight) {
                        dist[nodeDst] = dist[nodeSrc] + weight;
                        // Saves the path
                        parent[nodeDst] = nodeSrc;
                    }
            }
        }

        /* Detect the negative cycle. Not needed for this problem.
        ** We run all the edges one more time, if we can still find even
        ** shorter path weight, we know there is a negative cycle. */
        // for (int v = 0; v < n - 1; ++v) {
        //     for (int i = 0; i < times.size(); ++i){
        //         int nodeSrc = times[i][0]; 
        //         int nodeDst = times[i][1];
        //         int weight = times[i][2];
        //         if (dist[nodeSrc] != INT_MAX && 
        //             dist[nodeDst] > dist[nodeSrc] + weight) {
        //             cout << "We've detected a negative cycle!";
        //             return -1;
        //         }
        //     }
        // }

        // Print the path!
        printShortestPath(n, parent);

        int res = 0;
        // The first node in dist is a dummy node. We will skip it.
        for (int i = 1; i < dist.size(); ++i) {
            res = max(res, dist[i]);
        }
        return res == INT_MAX ? -1 : res;
    }
};

// Floyd algorithm: It's a dp approach and works with graph with weighted edges.
// If with negative weight, then the graph cannot have negative cycle.
class Solution {
private:
    void printPath(int source, int target, vector<vector<int>>& next) {
        // No path found!
        if (next[source][target] == -1) return;
        cout << source + 1 << "->";
        int nextIndex = source;
        while (nextIndex != -1) {
            nextIndex = next[nextIndex][target];
            cout << nextIndex + 1 << "->";
        }
    }
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        // For floyd algorithm, it's easier to work with a matrix presentation.
        vector<vector<int>> graph(n, vector<int>(n, INT_MAX));

        // Saves the path. -1 means that we do not have a path between i -> j.
        vector<vector<int>> next(n, vector<int>(n, -1));

        for (auto& e : times) {
            graph[e[0]-1][e[1]-1] = e[2];
        }

        // Build path array!
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // No valid path!
                if (graph[i][j] == INT_MAX) continue;
                // We can have one path from i to j.
                next[i][j] = j;
            }
        }

        // Floyd impl: it's a dp approach. We pick up a node k which sits between
        // node source and target, and test whether it can formulate a shortest 
        // path from source to target. We update the shortest path on the fly.
        // After the execution, graph[i][j] saves the shortest path from i to j.
        for (int k = 0; k < n; ++k) {
            // Pick up the source
            for (int i = 0; i < n; ++i) {
                // Pick up the target
                for (int j = 0; j < n; ++j) {
                    // We can find a path from i to k and k to j.
                    if (graph[i][k] != INT_MAX && graph[k][j] != INT_MAX
                    && graph[i][j] > graph[i][k] + graph[k][j]) {
                        graph[i][j] = graph[i][k] + graph[k][j];
                        // Record the next index from i to k!!
                        next[i][j] = next[i][k];
                    }
                }
            }
        }
        
        int res = 0;
        for (int j = 0; j < n; ++j) {
            // Needs to skip the node k itself. 
            // Or we can also initialize graph[i][i] to be 0.
            if (res < graph[k-1][j] && j != k-1) res = max(res, graph[k-1][j]);
        }

        printPath(k-1, n-1, next);

        return res == INT_MAX ? -1 : res;
    }
};

 /*
    1584. Min Cost to Connect All Points
    https://leetcode.com/problems/min-cost-to-connect-all-points/
    You are given an array points representing integer coordinates of some points on a 2D-plane, 
    where points[i] = [xi, yi].

    The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between 
    them: |xi - xj| + |yi - yj|, where |val| denotes the absolute value of val.

    Return the minimum cost to make all points connected. All points are connected if there is 
    exactly one simple path between any two points.

    

    Example 1:
    Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
    Output: 20
    Explanation: 
    We can connect the points as shown above to get the minimum cost of 20.
    Notice that there is a unique path between every pair of points.

    Example 2:
    Input: points = [[3,12],[-2,5],[-4,1]]
    Output: 18
    

    Constraints:
    1 <= points.length <= 1000
    -106 <= xi, yi <= 106
    All pairs (xi, yi) are distinct.
 */
// MST: Kruskal’s algorithm + Union Find with rank
class UnionSet {
private:
    vector<int> parent;
    vector<int> rank;

public:
    UnionSet(int n) {
        parent = vector<int>(n, -1);
        rank = vector<int>(n, 0);
    }

    // Return the parent of node v
    int find(int v) {
        if (parent[v] == -1) return v;

        return find(parent[v]); 
    }

    void unite(int s, int t) {
        int parentS = find(s);
        int parentT = find(t);
        // Already united
        if (parentS == parentT) return;
        
        if (rank[parentS] > rank[parentT]) parent[parentT] = parentS;
        else if (rank[parentS] < rank[parentT]) parent[parentS] = parentT;
        else {
            parent[parentT] = parentS;
            rank[parentT] ++;
        } 
    }

};

class Solution {
private:
    int getDist(vector<int>& p1, vector<int>& p2) {
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]);
    }
public:
    int minCostConnectPoints(vector<vector<int>>& points) {
        int n = points.size();
        // Build the graph with edge representation!
        // [i, j, w]
        // Note using vector<vector<int>> will cause TLE. Use array is much faster!
        vector<array<int, 3>> edges;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                edges.push_back({i, j, getDist(points[i], points[j])});
            }
        }

        // Build Union set
        UnionSet uSet(n);

        auto comp = [](array<int, 3>& e1, array<int, 3>& e2) {
            return e1[2] > e2[2];
        };
        priority_queue<array<int, 3>, vector<array<int, 3>>, decltype(comp)> minQ(comp);
    
        for(int i = 0; i < edges.size(); ++i) {
            minQ.push(edges[i]);
        }
        int res = 0;
        // Kruskal's algorithm:
        /*
        In Kruskal’s algorithm, sort all edges of the given graph in increasing order. 
        Then it keeps on adding new edges and nodes in the MST if the newly added edge 
        does not form a cycle. It picks the minimum weighted edge at first at the 
        maximum weighted edge at last. Thus we can say that it makes a locally 
        optimal choice in each step in order to find the optimal solution.
        */
        while(!minQ.empty()) {
            // We always start with the edge of the minimum weight!
            auto e = minQ.top();
            minQ.pop();
            
            int s = e[0];
            int t = e[1];
            int w = e[2];
            if (uSet.find(s) != uSet.find(t)) {
                uSet.unite(s, t);
                res += w;
            }
        }
        
        return res;
    }
};

// Prim's algorithm
class Solution {
private:
    int getDist(vector<int>& p1, vector<int>& p2) {
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]);
    }
public:
    int minCostConnectPoints(vector<vector<int>>& points) {
        int n = points.size();
        
        // pair.first is the weight, pair.second is the index
        // We need to build a complete graph with both [i, j] & [j, i] to be the path.
        vector<vector<pair<int, int>>> Graph(n);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                Graph[i].push_back({getDist(points[i], points[j]), j});
                Graph[j].push_back({getDist(points[i], points[j]), i});
            }
        }
        // The master priority queue which maintains the next minimum distance.
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minQ;
        
        vector<int> visited(n, false);
        int next = 0;
        int res = 0;
        int count = 0;
        // Here we need to construct n - 1 edges to form a MST.
        while(count < n - 1) {
            visited[next] = true;
            count++;
            

            for (pair<int, int> e :  Graph[next]) {
                // Only adds unvisited nodes to the set.
                // With this for loop, we will include all the edges which connected with
                // current set to the pq.
                if (!visited[e.second])
                    minQ.push(e);
            }
            // cout << count << endl;

            // Remove cycles. Take the following graph as an example:
            /*    / 1 - 3 - 4
                0   |
                  \ 2
                If we first process 0, then we added 1 & 2, then let's say, we process
                2, we will add 1 one more time. Given we need to find the minimum of the 
                unvisited node, the next time when we visit 1, we will get the smallest
                first, then we remove the duplicated 1 immediately!
            */
            while(!minQ.empty() && visited[minQ.top().second]) {
                minQ.pop();
            }
            
            auto nextElement = minQ.top();
            // We can also define a parent[n] array to keep track of the path
            // parent[nextElement.second] = next;
            next = nextElement.second;
            res += nextElement.first;
            minQ.pop();
        }

        return res;
    }
};
